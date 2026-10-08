#!/usr/bin/env python
"""Melt-quench a silica cell with a model and compare the result with AM26 cells made at the same rate.

Protocol, in the NVT ensemble at the density of the starting AM26 cell: Langevin at T_melt for
melt_ps, a linear ramp to T_end at `rate` K/ps, then hold_ps at T_end. The default rate of
100 K/ps is 10^14 K/s, the fastest rate in AM26, so the generated cells can be compared with
AM26's own 10^14 K/s cells descriptor by descriptor. Several seeds give a small population.

A hold that is too cold or too short does not melt the network: the atoms rattle in their
cages and the cell comes back with its starting topology. Every seed therefore records, at the
end of the hold, the mean squared displacement of Si and O against (V/N)^(2/3) and the fraction
of the starting cell's Si-O-Si bridges still present; `network_reset` is true only when fewer
than a fifth survive. At 3000 K for 10 ps more than 90 percent survived. The defaults below
are 5000 K for 20 ps, which did melt it; the first attempt is reproduced with --t-melt 3000 --melt-ps 10.

If more than one model is given, the first drives the dynamics and the others are evaluated
on the saved frames, which gives the force spread of the committee along the quench, after
Beck, Simko, Schaaf, Marsalek and Schran (J. Chem. Phys. 163, 234103, 2025).

Example: the five 5000 K runs behind the README, one per GPU job (hpc/job_mq_melt.pbs), all
writing into one folder; the last job to finish combines them
  python modules/M1_finetune_asio2/melt_quench.py --seeds 1 --first-seed 0       --out modules/M1_finetune_asio2/outputs/melt_quench_5000K \
      --model naive=modules/M1_finetune_asio2/checkpoints/am26_asio2_mpa0_naive.model \
      --model s1=modules/M1_finetune_asio2/checkpoints/am26_asio2_mpa0_naive_s1.model \
      --model s2=modules/M1_finetune_asio2/checkpoints/am26_asio2_mpa0_naive_s2.model
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ase import units
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.neighborlist import neighbor_list

from casebook.io import filter_system, find_key, group_from, read_frames, repo_relative
from casebook.models import load_calculator
from casebook.structure import si_bridges, silica_descriptors

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def parse_model(spec: str) -> tuple[str, dict]:
    label, _, s = spec.partition("=")
    if not s:
        raise SystemExit(f"--model needs label=spec, got {spec!r}")
    if s.startswith("mace_mp:"):
        return label, {"name": label, "loader": "mace_mp", "model": s.split(":", 1)[1]}
    return label, {"name": label, "loader": "file", "path": s}


def min_distance(atoms) -> float:
    i, j, d = neighbor_list("ijd", atoms, 2.5)
    return float(d.min()) if len(d) else float("nan")


def scalar_descriptors(atoms) -> dict[str, float]:
    d = silica_descriptors(atoms, with_rings=True)
    out = {k: float(v) for k, v in d.items() if np.ndim(v) == 0}
    if "ring_pmf" in d:
        pmf = np.asarray(d["ring_pmf"])
        lengths = np.arange(3, 3 + len(pmf))
        out["ring_mean_length_atoms"] = float((pmf * lengths).sum()) if pmf.sum() > 0 else float("nan")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "data" / "am26.extxyz"))
    p.add_argument("--model", action="append", default=None, help="label=spec; the first drives the MD")
    p.add_argument("--group-key", default="label", help="info key holding the quench rate ('auto' to pattern-match key names)")
    p.add_argument("--group-regex", default=r"10-(\d+)", help="regex whose capture group is the numeric rate inside the key")
    p.add_argument("--compare-only", action="store_true", help="skip the dynamics; recompute the AM26 comparison from the final_*.xyz cells already in --out")
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--first-seed", type=int, default=0, help="run seeds first..first+seeds-1, so seeds can go to separate jobs writing to one --out")
    p.add_argument("--t-melt", type=float, default=5000.0)
    p.add_argument("--melt-ps", type=float, default=20.0)
    p.add_argument("--rate-K-per-ps", type=float, default=100.0, help="100 K/ps = 1e14 K/s")
    p.add_argument("--t-end", type=float, default=300.0)
    p.add_argument("--hold-ps", type=float, default=5.0)
    p.add_argument("--dt-fs", type=float, default=1.0)
    p.add_argument("--friction-per-fs", type=float, default=0.01)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--save-traj", action="store_true", help="write the saved frames of each seed to traj_<driver>_seed<k>.xyz, with t_ps and T_set_K in info")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--out", default=str(HERE / "outputs" / "melt_quench"))
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = filter_system(read_frames(a.data), "a-SiO2")
    key = find_key(cells) if a.group_key == "auto" else a.group_key
    vals = np.array([np.nan if key is None else (group_from(c.info.get(key), a.group_regex) or np.nan) for c in cells], dtype=float)
    if key is not None and np.isfinite(vals).any():
        target = float(np.nanmax(vals))
        ref_cells = [c for c, v in zip(cells, vals) if v == target]
        ref_label = f"{key} ~ {a.group_regex} = {target:g} (10^{target:g} K/s)"
    else:
        ref_cells, ref_label = cells, "all a-SiO2 cells (no quench-rate key found)"
    print(f"[MQ] reference group: {ref_label}, {len(ref_cells)} cells")
    ref_desc = pd.DataFrame([scalar_descriptors(c) for c in ref_cells])

    def rebuild() -> int:
        """Combine per-seed summaries and compare every final cell in --out with the AM26 group."""
        per_seed = sorted(out.glob("melt_quench_summary_seed*.csv"))
        if per_seed:
            pd.concat([pd.read_csv(f) for f in per_seed], ignore_index=True).sort_values("seed").to_csv(out / "melt_quench_summary.csv", index=False)
        gen_cells = [read_frames(q)[0] for q in sorted(out.glob("final_*_seed*.xyz"))]
        if not gen_cells:
            print(f"[MQ] no final_*_seed*.xyz in {out}; nothing to compare")
            return 0
        gen = pd.DataFrame([scalar_descriptors(c) for c in gen_cells])
        comp = [{"descriptor": col, "generated_mean": gen[col].mean(), "generated_sd": gen[col].std(ddof=1) if len(gen) > 1 else np.nan,
                 "n_generated": len(gen), "am26_mean": ref_desc[col].mean(), "am26_sd": ref_desc[col].std(ddof=1), "n_am26": len(ref_desc),
                 "am26_group": ref_label} for col in sorted(set(gen.columns) & set(ref_desc.columns))]
        pd.DataFrame(comp).to_csv(out / "melt_quench_vs_am26.csv", index=False)
        print(f"[MQ] comparison from {len(gen_cells)} final cells against {len(ref_cells)} AM26 cells -> {out / 'melt_quench_vs_am26.csv'}")
        return len(gen_cells)

    if a.compare_only:
        if rebuild() == 0:
            raise SystemExit(f"no final_*_seed*.xyz in {out}")
        return 0

    if not a.model:
        raise SystemExit("--model is required unless --compare-only is given")
    models = [parse_model(s) for s in a.model]
    driver_label, driver_entry = models[0]
    driver = load_calculator(driver_entry, device=a.device, dtype=a.dtype, cache_dir=ROOT / "models_cache")
    committee = [(lab, load_calculator(e, device=a.device, dtype=a.dtype, cache_dir=ROOT / "models_cache")) for lab, e in models[1:]]

    ramp_ps = (a.t_melt - a.t_end) / a.rate_K_per_ps
    n_melt, n_ramp, n_hold = (int(round(x * 1000.0 / a.dt_fs)) for x in (a.melt_ps, ramp_ps, a.hold_ps))
    print(f"[MQ] {a.t_melt:.0f} K for {a.melt_ps} ps, ramp {ramp_ps:.1f} ps at {a.rate_K_per_ps} K/ps, hold {a.hold_ps} ps at {a.t_end:.0f} K; {n_melt + n_ramp + n_hold} steps")

    summary, generated = [], []
    for seed in range(a.first_seed, a.first_seed + a.seeds):
        start = ref_cells[seed % len(ref_cells)]
        atoms = start.copy()
        atoms.calc = driver
        rng = np.random.default_rng(seed)
        MaxwellBoltzmannDistribution(atoms, temperature_K=a.t_melt, rng=rng)
        Stationary(atoms)
        dyn = Langevin(atoms, a.dt_fs * units.fs, temperature_K=a.t_melt, friction=a.friction_per_fs / units.fs)
        frames, series, crashed = [], [], None
        t0 = time.perf_counter()
        # melt diagnostics: did the hold erase the starting network? (positions are unwrapped in ASE MD)
        p_start = start.get_positions().copy()
        sym = np.asarray(start.get_chemical_symbols())
        bridges_start = si_bridges(start)
        ell2 = (start.get_volume() / len(start)) ** (2.0 / 3.0)
        melt_diag: dict = {}

        def target_T(step: int) -> float:
            if step < n_melt:
                return a.t_melt
            if step < n_melt + n_ramp:
                return a.t_melt - a.rate_K_per_ps * (step - n_melt) * a.dt_fs / 1000.0
            return a.t_end

        def observe():
            T_set = target_T(dyn.nsteps)
            dyn.set_temperature(temperature_K=T_set)
            if dyn.nsteps == n_melt and not melt_diag:
                d2 = ((atoms.get_positions() - p_start) ** 2).sum(axis=1)
                melt_diag.update({"msd_Si_end_hold_A2": float(d2[sym == "Si"].mean()), "msd_O_end_hold_A2": float(d2[sym == "O"].mean()),
                                  "bridges_kept_end_hold": len(bridges_start & si_bridges(atoms)) / max(len(bridges_start), 1)})
            if dyn.nsteps % a.save_every == 0:
                frames.append(atoms.copy())
                series.append({"step": dyn.nsteps, "t_ps": dyn.nsteps * a.dt_fs / 1000.0, "T_set_K": T_set,
                               "T_K": atoms.get_kinetic_energy() / len(atoms) / (1.5 * units.kB),
                               "epot_eV_atom": atoms.get_potential_energy() / len(atoms)})

        dyn.attach(observe, interval=1)
        try:
            dyn.run(n_melt + n_ramp + n_hold)
        except Exception as exc:
            crashed = str(exc)
        wall = time.perf_counter() - t0
        pd.DataFrame(series).to_csv(out / f"series_{driver_label}_seed{seed}.csv", index=False)
        row = {"driver": driver_label, "seed": seed, "finished": crashed is None, "error": crashed or "",
               "steps_done": dyn.nsteps, "wall_s": wall, "ms_per_step": 1000.0 * wall / max(dyn.nsteps, 1),
               "dmin_end_A": min_distance(atoms), "start_label": str(start.info.get("label", "")),
               "t_melt_K": a.t_melt, "melt_ps": a.melt_ps, "ell2_A2": ell2, **melt_diag,
               "bridges_kept_end": len(bridges_start & si_bridges(atoms)) / max(len(bridges_start), 1)}
        row["network_reset"] = bool(row.get("bridges_kept_end_hold", 1.0) < 0.2)
        if crashed is None:
            desc = scalar_descriptors(atoms)
            row.update({f"{k}_end": v for k, v in desc.items()})
            generated.append(desc)
            write(str(out / f"final_{driver_label}_seed{seed}.xyz"), atoms, format="extxyz")
        summary.append(row)
        pd.DataFrame([row]).to_csv(out / f"melt_quench_summary_seed{seed}.csv", index=False)  # one file per seed, safe for parallel jobs
        if a.save_traj and frames:
            for fr, s in zip(frames, series):
                fr.info["t_ps"] = s["t_ps"]
                fr.info["T_set_K"] = s["T_set_K"]
            write(str(out / f"traj_{driver_label}_seed{seed}.xyz"), frames, format="extxyz")
        print(f"[MQ] seed {seed}: {'ok' if crashed is None else 'CRASHED'} after {dyn.nsteps} steps, {row['ms_per_step']:.1f} ms/step; "
              f"end of hold: MSD Si {row.get('msd_Si_end_hold_A2', float('nan')):.2f} A2 (l^2 = {ell2:.2f}), "
              f"bridges kept {row.get('bridges_kept_end_hold', float('nan')):.3f}", flush=True)

        if committee and frames:
            spread_rows = []
            for fr, s in zip(frames, series):
                preds = []
                for lab, calc in [(driver_label, driver)] + committee:
                    g = fr.copy()
                    g.calc = calc
                    preds.append(np.asarray(g.get_forces()))
                stack = np.stack(preds)
                spread = np.linalg.norm(stack.std(axis=0, ddof=1), axis=1)
                spread_rows.append({"t_ps": s["t_ps"], "T_set_K": s["T_set_K"], "spread_mean_eV_A": float(spread.mean()),
                                    "spread_p95_eV_A": float(np.percentile(spread, 95)), "spread_max_eV_A": float(spread.max())})
            pd.DataFrame(spread_rows).to_csv(out / f"committee_along_quench_seed{seed}.csv", index=False)

    rebuild()  # combines every seed's summary and final cell found in --out, including other jobs'
    (out / "melt_quench_run.json").write_text(json.dumps({
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "driver": a.model[0].split("=", 1)[0] + "=" + repo_relative(a.model[0].split("=", 1)[-1]),
        "committee": a.model[1:], "seeds": a.seeds, "first_seed": a.first_seed, "t_melt_K": a.t_melt, "melt_ps": a.melt_ps, "rate_K_per_ps": a.rate_K_per_ps,
        "group_key": key, "group_regex": a.group_regex,
        "t_end_K": a.t_end, "hold_ps": a.hold_ps, "dt_fs": a.dt_fs, "ensemble": "NVT Langevin at the AM26 cell density",
        "reference_group": ref_label, "device": a.device, "dtype": a.dtype}, indent=2) + "\n", encoding="utf-8")
    print(f"[MQ] wrote {out / 'melt_quench_summary.csv'} and melt_quench_vs_am26.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

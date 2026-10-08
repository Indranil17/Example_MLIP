#!/usr/bin/env python
"""Short NVT runs on one held-out cell: does the model stay physical when it has to move atoms?

A force RMSE on static frames says nothing about what happens over 20 ps. This script runs
Langevin dynamics at one or more temperatures with each model, logs potential energy per
atom and temperature, and compares coordination statistics and the shortest interatomic
distance at the start and the end. The summary line per (model, temperature) is what the
README quotes.

Example
  python modules/M1_finetune_asio2/md_stability.py \
      --test data/splits/a-SiO2/test.xyz --frame 0 --temps 300 1500 --ps 20 \
      --model zero-shot=models_cache/mace-mpa-0-medium.model \
      --model naive=modules/M1_finetune_asio2/checkpoints/am26_asio2_mpa0_naive.model
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
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.neighborlist import neighbor_list

from casebook.io import read_frames, repo_relative, system_label
from casebook.models import load_calculator
from casebook.structure import DEFAULT_CUTOFFS, coordination_fraction, coordination_numbers

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


def coordination_summary(atoms, system: str) -> dict[str, float]:
    sym = atoms.get_chemical_symbols()
    if system == "a-SiO2":
        c = DEFAULT_CUTOFFS["a-SiO2"]
        cn = coordination_numbers(atoms, {"Si-O": c["Si-O"], "O-Si": c["O-Si"]}, bonded_only={"Si": "O", "O": "Si"})
        return {"frac_Si_CN4": coordination_fraction(cn, sym, "Si", 4), "frac_O_CN2": coordination_fraction(cn, sym, "O", 2)}
    if system == "a-C":
        cn = coordination_numbers(atoms, {"C-C": DEFAULT_CUTOFFS["a-C"]["C-C"]})
        return {"frac_sp2": coordination_fraction(cn, sym, "C", 3), "frac_sp3": coordination_fraction(cn, sym, "C", 4)}
    return {}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", default=str(ROOT / "data" / "splits" / "a-SiO2" / "test.xyz"))
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--model", action="append", required=True)
    p.add_argument("--temps", type=float, nargs="+", default=[300.0, 1500.0])
    p.add_argument("--ps", type=float, default=20.0)
    p.add_argument("--dt-fs", type=float, default=1.0)
    p.add_argument("--friction-per-fs", type=float, default=0.01)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(HERE / "outputs" / "md_stability"))
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    start_atoms = read_frames(a.test)[a.frame]
    system = system_label(start_atoms)
    n_steps = int(round(a.ps * 1000.0 / a.dt_fs))
    print(f"[MD] {system}, cell of {len(start_atoms)} atoms from frame {a.frame} of {a.test}; {n_steps} steps of {a.dt_fs} fs")

    rows = []
    for spec in a.model:
        label, entry = parse_model(spec)
        calc = load_calculator(entry, device=a.device, dtype=a.dtype, cache_dir=ROOT / "models_cache")
        for T in a.temps:
            atoms = start_atoms.copy()
            atoms.calc = calc
            rng = np.random.default_rng(a.seed)
            MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=rng)
            Stationary(atoms)
            dyn = Langevin(atoms, a.dt_fs * units.fs, temperature_K=T, friction=a.friction_per_fs / units.fs)
            series = []
            t0 = time.perf_counter()
            crashed = None

            def log():
                epot = atoms.get_potential_energy() / len(atoms)
                ekin = atoms.get_kinetic_energy() / len(atoms)
                series.append({"step": dyn.nsteps, "t_ps": dyn.nsteps * a.dt_fs / 1000.0, "epot_eV_atom": epot,
                               "T_K": ekin / (1.5 * units.kB), "fmax_eV_A": float(np.abs(atoms.get_forces()).max())})

            dyn.attach(log, interval=a.log_every)
            start_cn = coordination_summary(atoms, system)
            start_dmin = min_distance(atoms)
            try:
                dyn.run(n_steps)
            except Exception as exc:  # a blown-up trajectory is a result, not a bug
                crashed = str(exc)
            wall = time.perf_counter() - t0
            df = pd.DataFrame(series)
            df.to_csv(out / f"series_{label}_{int(T)}K.csv", index=False)
            second_half = df[df["t_ps"] >= df["t_ps"].max() / 2.0] if len(df) else df
            drift = float(np.polyfit(second_half["t_ps"], second_half["epot_eV_atom"], 1)[0] * 1000.0) if len(second_half) > 2 else float("nan")
            end_cn = coordination_summary(atoms, system)
            row = {"model": label, "T_K": T, "ps_completed": float(df["t_ps"].max()) if len(df) else 0.0, "ps_requested": a.ps,
                   "finished": crashed is None, "error": crashed or "", "wall_s": wall, "ms_per_step": 1000.0 * wall / max(dyn.nsteps, 1),
                   "epot_drift_second_half_meV_atom_per_ps": drift,
                   "T_mean_second_half_K": float(second_half["T_K"].mean()) if len(second_half) else float("nan"),
                   "fmax_max_eV_A": float(df["fmax_eV_A"].max()) if len(df) else float("nan"),
                   "dmin_start_A": start_dmin, "dmin_end_A": min_distance(atoms)}
            for k, v in start_cn.items():
                row[f"{k}_start"] = v
            for k, v in end_cn.items():
                row[f"{k}_end"] = v
            rows.append(row)
            pd.DataFrame(rows).to_csv(out / "md_stability_summary.csv", index=False)  # written after every run, so a walltime kill keeps what finished
            print(f"[MD] {label} @ {T:.0f} K: {'ok' if crashed is None else 'CRASHED'} after {row['ps_completed']:.1f} ps, "
                  f"drift {drift:.2f} meV/atom/ps, dmin {start_dmin:.2f}->{row['dmin_end_A']:.2f} A, {row['ms_per_step']:.1f} ms/step", flush=True)

    pd.DataFrame(rows).to_csv(out / "md_stability_summary.csv", index=False)
    (out / "md_stability_run.json").write_text(json.dumps({
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "test": repo_relative(a.test), "frame": a.frame,
        "system": system, "n_atoms": len(start_atoms), "temps_K": a.temps, "ps": a.ps, "dt_fs": a.dt_fs,
        "thermostat": f"Langevin, friction {a.friction_per_fs} per fs", "device": a.device, "dtype": a.dtype,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[MD] wrote {out / 'md_stability_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""M0: zero-shot triage of MACE foundation models on AM26 systems.

For every system and model: energies and forces on every cell, per-element force RMSE,
per-atom energy errors, relative RMSE, and seconds per frame. With two or more models the
inter-model force spread is compared with the error of the model mean (Spearman rho).

Outputs (modules/M0_foundation_triage/outputs/)
  metrics.csv                      one row per (system, model)
  per_frame_<system>_<model>.csv   per-frame energies and force RMSE
  spread_vs_error_<system>.csv     per-atom spread and error
  spread_vs_error_<system>.png     the plot, if matplotlib is installed
  run_summary.json                 versions, device, dtype, timestamp, models used
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casebook.io import filter_system, read_frames, reference_energy, reference_forces
from casebook.metrics import (
    energy_per_atom_errors,
    force_component_errors,
    per_element_force_rmse,
    spread_vs_error,
)
from casebook.models import load_calculator, load_registry

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def evaluate(calc, frames):
    pred_e, pred_f, secs = [], [], []
    for a in frames:
        atoms = a.copy()  # drops the reference calculator, keeps info/arrays
        atoms.calc = calc
        t0 = time.perf_counter()
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        secs.append(time.perf_counter() - t0)
        pred_e.append(float(e))
        pred_f.append(np.asarray(f, dtype=float))
    return np.asarray(pred_e), pred_f, np.asarray(secs)


def safe(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "data" / "am26.extxyz"))
    p.add_argument("--systems", nargs="+", default=["a-SiO2", "a-C"])
    p.add_argument("--models", default=str(HERE / "models.yaml"))
    p.add_argument("--only", nargs="*", default=None, help="subset of model names from the registry")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--max-frames", type=int, default=None, help="limit frames per system (smoke tests)")
    p.add_argument("--out", default=str(HERE / "outputs"))
    p.add_argument("--energy-key", default=None)
    p.add_argument("--forces-key", default=None)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = read_frames(a.data)
    registry = load_registry(a.models)
    if a.only:
        registry = [m for m in registry if m["name"] in set(a.only)]

    rows = []
    summary = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "device": a.device, "dtype": a.dtype, "dispersion": False,
        "models": [{k: m.get(k) for k in ("name", "loader", "model", "url", "head", "licence")} for m in registry],
        "systems": {},
    }
    try:
        import mace, torch  # noqa: F401
        summary["mace_torch_version"] = getattr(mace, "__version__", "unknown")
        summary["torch_version"] = torch.__version__
    except Exception as exc:  # pragma: no cover
        summary["versions_error"] = str(exc)

    for system in a.systems:
        fs = filter_system(frames, system)
        if a.max_frames:
            fs = fs[: a.max_frames]
        if not fs:
            print(f"[M0] no frames for {system}; skipping")
            continue
        ref_e = np.array([reference_energy(f, a.energy_key) for f in fs])
        ref_f_list = [reference_forces(f, a.forces_key) for f in fs]
        ref_f = np.concatenate(ref_f_list)
        symbols = np.concatenate([np.asarray(f.get_chemical_symbols()) for f in fs])
        natoms = np.array([len(f) for f in fs])
        summary["systems"][system] = {"n_frames": int(len(fs)), "n_atoms_total": int(natoms.sum())}
        print(f"[M0] {system}: {len(fs)} frames, {natoms.sum()} atoms")

        stacks: dict[str, np.ndarray] = {}
        for entry in registry:
            name = entry["name"]
            print(f"[M0]   {name} ...", flush=True)
            try:
                calc = load_calculator(entry, device=a.device, dtype=a.dtype, cache_dir=ROOT / "models_cache")
                pe, pf, secs = evaluate(calc, fs)
            except Exception as exc:
                print(f"[M0]   {name} FAILED: {exc}")
                rows.append({"system": system, "model": name, "licence": entry["licence"], "status": f"failed: {exc}"})
                continue
            pred_f = np.concatenate(pf)
            stacks[name] = pred_f
            row = {"system": system, "model": name, "licence": entry["licence"], "status": "ok",
                   "n_frames": len(fs), "n_atoms": int(natoms.sum()), "sec_per_frame": float(secs.mean())}
            row.update(force_component_errors(ref_f, pred_f))
            row.update(energy_per_atom_errors(ref_e, pe, natoms))
            for el, v in per_element_force_rmse(symbols, ref_f, pred_f).items():
                row[f"force_rmse_{el}_meV_A"] = v
            rows.append(row)
            per_frame = pd.DataFrame({
                "frame": np.arange(len(fs)), "natoms": natoms,
                "energy_error_meV_atom": 1000.0 * (pe - ref_e) / natoms,  # AM26 labels themselves are not republished
                "frame_force_rmse_meV_A": [1000.0 * float(np.sqrt(np.mean((pf[i] - ref_f_list[i]) ** 2))) for i in range(len(fs))],
                "sec": secs,
            })
            per_frame.to_csv(out / f"per_frame_{safe(system)}_{safe(name)}.csv", index=False)
            print(f"[M0]   {name}: F RMSE {row['force_rmse_meV_A']:.1f} meV/A, E MAE {row['energy_mae_meV_atom']:.2f} meV/atom, {row['sec_per_frame']:.2f} s/frame")

        if len(stacks) >= 2:
            sve = spread_vs_error(np.stack(list(stacks.values())), ref_f)
            pd.DataFrame({"element": symbols, "spread_eV_A": sve["spread_eV_A"], "error_mean_model_eV_A": sve["error_mean_model_eV_A"]}).to_csv(
                out / f"spread_vs_error_{safe(system)}.csv", index=False)
            summary["systems"][system]["spearman_rho_spread_vs_error"] = sve["spearman_rho"]
            summary["systems"][system]["models_in_spread"] = list(stacks)
            print(f"[M0] {system}: Spearman rho(spread, error) = {sve['spearman_rho']:.3f} over {sve['n_atoms']} atoms")
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(4.5, 4))
                ax.scatter(sve["spread_eV_A"], sve["error_mean_model_eV_A"], s=4, alpha=0.3)
                ax.set_xscale("log"); ax.set_yscale("log")
                ax.set_xlabel("inter-model force spread (eV/Å)")
                ax.set_ylabel("error of model mean (eV/Å)")
                ax.set_title(f"{system}: Spearman rho = {sve['spearman_rho']:.2f}")
                fig.tight_layout()
                fig.savefig(out / f"spread_vs_error_{safe(system)}.png", dpi=150)
                plt.close(fig)
            except Exception as exc:  # plotting is optional
                print(f"[M0] plot skipped: {exc}")

    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[M0] wrote {out / 'metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

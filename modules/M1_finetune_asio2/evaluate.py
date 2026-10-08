#!/usr/bin/env python
"""M1 evaluation: zero-shot versus fine-tuned models on the held-out split.

Example
  python modules/M1_finetune_asio2/evaluate.py \
      --test data/splits/a-SiO2/test.xyz \
      --model zero-shot=mace_mp:medium-mpa-0 \
      --model naive=modules/M1_finetune_asio2/checkpoints/am26_asio2_mpa0_naive.model \
      --model replay=modules/M1_finetune_asio2/checkpoints/am26_asio2_mpa0_replay.model

Each --model is label=spec, where spec is either "mace_mp:<name>" or a path to a .model file.
Writes outputs/heldout_metrics.csv with per-element force RMSE and per-atom energy errors,
and outputs/heldout_per_frame_<label>.csv.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from casebook.io import read_frames, reference_energy, reference_forces, repo_relative
from casebook.metrics import energy_per_atom_errors, force_component_errors, per_element_force_rmse
from casebook.models import load_calculator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def parse_model(spec: str) -> tuple[str, dict]:
    label, _, s = spec.partition("=")
    if not s:
        raise SystemExit(f"--model needs label=spec, got {spec!r}")
    if s.startswith("mace_mp:"):
        return label, {"name": label, "loader": "mace_mp", "model": s.split(":", 1)[1]}
    return label, {"name": label, "loader": "file", "path": s}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", default=str(ROOT / "data" / "splits" / "a-SiO2" / "test.xyz"))
    p.add_argument("--model", action="append", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--out", default=str(HERE / "outputs"))
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = read_frames(a.test)
    ref_e = np.array([reference_energy(f, "REF_energy") for f in frames])
    ref_f_list = [reference_forces(f, "REF_forces") for f in frames]
    ref_f = np.concatenate(ref_f_list)
    symbols = np.concatenate([np.asarray(f.get_chemical_symbols()) for f in frames])
    natoms = np.array([len(f) for f in frames])
    groups = sorted({str(f.info.get("split_group_value")) for f in frames})
    print(f"[M1-eval] {len(frames)} held-out frames; group value(s): {groups}")

    rows = []
    for spec in a.model:
        label, entry = parse_model(spec)
        calc = load_calculator(entry, device=a.device, dtype=a.dtype, cache_dir=ROOT / "models_cache")
        pe, pf = [], []
        for f in frames:
            atoms = f.copy()
            atoms.calc = calc
            pe.append(float(atoms.get_potential_energy()))
            pf.append(np.asarray(atoms.get_forces(), dtype=float))
        pe = np.asarray(pe)
        pred_f = np.concatenate(pf)
        row = {"model": label, "spec": spec.split("=", 1)[1], "n_frames": len(frames), "n_atoms": int(natoms.sum())}
        row.update(force_component_errors(ref_f, pred_f))
        row.update(energy_per_atom_errors(ref_e, pe, natoms))
        for el, v in per_element_force_rmse(symbols, ref_f, pred_f).items():
            row[f"force_rmse_{el}_meV_A"] = v
        rows.append(row)
        pd.DataFrame({
            "frame": np.arange(len(frames)), "natoms": natoms,
            "energy_error_meV_atom": 1000.0 * (pe - ref_e) / natoms,  # AM26 labels themselves are not republished
            "frame_force_rmse_meV_A": [1000.0 * float(np.sqrt(np.mean((pf[i] - ref_f_list[i]) ** 2))) for i in range(len(frames))],
        }).to_csv(out / f"heldout_per_frame_{label}.csv", index=False)
        print(f"[M1-eval] {label}: F RMSE {row['force_rmse_meV_A']:.1f} meV/A, E MAE {row['energy_mae_meV_atom']:.2f} meV/atom")

    pd.DataFrame(rows).to_csv(out / "heldout_metrics.csv", index=False)
    (out / "heldout_summary.json").write_text(json.dumps({
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "test_file": repo_relative(a.test), "heldout_group_values": groups, "device": a.device, "dtype": a.dtype,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[M1-eval] wrote {out / 'heldout_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""M4: how clean are the reference forces? Net-force audit of the AM26 labels.

For a periodic DFT calculation the forces must sum to zero. Whatever is left over is
numerical noise in the labels (SCF, grid, Pulay terms), and it is a lower bound on the
force error any model trained on them can reach. This follows the net-force check that
Kuryla, Berger, Csányi and Michaelides applied to molecular datasets (J. Chem. Phys. 163,
224313, 2025); here it is applied to the AM26 periodic cells, per system.

Outputs (modules/M4_label_quality/outputs/)
  net_force_per_frame.csv   system, frame, natoms, |sum F| in eV/A, |sum F|/N in meV/A, max |F|
  net_force_summary.csv     per system: median, 95th percentile and max of |sum F|/N, fraction above 1 meV/A
  net_force_hist.png        per-system histograms, if matplotlib is installed
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from casebook.io import read_frames, reference_forces, repo_relative, system_label

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "data" / "am26.extxyz"))
    p.add_argument("--systems", nargs="*", default=None, help="default: every system in the file")
    p.add_argument("--threshold-meV-A", type=float, default=1.0)
    p.add_argument("--out", default=str(HERE / "outputs"))
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = read_frames(a.data)
    rows = []
    for k, fr in enumerate(frames):
        system = system_label(fr)
        if a.systems and system not in a.systems:
            continue
        f = reference_forces(fr)
        net = np.linalg.norm(f.sum(axis=0))
        rows.append({"system": system, "frame": k, "natoms": len(fr), "net_force_eV_A": net,
                     "net_force_per_atom_meV_A": 1000.0 * net / len(fr), "max_abs_force_eV_A": float(np.abs(f).max())})
    df = pd.DataFrame(rows)
    df.to_csv(out / "net_force_per_frame.csv", index=False)

    summ = []
    for system, g in df.groupby("system"):
        x = g["net_force_per_atom_meV_A"].to_numpy()
        summ.append({"system": system, "n_frames": len(g), "median_meV_A": float(np.median(x)),
                     "p95_meV_A": float(np.percentile(x, 95)), "max_meV_A": float(x.max()),
                     f"fraction_above_{a.threshold_meV_A:g}_meV_A": float(np.mean(x > a.threshold_meV_A))})
        print(f"[M4] {system:8s} n={len(g):4d}  |sum F|/N  median {np.median(x):.3f}  p95 {np.percentile(x, 95):.3f}  max {x.max():.3f} meV/A")
    pd.DataFrame(summ).to_csv(out / "net_force_summary.csv", index=False)
    (out / "run_summary.json").write_text(json.dumps({
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "data": repo_relative(a.data),
        "n_frames": int(len(df)), "threshold_meV_A": a.threshold_meV_A}, indent=2) + "\n", encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        systems = sorted(df["system"].unique())
        fig, axes = plt.subplots(1, len(systems), figsize=(3.2 * len(systems), 3), squeeze=False)
        for ax, system in zip(axes[0], systems):
            x = df.loc[df["system"] == system, "net_force_per_atom_meV_A"]
            ax.hist(x, bins=30)
            ax.set_title(system)
            ax.set_xlabel("|sum F| / N (meV/Å)")
        axes[0][0].set_ylabel("frames")
        fig.tight_layout()
        fig.savefig(out / "net_force_hist.png", dpi=150)
    except Exception as exc:
        print(f"[M4] plot skipped: {exc}")
    print(f"[M4] wrote {out / 'net_force_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Dynamic story figures with CRISP: coordination along a quench and an animated RDF.

Uses two CRISP routines (Saha, Willimetz, Grajciar, J. Comput. Chem. 47, e70384, 2026):
compute_pairwise_rdf for every frame, and calculate_coordination plus
calculate_avg_percentages for the coordination-number fractions.

  python tools/story_dynamic.py --crisp-src /path/to/CRISP_Project \
      --traj modules/M1_finetune_asio2/outputs/melt_quench_traj/traj_naive_seed0.xyz

Outputs in modules/story/:
  rdf_quench.gif                 Si-O and Si-Si RDFs along the quench; solid = current frame,
                                 dashed = AM26 1e14 K/s average (or the running average with --dashed running)
  fig_E_quench_snapshots.png     three frames of the same, liquid / mid-quench / glass, as a static fallback
  fig_D_coordination.png         Si coordination fractions along the quench (CRISP), T overlaid,
                                 with the generated-versus-AM26 defect fractions printed in the corner
  coordination_along_quench.csv  the numbers behind panel D
  coordination_generated_vs_am26.csv
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from casebook.io import filter_system, group_from, read_frames

ROOT = Path(__file__).resolve().parents[1]
M1 = ROOT / "modules" / "M1_finetune_asio2" / "outputs"


def load_module(crisp_src: str, rel: str, name: str):
    path = Path(crisp_src) / "CRISP" / "data_analysis" / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rdf_pair(fn, atoms, a, b, rmax, nbins):
    sym = np.asarray(atoms.get_chemical_symbols())
    ref = [int(k) for k in np.where(sym == a)[0]]
    tgt = [int(k) for k in np.where(sym == b)[0]]
    g, r = fn(atoms, ref, tgt, rmax, nbins)
    return np.asarray(r), np.asarray(g)


def coordination_fractions(cn_mod, cells, atom, pair, cutoff):
    """CRISP calculate_coordination per cell, then calculate_avg_percentages across cells."""
    data = []
    for c in cells:
        dm = c.get_all_distances(mic=True)
        idx = [int(k) for k in range(len(c))]
        data.append(cn_mod.calculate_coordination(c, dm, idx, atom, {pair: cutoff}))
    pct = cn_mod.calculate_avg_percentages(data)  # {CN: [percent per cell]}
    return {int(k): np.asarray(v) for k, v in pct.items()}


def grouped_coordination(si_t: dict):
    """Collapse per-CN percentages into four lines for plotting: <=2, 3, 4, >=5 oxygens."""
    n = len(next(iter(si_t.values())))
    zero = np.zeros(n)
    low = sum((np.asarray(v) for k, v in si_t.items() if k <= 2), zero)
    high = sum((np.asarray(v) for k, v in si_t.items() if k >= 5), zero)
    return [("Si with ≤2 O", low, "0.55", 1.0), ("Si with 3 O", np.asarray(si_t.get(3, zero)), "C1", 1.2),
            ("Si with 4 O", np.asarray(si_t.get(4, zero)), "C3", 2.0), ("Si with ≥5 O", high, "C0", 1.0)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crisp-src", required=True)
    p.add_argument("--traj", default=str(M1 / "melt_quench_5000K" / "traj_naive_seed4.xyz"))
    p.add_argument("--generated-dir", default=str(M1 / "melt_quench_5000K"))
    p.add_argument("--am26", default=str(ROOT / "data" / "am26.extxyz"))
    p.add_argument("--dashed", choices=["reference", "running"], default="reference")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--out", default=str(ROOT / "modules" / "story"))
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prdf = load_module(a.crisp_src, "prdf.py", "crisp_prdf")
    cn = load_module(a.crisp_src, "cn_cal.py", "crisp_cn")
    fn = prdf.compute_pairwise_rdf

    silica = filter_system(read_frames(a.am26), "a-SiO2")
    rate = np.array([group_from(c.info.get("label"), r"10-(\d+)") for c in silica], dtype=float)
    ref = [c for c, v in zip(silica, rate) if v == rate.max()]
    gen = [read_frames(q)[0] for q in sorted(Path(a.generated_dir).glob("final_*_seed*.xyz"))]

    # ---- coordination: generated vs AM26 (CRISP) ----
    rows = []
    for label, cells in (("generated", gen), ("AM26 1e14 K/s", ref)):
        si = coordination_fractions(cn, cells, "Si", ("Si", "O"), 2.0)
        ox = coordination_fractions(cn, cells, "O", ("O", "Si"), 2.0)
        rows.append({"set": label, "n_cells": len(cells),
                     "Si_CN4_percent": float(si.get(4, np.zeros(1)).mean()), "Si_not4_percent": float(100 - si.get(4, np.zeros(1)).mean()),
                     "O_CN2_percent": float(ox.get(2, np.zeros(1)).mean()), "O_not2_percent": float(100 - ox.get(2, np.zeros(1)).mean())})
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(out / "coordination_generated_vs_am26.csv", index=False)
    print(cmp_df.to_string(index=False))

    # ---- trajectory ----
    traj = Path(a.traj)
    if not traj.exists():
        print(f"[dynamic] no trajectory at {traj}; coordination comparison written, animation skipped")
        return 0
    frames = read_frames(traj)
    t = np.array([f.info.get("t_ps", np.nan) for f in frames], dtype=float)
    T = np.array([f.info.get("T_set_K", np.nan) for f in frames], dtype=float)
    print(f"[dynamic] {len(frames)} frames, {t[0]:.1f} to {t[-1]:.1f} ps")

    # coordination along the quench (CRISP), Si
    si_t = coordination_fractions(cn, frames, "Si", ("Si", "O"), 2.0)
    cns = sorted(si_t)
    cn_df = pd.DataFrame({"t_ps": t, "T_set_K": T, **{f"Si_CN{k}_percent": si_t[k] for k in cns}})
    cn_df.to_csv(out / "coordination_along_quench.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    for label, series, color, lw in grouped_coordination(si_t):
        ax.plot(t, series, lw=lw, color=color, label=label)
    ax.set_xlabel("melt-quench time (ps)")
    ax.set_ylabel("fraction of Si atoms (%)")
    ax.set_ylim(0, 102)
    ax2 = ax.twinx()
    ax2.plot(t, T, color="0.5", lw=1, ls=":")
    ax2.set_ylabel("T (K)", color="0.4")
    ax2.set_ylim(0, 1.12 * float(np.nanmax(T)))
    ax.legend(fontsize=8, frameon=False, loc="center right")
    ax.set_title("D  Si coordination along the quench (CRISP)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig_D_coordination.png", dpi=200)
    plt.close(fig)

    # RDFs per frame (CRISP) and the dashed reference
    pairs = (("Si", "O", 1.2, 3.6, 22), ("Si", "Si", 2.4, 6.0, 5.5))
    per_frame = {pr[:2]: [] for pr in pairs}
    r_axis = {}
    nbins = 90  # one 300-atom frame is noisy at finer bins; the reference uses the same grid
    for f in frames:
        for pr in pairs:
            r, g_ = rdf_pair(fn, f, pr[0], pr[1], 6.0, nbins)
            per_frame[pr[:2]].append(g_)
            r_axis[pr[:2]] = r
    ref_mean = {}
    for pr in pairs:
        curves = np.array([rdf_pair(fn, c, pr[0], pr[1], 6.0, nbins)[1] for c in ref])
        ref_mean[pr[:2]] = curves.mean(axis=0)

    def dashed_curve(key, i):
        if a.dashed == "reference":
            return ref_mean[key], "AM26 average, 10$^{14}$ K/s"
        return np.mean(per_frame[key][: i + 1], axis=0), "running average"

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    lines, dashes = [], []
    for ax, pr in zip(axes, pairs):
        key = pr[:2]
        d, dl = dashed_curve(key, 0)
        (dash,) = ax.plot(r_axis[key], d, ls="--", color="0.35", lw=1.4, label=dl)
        (line,) = ax.plot(r_axis[key], per_frame[key][0], color="C3", lw=2.2, label="this frame")
        ax.set_xlim(pr[2], pr[3]); ax.set_ylim(0, pr[4])
        ax.set_xlabel("r (Å)"); ax.set_ylabel(f"g({pr[0]}–{pr[1]})")
        lines.append(line); dashes.append(dash)
    axes[0].legend(fontsize=8, frameon=False, loc="center right")
    title = fig.suptitle("", fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    def update(i):
        for line, dash, pr in zip(lines, dashes, pairs):
            key = pr[:2]
            line.set_ydata(per_frame[key][i])
            if a.dashed == "running":
                dash.set_ydata(dashed_curve(key, i)[0])
        title.set_text(f"Fine-tuned model, heat and quench:  t = {t[i]:5.1f} ps,  T = {T[i]:4.0f} K")
        return lines + dashes + [title]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / a.fps, blit=False)
    anim.save(out / "rdf_quench.gif", writer=PillowWriter(fps=a.fps))
    plt.close(fig)

    # static fallback: three frames
    # end of the hot hold (the liquid), mid-quench, and the final glass; frame 0 is still the starting cell
    hot = np.where(T >= np.nanmax(T) - 1e-6)[0]
    mid_T = 0.5 * (np.nanmax(T) + np.nanmin(T))
    picks = [int(hot[-1]), int(np.argmin(np.abs(T - mid_T) + (np.arange(len(T)) <= hot[-1]) * 1e9)), len(frames) - 1]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6))
    for col, i in enumerate(picks):
        for row, pr in enumerate(pairs):
            key = pr[:2]
            ax = axes[row, col]
            ax.plot(r_axis[key], ref_mean[key], ls="--", color="0.35", lw=1.2)
            ax.plot(r_axis[key], per_frame[key][i], color="C3", lw=2)
            ax.set_xlim(pr[2], pr[3]); ax.set_ylim(0, pr[4])
            if row == 1:
                ax.set_xlabel("r (Å)")
            if col == 0:
                ax.set_ylabel(f"g({pr[0]}–{pr[1]})")
            if row == 0:
                ax.set_title(f"{t[i]:.0f} ps, {T[i]:.0f} K", fontsize=10)
    axes[0, 0].plot([], [], ls="--", color="0.35", label="AM26 average, 10$^{14}$ K/s")
    axes[0, 0].plot([], [], color="C3", lw=2, label="this frame")
    axes[0, 0].legend(fontsize=8, frameon=False, loc="center right")
    fig.suptitle(f"E  {np.nanmax(T):.0f} K to {np.nanmin(T):.0f} K with the fine-tuned model (CRISP PRDF)", x=0.02, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig_E_quench_snapshots.png", dpi=200)
    plt.close(fig)
    print(f"[dynamic] wrote rdf_quench.gif, fig_E_quench_snapshots.png, fig_D_coordination.png in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

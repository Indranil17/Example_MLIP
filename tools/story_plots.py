#!/usr/bin/env python
"""Three small figures for a slide, built from the committed CSVs and the generated cells.

  A  partial RDFs, computed with CRISP, of the cells the fine-tuned model produced by
     melt-quench at 1e14 K/s against the AM26 cells quenched at the same rate
  B  force RMSE before and after fine-tuning for each hold-out
  C  committee spread along the melt-quench against the held-out error; both as per-atom force
     vectors: the spread is |std over seeds|, the error line is sqrt(3) x the per-component RMSE

CRISP (Saha, Willimetz, Grajciar, J. Comput. Chem. 47, e70384, 2026) is used for the RDFs:
either the installed package (pip install crisp-ase) or its prdf module loaded from a source
checkout given with --crisp-src. If neither loads, the figure is still made with a plain
numpy RDF and the panel title says so.

  python tools/story_plots.py --crisp-src /path/to/CRISP_Project
Outputs: modules/story/fig_story.png (three panels) and fig_A_prdf.png, fig_B_bars.png,
fig_C_committee.png separately, plus the CSVs behind panel A.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from casebook.io import filter_system, group_from, read_frames, repo_relative

ROOT = Path(__file__).resolve().parents[1]
M1 = ROOT / "modules" / "M1_finetune_asio2" / "outputs"


def load_crisp_prdf(crisp_src: str | None):
    try:
        from CRISP.data_analysis.prdf import compute_pairwise_rdf  # installed crisp-ase
        return compute_pairwise_rdf, "CRISP"
    except Exception:
        pass
    if crisp_src:
        path = Path(crisp_src) / "CRISP" / "data_analysis" / "prdf.py"
        if path.exists():
            spec = importlib.util.spec_from_file_location("crisp_prdf", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.compute_pairwise_rdf, "CRISP"
    return None, "numpy fallback"


def numpy_prdf(atoms, ref, tgt, rmax, nbins):
    from ase.neighborlist import neighbor_list
    i, j, d = neighbor_list("ijd", atoms, rmax)
    ref_set, tgt_set = set(ref), set(tgt)
    m = np.array([a in ref_set and b in tgt_set for a, b in zip(i, j)])
    hist, edges = np.histogram(d[m], bins=nbins, range=(0, rmax))
    r = 0.5 * (edges[1:] + edges[:-1])
    dr = rmax / nbins
    rho_t = len(tgt) / atoms.get_volume()
    g = hist / (len(ref) * 4 * np.pi * r ** 2 * dr * rho_t)
    return g, r


def prdf_over_cells(cells, a, b, rmax, nbins, fn):
    curves = []
    for at in cells:
        sym = np.asarray(at.get_chemical_symbols())
        ref = [int(k) for k in np.where(sym == a)[0]]
        tgt = [int(k) for k in np.where(sym == b)[0]]
        g, r = fn(at, ref, tgt, rmax, nbins)
        curves.append(np.asarray(g))
    curves = np.array(curves)
    return r, curves.mean(axis=0), curves.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(curves[0])


def panel_a(ax_list, crisp_fn, label, generated_dir: Path, am26: Path, out: Path):
    gen = [read_frames(p)[0] for p in sorted(generated_dir.glob("final_*_seed*.xyz"))]
    silica = filter_system(read_frames(am26), "a-SiO2")
    rate = np.array([group_from(c.info.get("label"), r"10-(\d+)") for c in silica], dtype=float)
    ref = [c for c, v in zip(silica, rate) if v == rate.max()]
    fn = crisp_fn or numpy_prdf
    rows = []
    for ax, (a, b, lo, hi) in zip(ax_list, (("Si", "O", 1.2, 3.6), ("Si", "Si", 2.4, 6.0))):
        r, g_gen, s_gen = prdf_over_cells(gen, a, b, 6.0, 120, fn)
        _, g_ref, s_ref = prdf_over_cells(ref, a, b, 6.0, 120, fn)
        ax.fill_between(r, g_ref - s_ref, g_ref + s_ref, color="0.75", label="AM26, 10$^{14}$ K/s")
        ax.plot(r, g_ref, color="0.3", lw=1.2)
        ax.plot(r, g_gen, color="C3", lw=2.2, label="fine-tuned model")
        ax.set_xlim(lo, hi)
        ax.set_xlabel("r (Å)")
        ax.set_ylabel(f"g({a}–{b})")
        for rr, gg, ss, gr, sr in zip(r, g_gen, s_gen, g_ref, s_ref):
            rows.append({"pair": f"{a}-{b}", "r_A": rr, "generated_mean": gg, "generated_sd": ss, "am26_mean": gr, "am26_sd": sr})
    ax_list[0].legend(fontsize=8, frameon=False, loc="center right")  # r > 2 A is empty in g(Si-O)
    run = generated_dir / "melt_quench_run.json"
    t_melt = json.loads(run.read_text(encoding="utf-8")).get("t_melt_K") if run.exists() else None
    head = f"{t_melt:.0f} K, then 10$^{{14}}$ K/s to 300 K" if t_melt else "model's quench"
    ax_list[0].set_title(f"A  {head}, vs AM26  ({label} PRDF)", loc="left", fontsize=11)
    pd.DataFrame(rows).to_csv(out / "fig_A_prdf.csv", index=False)
    return len(gen), len(ref)


def heldout(path: Path, model_col: str):
    df = pd.read_csv(path)
    zs = float(df.loc[df["model"] == "zero-shot", "force_rmse_meV_A"].iloc[0])
    ft = float(df.loc[df["model"] != "zero-shot", "force_rmse_meV_A"].iloc[0])
    return zs, ft


def panel_b(ax):
    cases = [
        ("silica\nrandom", M1 / "heldout_metrics.csv"),
        ("silica\nfastest\nrate out", M1 / "am26_asio2_mpa0_naive_rate14" / "heldout_metrics.csv"),
        ("silica\nslowest\nrate out", M1 / "am26_asio2_mpa0_naive_rate11" / "heldout_metrics.csv"),
        ("carbon\nporous\nquarter out", M1 / "am26_aC_mpa0_naive" / "heldout_metrics.csv"),
    ]
    labels, zs, ft = [], [], []
    for lab, p in cases:
        if p.exists():
            a, b = heldout(p, "model")
            labels.append(lab); zs.append(a); ft.append(b)
    x = np.arange(len(labels))
    ax.bar(x - 0.18, zs, 0.36, color="0.6", label="zero-shot")
    ax.bar(x + 0.18, ft, 0.36, color="C3", label="fine-tuned")
    for xi, (a, b) in enumerate(zip(zs, ft)):
        ax.text(xi - 0.18, a * 1.08, f"{a:.0f}", ha="center", fontsize=8)
        ax.text(xi + 0.18, b * 1.08, f"{b:.0f}", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("held-out force RMSE (meV/Å)")
    ax.set_ylim(5, 3000)
    ax.legend(fontsize=8, frameon=False, loc="upper left")  # bars there stay below 100
    ax.set_title("B  What fine-tuning buys", loc="left", fontsize=11)


def panel_c(ax, mq_dir: Path, truth_path: Path):
    files = sorted(mq_dir.glob("committee_along_quench_seed*.csv"))
    # per-atom vector error = sqrt(3) x per-component RMSE, the same footing as the per-atom spread norm
    truth = np.sqrt(3.0) * heldout(truth_path, "model")[1] if truth_path.exists() else None
    for k, f in enumerate(files):
        df = pd.read_csv(f)
        ax.plot(df["t_ps"], 1000 * df["spread_mean_eV_A"], color="C0", alpha=0.8, lw=1.2, label="3-seed spread per atom, mean" if k == 0 else None)
        ax.plot(df["t_ps"], 1000 * df["spread_p95_eV_A"], color="C0", alpha=0.35, lw=0.8, label="95th percentile" if k == 0 else None)
    if truth is not None:
        ax.axhline(truth, color="C3", ls="--", lw=1.5, label=f"held-out error per atom (glass), {truth:.0f}")
    ax.set_ylabel("force disagreement (meV/Å)")
    ax.set_xlabel("melt-quench time (ps)")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 40)
    ax2 = ax.twinx()
    if files:
        df = pd.read_csv(files[0])
        ax2.plot(df["t_ps"], df["T_set_K"], color="0.5", lw=1, ls=":")
        ax2.set_ylabel("T (K)", color="0.4")
        ax2.set_ylim(0, 1.12 * float(df["T_set_K"].max()))
    # legend in the empty band under the curves, right of the t = 0 spike
    ax.legend(fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0.06, 0.0), ncol=1)
    ax.set_title("C  Seed committee vs error", loc="left", fontsize=11)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crisp-src", default=None)
    p.add_argument("--am26", default=str(ROOT / "data" / "am26.extxyz"))
    p.add_argument("--mq-dir", default=str(M1 / "melt_quench_5000K"))
    p.add_argument("--out", default=str(ROOT / "modules" / "story"))
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    crisp_fn, label = load_crisp_prdf(a.crisp_src)
    print(f"[story] PRDF engine: {label}")

    # composite
    fig = plt.figure(figsize=(17, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.15, 1.3], wspace=0.42)
    axA1, axA2, axB, axC = (fig.add_subplot(gs[0, k]) for k in range(4))
    n_gen, n_ref = panel_a([axA1, axA2], crisp_fn, label, Path(a.mq_dir), Path(a.am26), out)
    panel_b(axB)
    panel_c(axC, Path(a.mq_dir), M1 / "heldout_metrics.csv")
    fig.savefig(out / "fig_story.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # singles
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    panel_a(list(axes), crisp_fn, label, Path(a.mq_dir), Path(a.am26), out)
    fig.tight_layout(); fig.savefig(out / "fig_A_prdf.png", dpi=200); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3.8)); panel_b(ax); fig.tight_layout(); fig.savefig(out / "fig_B_bars.png", dpi=200); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 3.8)); panel_c(ax, Path(a.mq_dir), M1 / "heldout_metrics.csv"); fig.tight_layout(); fig.savefig(out / "fig_C_committee.png", dpi=200); plt.close(fig)

    (out / "fig_story.json").write_text(json.dumps({"prdf_engine": label, "generated_cells": n_gen, "am26_reference_cells": n_ref,
                                                    "mq_dir": repo_relative(a.mq_dir)}, indent=2) + "\n", encoding="utf-8")
    print(f"[story] wrote {out / 'fig_story.png'} and the three single panels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

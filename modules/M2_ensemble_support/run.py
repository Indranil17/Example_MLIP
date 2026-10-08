#!/usr/bin/env python
"""M2: Vitriflow's support criterion applied to the AM26 silica (or carbon) cells.

For each group of cells (one quench rate for silica, or all cells if no rate key exists),
compute per-cell descriptors, then R_y(n) = h_n(y)/tau_y over random n-cell subsets and
Q(n) = max_y R_y(n). Report the n at which each descriptor reaches q_conv = 0.2, or that it
is not reached with the cells available, plus a rough n^-1/2 extrapolation.

This re-implements the support condition of Cottom, Delhomme and Olsson (Comput. Mater.
Sci. 275, 115098, 2026) on someone else's structures. It is not a Vitriflow run.

Outputs (modules/M2_ensemble_support/outputs/)
  descriptors_<system>_<group>.csv   per-cell scalars (density, coordination fractions)
  support_curve_<system>_<group>.csv R_median/q25/q75 per descriptor and n, plus Q
  support_summary.csv                per group and descriptor: N available, R at N, n required, n extrapolated
  support_curve_<system>_<group>.png the plot, if matplotlib is installed
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from casebook.io import filter_system, find_key, group_from, read_frames
from casebook.structure import carbon_descriptors, density_g_cm3, silica_descriptors
from casebook.support import Descriptor, Q_CONV_DEFAULT, extrapolate_n, n_required, support_curve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

KINDS = {
    "density": "density",
    "frac_Si_CN4": "fraction", "frac_O_CN2": "fraction", "frac_sp2": "fraction", "frac_sp3": "fraction",
    "SiO_cdf": "cdf", "SiSi_cdf": "cdf", "OO_cdf": "cdf", "CC_cdf": "cdf",
    "ring_pmf": "pmf",
}


def group_of(atoms, key, regex=None):
    if key is None:
        return "all"
    if key == "density":
        return f"rho{density_g_cm3(atoms):.2f}"
    v = group_from(atoms.info.get(key), regex)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v if v is not None else atoms.info.get(key))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "data" / "am26.extxyz"))
    p.add_argument("--system", default="a-SiO2", choices=["a-SiO2", "a-C"])
    p.add_argument("--group-key", default=None, help="info key defining groups; default auto (quench|rate|cool); 'none' for one group")
    p.add_argument("--group-regex", default=None, help="regex whose first capture group holds the numeric group value inside the key; AM26 silica labels: '10-(\\d+)', carbon: 'mq_([0-9.]+)'")
    p.add_argument("--n-resamples", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--q-conv", type=float, default=Q_CONV_DEFAULT)
    p.add_argument("--no-rings", action="store_true")
    p.add_argument("--out", default=str(HERE / "outputs"))
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = filter_system(read_frames(a.data), a.system)
    if not frames:
        raise SystemExit(f"no {a.system} frames in {a.data}")
    key = None if a.group_key == "none" else (a.group_key or find_key(frames))
    print(f"[M2] {a.system}: {len(frames)} cells; group key {key!r}")

    groups: dict[str, list] = defaultdict(list)
    for f in frames:
        groups[group_of(f, key, a.group_regex)].append(f)

    summary_rows = []
    meta = {"timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "system": a.system,
            "group_key": key, "group_regex": a.group_regex, "q_conv": a.q_conv, "n_resamples": a.n_resamples, "seed": a.seed, "groups": {}}
    for gname, cells in sorted(groups.items()):
        print(f"[M2] group {gname}: {len(cells)} cells")
        per_cell = [silica_descriptors(c, with_rings=not a.no_rings) if a.system == "a-SiO2" else carbon_descriptors(c, with_rings=not a.no_rings) for c in cells]
        names = [k for k in per_cell[0] if all(k in d for d in per_cell)]
        scalars = pd.DataFrame({k: [d[k] for d in per_cell] for k in names if np.ndim(per_cell[0][k]) == 0})
        scalars.insert(0, "cell", np.arange(len(cells)))
        scalars.to_csv(out / f"descriptors_{a.system}_{gname}.csv", index=False)

        descs = [Descriptor(k, np.array([d[k] for d in per_cell], dtype=float), KINDS.get(k, "scalar")) for k in names]
        curve = support_curve(descs, n_resamples=a.n_resamples, seed=a.seed)
        curve.insert(0, "group", gname)
        curve.to_csv(out / f"support_curve_{a.system}_{gname}.csv", index=False)

        N = len(cells)
        for d in descs + [Descriptor("Q", np.zeros(N), "max")]:
            sub = curve[(curve["descriptor"] == d.name) & (curve["n"] == N)]
            r_at_n = float(sub["R_median"].iloc[0]) if len(sub) else float("nan")
            need = n_required(curve, d.name, a.q_conv)
            summary_rows.append({
                "system": a.system, "group": gname, "descriptor": d.name, "kind": d.kind, "N_available": N,
                "R_at_N": r_at_n, "reached_q_conv": need is not None,
                "n_required": need, "n_extrapolated_sqrt": extrapolate_n(curve, d.name, a.q_conv),
            })
            print(f"[M2]   {d.name:12s} R(N={N}) = {r_at_n:7.3f}   {'reached at n=' + str(need) if need else 'NOT reached with ' + str(N) + ' cells'}")
        meta["groups"][gname] = {"n_cells": N, "descriptors": names, "rings_included": "ring_pmf" in names}

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 4))
            for name in names + ["Q"]:
                sub = curve[curve["descriptor"] == name]
                ax.plot(sub["n"], sub["R_median"], marker="o", ms=3, lw=2 if name == "Q" else 1, label=name, color="k" if name == "Q" else None)
            ax.axhline(a.q_conv, ls="--", color="grey", label=f"q_conv = {a.q_conv}")
            ax.set_yscale("log"); ax.set_xlabel("cells n"); ax.set_ylabel("R_y(n) = h_n / tau_y")
            ax.set_title(f"{a.system}, group {gname}: {N} cells")
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(out / f"support_curve_{a.system}_{gname}.png", dpi=150)
            plt.close(fig)
        except Exception as exc:
            print(f"[M2] plot skipped: {exc}")

    summary = pd.DataFrame(summary_rows)
    path = out / "support_summary.csv"
    if path.exists():  # keep rows of other systems
        old = pd.read_csv(path)
        summary = pd.concat([old[old["system"] != a.system], summary], ignore_index=True)
    summary.to_csv(path, index=False)
    (out / f"run_summary_{a.system}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[M2] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Corrupt the labels of a random fraction of training frames, to measure what bad labels cost.

  python data/inject_noise.py --split data/splits/a-SiO2 --fraction 0.10 --sigma-force-meV-A 300 --sigma-energy-meV-atom 50 --seed 0
  -> data/splits/a-SiO2/train_noisy_f10.xyz and train_noisy_f10.json (which frames were corrupted)

Gaussian noise is added to every force component of the chosen frames and to their total
energy (sigma per atom times the number of atoms). The clean frames are written unchanged.
Validation and test files are not touched. This is the "noisy training without weighting"
baseline of Lam, O'Neill, Schran and Schaaf (arXiv:2602.08849); their down-weighting is not
implemented here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import write

from casebook.io import read_frames


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True)
    p.add_argument("--fraction", type=float, default=0.10)
    p.add_argument("--sigma-force-meV-A", type=float, default=300.0)
    p.add_argument("--sigma-energy-meV-atom", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    split = Path(a.split)
    frames = read_frames(split / "train.xyz")
    rng = np.random.default_rng(a.seed)
    n_bad = max(1, int(round(a.fraction * len(frames))))
    bad = set(rng.choice(len(frames), size=n_bad, replace=False).tolist())
    out_frames = []
    for i, f in enumerate(frames):
        g = f.copy()
        if i in bad:
            g.arrays["REF_forces"] = g.arrays["REF_forces"] + rng.normal(0.0, a.sigma_force_meV_A / 1000.0, size=g.arrays["REF_forces"].shape)
            g.info["REF_energy"] = float(g.info["REF_energy"] + rng.normal(0.0, a.sigma_energy_meV_atom / 1000.0 * len(g)))
            g.info["label_noise"] = 1
        else:
            g.info["label_noise"] = 0
        out_frames.append(g)
    tag = f"f{int(round(100 * a.fraction))}"
    out = split / f"train_noisy_{tag}.xyz"
    write(str(out), out_frames, format="extxyz")
    (split / f"train_noisy_{tag}.json").write_text(json.dumps({
        "source": "train.xyz", "fraction": a.fraction, "n_corrupted": n_bad, "seed": a.seed,
        "sigma_force_meV_A": a.sigma_force_meV_A, "sigma_energy_meV_atom": a.sigma_energy_meV_atom,
        "corrupted_am26_index": sorted(int(frames[i].info["am26_index"]) for i in bad),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[noise] wrote {out}: {n_bad} of {len(frames)} frames corrupted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

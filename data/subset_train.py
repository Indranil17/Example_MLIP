#!/usr/bin/env python
"""Write a random subset of an existing train.xyz, for learning curves.

  python data/subset_train.py --split data/splits/a-SiO2 --n 10 --seed 0
  -> data/splits/a-SiO2/train_n10.xyz and train_n10.json (the am26_index of every frame kept)

The validation and test files are untouched, so every point of the learning curve is judged
on the same held-out cells.
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
    p.add_argument("--split", required=True, help="split directory holding train.xyz")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    split = Path(a.split)
    frames = read_frames(split / "train.xyz")
    if a.n > len(frames):
        raise SystemExit(f"asked for {a.n} frames, train.xyz has {len(frames)}")
    rng = np.random.default_rng(a.seed)
    keep = sorted(rng.choice(len(frames), size=a.n, replace=False).tolist())
    subset = [frames[i] for i in keep]
    out = split / f"train_n{a.n}.xyz"
    write(str(out), subset, format="extxyz")
    (split / f"train_n{a.n}.json").write_text(json.dumps({
        "source": "train.xyz", "n": a.n, "seed": a.seed,
        "am26_index": sorted(int(f.info["am26_index"]) for f in subset),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[subset] wrote {out} with {len(subset)} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

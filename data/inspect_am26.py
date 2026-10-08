#!/usr/bin/env python
"""Print what the AM26 file actually contains before anything is split or trained on it.

Reports: frames per system (by composition), atom counts, density ranges, which keys hold
the reference energy and forces, every info key with how many frames carry it and its most
common values, every per-atom array key, and the info keys that look like a quench rate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from casebook.io import (
    array_key_summary,
    detect_label_keys,
    find_key,
    info_key_summary,
    read_frames,
    system_label,
)
from casebook.structure import density_g_cm3

HERE = Path(__file__).resolve().parent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(HERE / "am26.extxyz"))
    p.add_argument("--max-values", type=int, default=12)
    a = p.parse_args()

    frames = read_frames(a.data)
    print(f"frames: {len(frames)}")

    by_sys: dict[str, list] = defaultdict(list)
    for fr in frames:
        by_sys[system_label(fr)].append(fr)
    print("\nsystems by composition:")
    for s, fs in sorted(by_sys.items()):
        n = np.array([len(f) for f in fs])
        rho = np.array([density_g_cm3(f) for f in fs])
        print(f"  {s:10s} frames {len(fs):5d}  atoms {n.min()}..{n.max()}  density {rho.min():.3f}..{rho.max():.3f} g/cm3")

    print("\nreference label keys (first frame):", detect_label_keys(frames))

    print("\ninfo keys (frames carrying key; most common values):")
    for k, (count, values) in sorted(info_key_summary(frames, a.max_values).items()):
        vals = ", ".join(f"{v} x{c}" for v, c in values.most_common(a.max_values))
        print(f"  {k:24s} {count:5d}   {vals}")

    print("\nper-atom array keys:", dict(array_key_summary(frames)))

    rate_key = find_key(frames)
    print(f"\ncandidate quench-rate key by pattern (quench|rate|cool): {rate_key!r}")
    if rate_key is None:
        print("  none found; make_splits.py will fall back to a random hold-out unless --group-key is given")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

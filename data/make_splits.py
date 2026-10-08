#!/usr/bin/env python
r"""Write train/valid/test extxyz files and a split manifest for one AM26 system.

With a group key the test set is a whole held-out group (one quench rate, or a density band),
so the fine-tune is tested on conditions it never saw. Without one (AM26 has no key named after
the quench rate) it falls back to a random hold-out and the manifest says split_type: random.
The manifest records every source index per split; --verify checks that no index is repeated.

Examples
  python data/make_splits.py --system a-SiO2                       # random 25 percent hold-out
  python data/make_splits.py --system a-SiO2 --group-key label --group-regex '10-(\d+)' --holdout max --out data/splits/a-SiO2_rate14
  python data/make_splits.py --system a-C --group-key density --holdout min --holdout-fraction 0.25
  python data/make_splits.py --verify data/splits/a-SiO2
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase.io import write

from casebook.io import (
    detect_label_keys,
    filter_system,
    find_key,
    group_from,
    read_frames,
    standardise_labels,
)
from casebook.structure import density_g_cm3

HERE = Path(__file__).resolve().parent


def group_value(atoms, key: str | None, regex: str | None = None):
    if key == "density":
        return round(density_g_cm3(atoms), 2)
    if key is None:
        return None
    v = group_from(atoms.info.get(key), regex)
    return v if v is not None else str(atoms.info.get(key))


def verify(split_dir: Path) -> int:
    seen: dict[int, str] = {}
    bad = 0
    for part in ("train", "valid", "test"):
        f = split_dir / f"{part}.xyz"
        if not f.exists():
            print(f"[verify] missing {f}", file=sys.stderr)
            return 2
        for a in read_frames(f):
            idx = int(a.info["am26_index"])
            if idx in seen:
                print(f"[verify] index {idx} in both {seen[idx]} and {part}", file=sys.stderr)
                bad += 1
            seen[idx] = part
    manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    for part in ("train", "valid", "test"):
        on_disk = sorted(i for i, p in seen.items() if p == part)
        if on_disk != sorted(manifest["indices"][part]):
            print(f"[verify] {part}: files and manifest disagree", file=sys.stderr)
            bad += 1
    print("[verify] disjoint and consistent with manifest" if bad == 0 else f"[verify] {bad} problem(s)")
    return 0 if bad == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(HERE / "am26.extxyz"))
    p.add_argument("--system", default="a-SiO2")
    p.add_argument("--group-key", default=None, help="info key for the hold-out group, or 'density' to compute it; default: auto-detect quench|rate|cool")
    p.add_argument("--group-regex", default=None, help="regex whose first capture group holds the numeric group value inside the key; AM26 silica labels: '10-(\\d+)', carbon: 'mq_([0-9.]+)'")
    p.add_argument("--holdout", choices=["max", "min"], default="max", help="hold out the group with the largest or smallest value")
    p.add_argument("--holdout-fraction", type=float, default=0.25, help="with --group-key density: fraction of cells held out as the lowest or highest density band")
    p.add_argument("--valid-fraction", type=float, default=0.15)
    p.add_argument("--random-test-fraction", type=float, default=0.25, help="used only when no group key is found")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="output directory; default data/splits/<system>")
    p.add_argument("--energy-key", default=None)
    p.add_argument("--forces-key", default=None)
    p.add_argument("--verify", default=None, metavar="SPLIT_DIR", help="verify an existing split directory and exit")
    a = p.parse_args()

    if a.verify:
        return verify(Path(a.verify))

    frames = read_frames(a.data)
    sys_frames = filter_system(frames, a.system)
    if not sys_frames:
        print(f"[splits] no frames labelled {a.system}", file=sys.stderr)
        return 2
    index_of = {id(f): i for i, f in enumerate(frames)}
    label_keys = detect_label_keys(sys_frames)

    key = a.group_key or find_key(sys_frames)
    groups: dict = defaultdict(list)
    rng = np.random.default_rng(a.seed)
    split_type = "group"
    if key is None:
        split_type = "random"
        print("[splits] WARNING: no group key found; falling back to a random hold-out. Say so in the README.")
        perm = rng.permutation(len(sys_frames))
        n_test = max(1, int(round(a.random_test_fraction * len(sys_frames))))
        test_idx = set(perm[:n_test].tolist())
        test = [f for i, f in enumerate(sys_frames) if i in test_idx]
        rest = [f for i, f in enumerate(sys_frames) if i not in test_idx]
        heldout_value = None
    elif key == "density":
        # continuous variable: hold out a band (lowest or highest fraction of cells by density)
        split_type = "density_band"
        order = sorted(sys_frames, key=density_g_cm3)
        n_test = max(1, int(round(a.holdout_fraction * len(order))))
        test = order[:n_test] if a.holdout == "min" else order[-n_test:]
        test_ids = {id(f) for f in test}
        rest = [f for f in sys_frames if id(f) not in test_ids]
        heldout_value = {
            "band": a.holdout,
            "fraction": a.holdout_fraction,
            "density_range_g_cm3": [round(density_g_cm3(test[0]), 3), round(density_g_cm3(test[-1]), 3)],
        }
        for f in sys_frames:  # rounded densities, recorded in the manifest only
            groups[group_value(f, key, a.group_regex)].append(f)
        print(f"[splits] density band held out ({a.holdout} {a.holdout_fraction:.0%}): {len(test)} frames, "
              f"{heldout_value['density_range_g_cm3'][0]} to {heldout_value['density_range_g_cm3'][1]} g/cm3")
    else:
        for f in sys_frames:
            groups[group_value(f, key, a.group_regex)].append(f)
        numeric = [g for g in groups if isinstance(g, (int, float))]
        if not numeric:
            print(f"[splits] group key {key!r} has non-numeric values {sorted(map(str, groups))}; cannot pick max/min", file=sys.stderr)
            return 2
        heldout_value = max(numeric) if a.holdout == "max" else min(numeric)
        test = groups[heldout_value]
        rest = [f for g, fs in groups.items() if g != heldout_value for f in fs]
        print(f"[splits] group key {key!r}; groups: " + ", ".join(f"{g}: {len(fs)}" for g, fs in sorted(groups.items(), key=lambda kv: str(kv[0]))))
        print(f"[splits] held out {a.holdout} group {heldout_value} with {len(test)} frames")

    perm = rng.permutation(len(rest))
    n_valid = max(1, int(round(a.valid_fraction * len(rest))))
    valid = [rest[i] for i in perm[:n_valid]]
    train = [rest[i] for i in perm[n_valid:]]

    out = Path(a.out) if a.out else HERE / "splits" / a.system
    out.mkdir(parents=True, exist_ok=True)
    indices: dict[str, list[int]] = {}
    for part, fs in (("train", train), ("valid", valid), ("test", test)):
        written = []
        for f in fs:
            s = standardise_labels(f, a.energy_key, a.forces_key)
            s.info["am26_index"] = index_of[id(f)]
            s.info["am26_system"] = a.system
            if key is not None:
                s.info["split_group_value"] = group_value(f, key, a.group_regex)
            written.append(s)
        write(str(out / f"{part}.xyz"), written, format="extxyz")
        indices[part] = sorted(int(s.info["am26_index"]) for s in written)
        print(f"[splits] {part:5s} {len(written):4d} frames -> {out / (part + '.xyz')}")

    checksums = HERE / "CHECKSUMS.json"
    manifest = {
        "source_file": str(Path(a.data).name),
        "source_sha256": json.loads(checksums.read_text(encoding="utf-8"))["sha256"] if checksums.exists() else None,
        "system": a.system,
        "split_type": split_type,
        "group_key": key,
        "group_regex": a.group_regex,
        "holdout": a.holdout if key is not None else None,
        "heldout_group_value": heldout_value,
        "group_sizes": {str(g): len(fs) for g, fs in groups.items()} if key is not None else None,
        "valid_fraction": a.valid_fraction,
        "seed": a.seed,
        "label_keys_in_source": label_keys,
        "label_keys_written": {"energy_key": "REF_energy", "forces_key": "REF_forces"},
        "counts": {k: len(v) for k, v in indices.items()},
        "indices": indices,
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[splits] manifest -> {out / 'split_manifest.json'}")
    return verify(out)


if __name__ == "__main__":
    raise SystemExit(main())

"""Reading extxyz frames and extracting reference labels without guessing.

The AM26 file is one extxyz with five systems. Its metadata keys were not documented
in the repository README when this was written, so nothing here hard-codes a key name:
systems are labelled from composition, label keys are searched in a fixed order and the
chosen key is reported, and group keys (quench rate, density) are found by pattern and
must be confirmed by the user with ``data/inspect_am26.py``.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from ase import Atoms
from ase.io import read

# Order matters: explicit reference keys first, ASE default names last.
ENERGY_KEYS: tuple[str, ...] = (
    "REF_energy", "ref_energy", "dft_energy", "DFT_energy", "energy_dft",
    "total_energy", "energy", "Energy",
)
FORCES_KEYS: tuple[str, ...] = (
    "REF_forces", "ref_forces", "dft_forces", "DFT_forces", "forces_dft",
    "forces", "Forces", "force",
)

# Composition -> AM26 system label. Used because it needs no metadata.
COMPOSITION_LABELS: dict[frozenset[str], str] = {
    frozenset({"C"}): "a-C",
    frozenset({"Si"}): "a-Si",
    frozenset({"Si", "O"}): "a-SiO2",
    frozenset({"Li", "P", "S"}): "a-LiPS",
    frozenset({"Ge", "Sb", "Te"}): "a-GST",
}

_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def read_frames(path: str | Path, index: str = ":") -> list[Atoms]:
    """Read all frames of an (ext)xyz file as a list of Atoms."""
    frames = read(str(path), index=index)
    if isinstance(frames, Atoms):
        frames = [frames]
    return list(frames)


def system_label(atoms: Atoms, key: str | None = None) -> str:
    """Label a frame by an explicit info key if given, else by composition."""
    if key is not None and key in atoms.info:
        return str(atoms.info[key])
    comp = frozenset(atoms.get_chemical_symbols())
    return COMPOSITION_LABELS.get(comp, "unknown:" + "".join(sorted(comp)))


def filter_system(frames: Iterable[Atoms], system: str, key: str | None = None) -> list[Atoms]:
    return [a for a in frames if system_label(a, key) == system]


def _calc_result(atoms: Atoms, name: str):
    calc = getattr(atoms, "calc", None)
    results = getattr(calc, "results", None)
    if results and name in results:
        return results[name]
    return None


def reference_energy(atoms: Atoms, key: str | None = None) -> float:
    """Total reference energy in eV. Explicit key, else first match in ENERGY_KEYS, else ASE calc."""
    if key is not None:
        if key in atoms.info:
            return float(atoms.info[key])
        raise KeyError(f"energy key {key!r} not in atoms.info; keys: {sorted(atoms.info)}")
    for k in ENERGY_KEYS:
        if k in atoms.info:
            return float(atoms.info[k])
    e = _calc_result(atoms, "energy")
    if e is not None:
        return float(e)
    raise KeyError(f"no reference energy found; info keys: {sorted(atoms.info)}")


def reference_forces(atoms: Atoms, key: str | None = None) -> np.ndarray:
    """Reference forces (N, 3) in eV/Å. Explicit key, else first match in FORCES_KEYS, else ASE calc."""
    if key is not None:
        if key in atoms.arrays:
            return np.asarray(atoms.arrays[key], dtype=float)
        raise KeyError(f"forces key {key!r} not in atoms.arrays; keys: {sorted(atoms.arrays)}")
    for k in FORCES_KEYS:
        if k in atoms.arrays:
            return np.asarray(atoms.arrays[k], dtype=float)
    f = _calc_result(atoms, "forces")
    if f is not None:
        return np.asarray(f, dtype=float)
    raise KeyError(f"no reference forces found; array keys: {sorted(atoms.arrays)}")


def detect_label_keys(frames: Sequence[Atoms]) -> dict[str, str]:
    """Report which energy and force keys the first frame carries, for the manifest."""
    a = frames[0]
    out: dict[str, str] = {}
    for k in ENERGY_KEYS:
        if k in a.info:
            out["energy_key"] = k
            break
    else:
        out["energy_key"] = "ase_calc:energy" if _calc_result(a, "energy") is not None else "none"
    for k in FORCES_KEYS:
        if k in a.arrays:
            out["forces_key"] = k
            break
    else:
        out["forces_key"] = "ase_calc:forces" if _calc_result(a, "forces") is not None else "none"
    return out


def info_key_summary(frames: Sequence[Atoms], max_values: int = 12) -> dict[str, tuple[int, Counter]]:
    """For every info key: how many frames carry it and the most common stringified values."""
    present: Counter = Counter()
    values: dict[str, Counter] = {}
    for a in frames:
        for k, v in a.info.items():
            present[k] += 1
            values.setdefault(k, Counter())[_short(v)] += 1
    return {k: (present[k], Counter(dict(values[k].most_common(max_values)))) for k in present}


def array_key_summary(frames: Sequence[Atoms]) -> Counter:
    c: Counter = Counter()
    for a in frames:
        for k in a.arrays:
            c[k] += 1
    return c


def find_key(frames: Sequence[Atoms], patterns: Sequence[str] = ("quench", "rate", "cool")) -> str | None:
    """First info key whose name contains any pattern (case-insensitive), or None."""
    keys = set()
    for a in frames:
        keys.update(a.info.keys())
    for k in sorted(keys):
        low = k.lower()
        if any(p in low for p in patterns):
            return k
    return None


def parse_float(value) -> float | None:
    """Pull the first number out of a metadata value such as '1e13 K/s' or 1e13."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    m = _FLOAT_RE.search(str(value))
    return float(m.group(0)) if m else None


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_relative(path) -> str:
    """Path relative to the repository root for run records, so no absolute or personal path is written."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except (ValueError, OSError):
        return Path(path).name


def group_from(value, regex: str | None = None):
    """Group value from a metadata field.

    With `regex`, the first capture group (or the whole match) is parsed as a float, so a label
    such as 'silica-mq_10-13_7' with regex '10-(\\d+)' gives 13.0 and 'carbon-mq_2.5_3' with
    'mq_([0-9.]+)' gives 2.5. Without a regex the first number in the value is used, which is
    wrong for labels like these, so pass the regex whenever the key is a composite string.
    """
    if value is None:
        return None
    if regex:
        m = re.search(regex, str(value))
        if not m:
            return None
        tok = m.group(1) if m.groups() else m.group(0)
        v = parse_float(tok)
        return v if v is not None else tok
    return parse_float(value)


def standardise_labels(atoms: Atoms, energy_key: str | None = None, forces_key: str | None = None) -> Atoms:
    """Return a calculator-free copy with labels in info['REF_energy'] and arrays['REF_forces'].

    Training and evaluation scripts then always read the same two keys, whatever the source used.
    """
    e = reference_energy(atoms, energy_key)
    f = reference_forces(atoms, forces_key)
    out = atoms.copy()  # copy() drops the calculator but keeps info and arrays
    out.info["REF_energy"] = e
    out.arrays["REF_forces"] = f
    return out


def _short(v, n: int = 40) -> str:
    s = str(v)
    return s if len(s) <= n else s[: n - 3] + "..."

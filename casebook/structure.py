"""Structural descriptors of one amorphous cell.

Minimal implementations on ASE neighbour lists. The intended production analysis layer is
CRISP (Saha, Willimetz, Grajciar, J. Comput. Chem. 47, e70384, 2026); these functions are
stand-ins that keep this repository runnable without it and will be swapped for CRISP calls.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

AMU_TO_G = 1.66053906660e-24
A3_TO_CM3 = 1.0e-24

# First-shell cutoffs in Å. Values are the usual RDF-minimum choices; confirm on the data
# with an RDF before quoting coordination numbers.
DEFAULT_CUTOFFS: dict[str, dict[str, float]] = {
    "a-SiO2": {"Si-O": 2.0, "O-Si": 2.0, "Si-Si": 3.6, "O-O": 3.2},
    "a-C": {"C-C": 1.85},
    "a-Si": {"Si-Si": 2.85},
}


def density_g_cm3(atoms: Atoms) -> float:
    return float(atoms.get_masses().sum() * AMU_TO_G / (atoms.get_volume() * A3_TO_CM3))


def si_bridges(atoms: Atoms, cutoff: float = 2.0) -> set[tuple[int, int]]:
    """Si-O-Si bridges as pairs of Si atom indices that share an oxygen within `cutoff`.

    Indices are conserved along an MD run, so the overlap of two such sets measures how much
    of a network survived. Periodic images are not distinguished, which is enough for that use.
    """
    i, j = neighbor_list("ij", atoms, cutoff)
    sym = np.asarray(atoms.get_chemical_symbols())
    m = (sym[i] == "O") & (sym[j] == "Si")
    at_o: dict[int, list[int]] = {}
    for o, s in zip(i[m], j[m]):
        at_o.setdefault(int(o), []).append(int(s))
    bridges: set[tuple[int, int]] = set()
    for sis in at_o.values():
        for a in range(len(sis)):
            for b in range(a + 1, len(sis)):
                x, y = sorted((sis[a], sis[b]))
                if x != y:
                    bridges.add((x, y))
    return bridges


def _pair_key(a: str, b: str) -> str:
    return f"{a}-{b}"


def neighbour_table(atoms: Atoms, cutoffs: Mapping[str, float]):
    """Return (i, j, d, sym_i, sym_j) for all pairs within their pair-specific cutoff."""
    rmax = max(cutoffs.values())
    i, j, d = neighbor_list("ijd", atoms, rmax)
    sym = np.asarray(atoms.get_chemical_symbols())
    si, sj = sym[i], sym[j]
    keep = np.zeros(len(i), dtype=bool)
    for n in range(len(i)):
        c = cutoffs.get(_pair_key(si[n], sj[n]))
        if c is None:
            c = cutoffs.get(_pair_key(sj[n], si[n]))
        if c is not None and d[n] <= c:
            keep[n] = True
    return i[keep], j[keep], d[keep], si[keep], sj[keep]


def coordination_numbers(atoms: Atoms, cutoffs: Mapping[str, float], bonded_only: Mapping[str, str] | None = None) -> np.ndarray:
    """Coordination number of every atom, counting neighbours within the pair cutoff.

    bonded_only maps a central species to the only neighbour species that counts, e.g.
    {"Si": "O", "O": "Si"} for silica so Si-Si and O-O contacts are not counted as bonds.
    """
    i, j, d, si, sj = neighbour_table(atoms, cutoffs)
    if bonded_only:
        keep = np.array([bonded_only.get(a, b) == b for a, b in zip(si, sj)], dtype=bool)
        i = i[keep]
    cn = np.bincount(i, minlength=len(atoms))
    return cn


def coordination_fraction(cn: np.ndarray, symbols, species: str, target: int) -> float:
    symbols = np.asarray(symbols)
    m = symbols == species
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(cn[m] == target))


def pair_distance_cdf(atoms: Atoms, a: str, b: str, grid: np.ndarray) -> np.ndarray:
    """Empirical CDF of a-b pair distances inside the window [grid.min, grid.max], values 0 to 1.

    Normalised to the number of a-b pairs found within grid.max, so it is a probability
    distribution of the first-shell distance and a tolerance in percentage points applies to
    it directly, as in Vitriflow. The number of pairs per atom is carried separately by the
    coordination descriptors.
    """
    rmax = float(grid.max())
    i, j, d = neighbor_list("ijd", atoms, rmax)
    sym = np.asarray(atoms.get_chemical_symbols())
    m = (sym[i] == a) & (sym[j] == b)
    dist = np.sort(d[m])
    counts = np.searchsorted(dist, grid, side="right")
    return counts / max(len(dist), 1)


def ring_pmf(atoms: Atoms, cutoff: float, max_ring: int = 12) -> np.ndarray | None:
    """Normalised ring-size distribution from matscipy, lengths 3..max_ring, or None if unavailable.

    Ring lengths are counted in atoms on the full network given to matscipy. For silica this
    means a six-membered Si ring appears as length 12. State this wherever the PMF is shown.
    """
    try:
        from matscipy.rings import ring_statistics
    except Exception:  # matscipy missing or built without the extension
        return None
    counts = np.asarray(ring_statistics(atoms, cutoff, maxlength=max_ring), dtype=float)
    pmf = np.zeros(max_ring + 1)
    pmf[: len(counts)] = counts
    pmf = pmf[3:]
    total = pmf.sum()
    return pmf / total if total > 0 else pmf


def silica_descriptors(atoms: Atoms, cutoffs: Mapping[str, float] | None = None, with_rings: bool = True) -> dict[str, object]:
    """The descriptor set used by M2 for a-SiO2. Returns scalars and vectors keyed by name."""
    cutoffs = dict(DEFAULT_CUTOFFS["a-SiO2"]) if cutoffs is None else dict(cutoffs)
    sym = atoms.get_chemical_symbols()
    cn = coordination_numbers(atoms, {"Si-O": cutoffs["Si-O"], "O-Si": cutoffs["O-Si"]}, bonded_only={"Si": "O", "O": "Si"})
    out: dict[str, object] = {
        "density": density_g_cm3(atoms),
        "frac_Si_CN4": coordination_fraction(cn, sym, "Si", 4),
        "frac_O_CN2": coordination_fraction(cn, sym, "O", 2),
        "SiO_cdf": pair_distance_cdf(atoms, "Si", "O", np.linspace(1.40, 2.00, 31)),
        "SiSi_cdf": pair_distance_cdf(atoms, "Si", "Si", np.linspace(2.60, 3.60, 51)),
        "OO_cdf": pair_distance_cdf(atoms, "O", "O", np.linspace(2.20, 3.20, 51)),
    }
    if with_rings:
        pmf = ring_pmf(atoms, cutoffs["Si-O"], max_ring=16)
        if pmf is not None:
            out["ring_pmf"] = pmf
    return out


def carbon_descriptors(atoms: Atoms, cutoff: float = 1.85, with_rings: bool = True) -> dict[str, object]:
    sym = atoms.get_chemical_symbols()
    cn = coordination_numbers(atoms, {"C-C": cutoff})
    out: dict[str, object] = {
        "density": density_g_cm3(atoms),
        "frac_sp2": coordination_fraction(cn, sym, "C", 3),
        "frac_sp3": coordination_fraction(cn, sym, "C", 4),
        "CC_cdf": pair_distance_cdf(atoms, "C", "C", np.linspace(1.20, 1.85, 27)),
    }
    if with_rings:
        pmf = ring_pmf(atoms, cutoff, max_ring=10)
        if pmf is not None:
            out["ring_pmf"] = pmf
    return out

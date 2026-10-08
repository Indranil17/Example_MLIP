"""Quantity-specific support criterion, after Vitriflow (Cottom, Delhomme, Olsson, 2026).

For a descriptor y measured on n independent cells, with confidence half-width h_n(y) and a
declared resolution scale tau_y, the ratio is R_y(n) = h_n(y) / tau_y, and the population
is supported when Q(n) = max_y R_y(n) <= q_conv, with q_conv = 0.2 by default. Vector
descriptors (CDFs, PMFs) take the maximum ratio over their grid.

Resolution scales follow Vitriflow's silica settings: density max(0.001 g/cm3, 1 percent of
the mean); pair-distance and coordination CDFs 2 percentage points; ring PMF 5 points.
This file re-implements the criterion for use on other people's structures; it does not
reproduce their calibration or screening stages.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

Q_CONV_DEFAULT = 0.2

# kind -> (relative tolerance, absolute floor). Relative applies to |mean|; the larger wins.
DEFAULT_TOLERANCES: dict[str, tuple[float, float]] = {
    "density": (0.01, 0.001),
    "fraction": (0.0, 0.02),
    "cdf": (0.0, 0.02),
    "pmf": (0.0, 0.05),
    "scalar": (0.01, 0.0),
}


@dataclass
class Descriptor:
    name: str
    values: np.ndarray  # (n_cells,) or (n_cells, grid)
    kind: str  # key of DEFAULT_TOLERANCES


def ci_half_width(x: np.ndarray, conf: float = 0.95) -> np.ndarray:
    """t-based confidence half-width over axis 0; inf when fewer than two samples."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if n < 2:
        return np.full(x.shape[1:], np.inf)
    t = stats.t.ppf(0.5 + conf / 2.0, n - 1)
    return t * x.std(axis=0, ddof=1) / np.sqrt(n)


def resolution_scale(mean: np.ndarray, kind: str, tolerances: Mapping[str, tuple[float, float]] | None = None) -> np.ndarray:
    rel, floor = (tolerances or DEFAULT_TOLERANCES)[kind]
    return np.maximum(rel * np.abs(np.asarray(mean, dtype=float)), floor)


def support_ratio(values: np.ndarray, kind: str, conf: float = 0.95, tolerances=None) -> float:
    """R_y for one descriptor on one set of cells: max over the grid of h / tau."""
    values = np.asarray(values, dtype=float)
    h = ci_half_width(values, conf)
    tau = resolution_scale(values.mean(axis=0), kind, tolerances)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(tau > 0, h / tau, np.inf)
    return float(np.nanmax(r))


def support_curve(
    descriptors: Sequence[Descriptor],
    n_values: Sequence[int] | None = None,
    n_resamples: int = 200,
    seed: int = 0,
    conf: float = 0.95,
    tolerances=None,
) -> pd.DataFrame:
    """R_y(n) for each descriptor over random n-cell subsets, plus Q(n).

    Returns a long DataFrame with columns n, descriptor, R_median, R_q25, R_q75, and rows with
    descriptor == 'Q' holding max_y R_median. With N cells available the subsets are drawn
    without replacement; for n == N there is a single subset.
    """
    rng = np.random.default_rng(seed)
    n_cells = int(descriptors[0].values.shape[0])
    for d in descriptors:
        if d.values.shape[0] != n_cells:
            raise ValueError(f"descriptor {d.name} has {d.values.shape[0]} cells, expected {n_cells}")
    if n_values is None:
        n_values = list(range(2, n_cells + 1))
    rows = []
    for n in n_values:
        if n < 2 or n > n_cells:
            continue
        subsets = [np.arange(n_cells)] if n == n_cells else [rng.choice(n_cells, size=n, replace=False) for _ in range(n_resamples)]
        medians = {}
        for d in descriptors:
            rs = np.array([support_ratio(d.values[s], d.kind, conf, tolerances) for s in subsets])
            q25, med, q75 = np.nanpercentile(rs, [25, 50, 75])
            medians[d.name] = med
            rows.append({"n": n, "descriptor": d.name, "kind": d.kind, "R_median": med, "R_q25": q25, "R_q75": q75})
        rows.append({"n": n, "descriptor": "Q", "kind": "max", "R_median": max(medians.values()), "R_q25": np.nan, "R_q75": np.nan})
    return pd.DataFrame(rows)


def n_required(curve: pd.DataFrame, descriptor: str = "Q", q_conv: float = Q_CONV_DEFAULT) -> int | None:
    """Smallest n from which the median ratio stays at or below q_conv for every larger n, or None.

    Requiring it to stay below avoids reporting a zero-width artefact at very small n (two
    identical cells give R = 0) as convergence when R rises again with more cells.
    """
    sub = curve[curve["descriptor"] == descriptor].sort_values("n")
    if not len(sub):
        return None
    ok = (sub["R_median"] <= q_conv).to_numpy()
    if not ok[-1]:
        return None
    first_bad_from_end = len(ok) - int(np.argmin(ok[::-1])) if not ok.all() else 0
    return int(sub["n"].to_numpy()[first_bad_from_end])


def extrapolate_n(curve: pd.DataFrame, descriptor: str, q_conv: float = Q_CONV_DEFAULT) -> float | None:
    """Rough n needed assuming R ~ n^-1/2, fitted on the available points. Report as an estimate only."""
    sub = curve[(curve["descriptor"] == descriptor) & np.isfinite(curve["R_median"])]
    if len(sub) < 3:
        return None
    n = sub["n"].to_numpy(dtype=float)
    r = sub["R_median"].to_numpy(dtype=float)
    # R = c / sqrt(n) -> c = median(R sqrt(n)); n* = (c / q_conv)^2
    c = float(np.median(r * np.sqrt(n)))
    return float((c / q_conv) ** 2)

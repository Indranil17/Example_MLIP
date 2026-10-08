"""Error metrics reported as distributions and per element, never as one pooled number only."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return float(np.mean(np.abs(a - b)))


def rel_rmse(rmse_value: float, reference: np.ndarray) -> float:
    """100 * RMSE / standard deviation of the reference labels.

    AM26 reports a relative RMSE of this form; check its exact definition against the
    paper before quoting the two side by side.
    """
    sd = float(np.std(np.asarray(reference, dtype=float).ravel()))
    return float("nan") if sd == 0 else 100.0 * rmse_value / sd


def force_component_errors(ref: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Component-wise force errors over all atoms of one or many frames (stacked (N, 3))."""
    ref = np.asarray(ref, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = (pred - ref).ravel()
    return {
        "force_rmse_meV_A": 1000.0 * float(np.sqrt(np.mean(err ** 2))),
        "force_mae_meV_A": 1000.0 * float(np.mean(np.abs(err))),
        "force_median_abs_meV_A": 1000.0 * float(np.median(np.abs(err))),
        "force_p95_abs_meV_A": 1000.0 * float(np.percentile(np.abs(err), 95)),
        "force_rel_rmse_percent": rel_rmse(float(np.sqrt(np.mean(err ** 2))), ref),
    }


def per_element_force_rmse(symbols: Sequence[str], ref: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Force RMSE in meV/Å per chemical element."""
    symbols = np.asarray(symbols)
    ref = np.asarray(ref, dtype=float)
    pred = np.asarray(pred, dtype=float)
    out: dict[str, float] = {}
    for el in sorted(set(symbols.tolist())):
        m = symbols == el
        out[el] = 1000.0 * rmse(ref[m], pred[m])
    return out


def energy_per_atom_errors(ref_e: Sequence[float], pred_e: Sequence[float], natoms: Sequence[int]) -> dict[str, float]:
    ref_e = np.asarray(ref_e, dtype=float)
    pred_e = np.asarray(pred_e, dtype=float)
    n = np.asarray(natoms, dtype=float)
    err = (pred_e - ref_e) / n
    return {
        "energy_mae_meV_atom": 1000.0 * float(np.mean(np.abs(err))),
        "energy_rmse_meV_atom": 1000.0 * float(np.sqrt(np.mean(err ** 2))),
        "energy_median_abs_meV_atom": 1000.0 * float(np.median(np.abs(err))),
        "energy_rel_rmse_percent": rel_rmse(float(np.sqrt(np.mean(err ** 2))), ref_e / n),
        "n_frames": int(len(ref_e)),
    }


def spread_vs_error(pred_stack: np.ndarray, ref: np.ndarray) -> dict[str, object]:
    """Inter-model force spread against actual error, per atom.

    pred_stack: (n_models, N, 3) predicted forces; ref: (N, 3) reference forces.
    Spread is the norm of the standard deviation across models; the error is that of the
    model mean. Returns per-atom arrays and the Spearman rank correlation. Report the
    coefficient; do not call disagreement an error estimate.
    """
    pred_stack = np.asarray(pred_stack, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if pred_stack.ndim != 3 or pred_stack.shape[0] < 2:
        raise ValueError("need at least two models stacked as (n_models, N, 3)")
    mean_pred = pred_stack.mean(axis=0)
    spread = np.linalg.norm(pred_stack.std(axis=0, ddof=1), axis=1)
    error = np.linalg.norm(mean_pred - ref, axis=1)
    rho, p = stats.spearmanr(spread, error)
    return {
        "spread_eV_A": spread,
        "error_mean_model_eV_A": error,
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "n_atoms": int(len(spread)),
    }


def bootstrap_rmse(frame_rmse: Sequence[float], natoms: Sequence[int], seed: int = 0, draws: int = 4000) -> tuple[float, float, float]:
    """Pooled per-component force RMSE over frames, with its 95 percent bootstrap interval.

    ``frame_rmse`` holds one RMSE per frame and ``natoms`` the atoms in that frame, so the
    pooled value weights each frame by its atom count, sqrt(sum(n_i r_i^2) / sum(n_i)).
    Frames are resampled with replacement ``draws`` times and the interval is the 2.5th to
    97.5th percentile of the resampled values. Returns (pooled, lower, upper).
    """
    r = np.asarray(frame_rmse, dtype=float)
    n = np.asarray(natoms, dtype=float)
    if r.shape != n.shape or r.ndim != 1 or len(r) == 0:
        raise ValueError("frame_rmse and natoms must be non-empty 1-d sequences of the same length")
    v = r ** 2 * n
    pooled = float(np.sqrt(v.sum() / n.sum()))
    idx = np.random.default_rng(seed).integers(0, len(v), (draws, len(v)))
    boot = np.sqrt(v[idx].sum(axis=1) / n[idx].sum(axis=1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return pooled, float(lo), float(hi)

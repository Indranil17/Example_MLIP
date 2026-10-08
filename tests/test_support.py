import numpy as np

from casebook.support import Descriptor, ci_half_width, n_required, support_curve, support_ratio


def test_half_width_shrinks_with_n():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, size=(400,))
    h_small = ci_half_width(x[:10]).item()
    h_large = ci_half_width(x).item()
    assert h_large < h_small
    assert np.isinf(ci_half_width(x[:1]))


def test_support_ratio_scalar_and_vector():
    rng = np.random.default_rng(2)
    dens = rng.normal(2.2, 0.02, size=64)
    r = support_ratio(dens, "density")
    assert r > 0
    cdf = np.clip(rng.normal(0.5, 0.01, size=(64, 20)), 0, 1)
    r_vec = support_ratio(cdf, "cdf")
    assert np.isfinite(r_vec)


def test_support_curve_reaches_threshold_for_tight_data():
    rng = np.random.default_rng(3)
    d = Descriptor("density", rng.normal(2.2, 0.005, size=64), "density")
    curve = support_curve([d], n_values=[2, 4, 8, 16, 32, 64], n_resamples=50, seed=0)
    assert set(curve["descriptor"]) == {"density", "Q"}
    q = curve[curve["descriptor"] == "Q"].sort_values("n")["R_median"].to_numpy()
    assert q[-1] < q[0]
    assert n_required(curve, "Q", 0.2) is not None


def test_support_curve_not_reached_for_wide_data():
    rng = np.random.default_rng(4)
    d = Descriptor("density", rng.normal(2.2, 0.3, size=12), "density")
    curve = support_curve([d], n_resamples=30, seed=0)
    assert n_required(curve, "Q", 0.2) is None


def test_n_required_ignores_early_zero_width_artefact():
    import pandas as pd

    from casebook.support import n_required

    # R dips to 0 at n = 2, rises above 0.2, and only settles below it from n = 6 on
    curve = pd.DataFrame({"n": [2, 3, 4, 5, 6, 7], "descriptor": ["Q"] * 6, "R_median": [0.0, 0.5, 0.4, 0.3, 0.15, 0.1]})
    assert n_required(curve, "Q", 0.2) == 6
    curve.loc[curve.n == 7, "R_median"] = 0.3
    assert n_required(curve, "Q", 0.2) is None

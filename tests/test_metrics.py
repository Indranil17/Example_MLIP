import numpy as np

from casebook.metrics import (
    energy_per_atom_errors,
    force_component_errors,
    per_element_force_rmse,
    rel_rmse,
    spread_vs_error,
)


def test_zero_error_for_identical_forces():
    f = np.random.default_rng(0).normal(size=(50, 3))
    m = force_component_errors(f, f)
    assert m["force_rmse_meV_A"] == 0.0
    assert m["force_rel_rmse_percent"] == 0.0


def test_per_element_rmse_isolates_elements():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(20, 3))
    pred = ref.copy()
    pred[:10] += 0.1  # only the first ten atoms are wrong
    symbols = ["Si"] * 10 + ["O"] * 10
    per = per_element_force_rmse(symbols, ref, pred)
    assert per["O"] == 0.0
    assert abs(per["Si"] - 100.0) < 1e-9  # 0.1 eV/A = 100 meV/A


def test_energy_per_atom_errors():
    m = energy_per_atom_errors([-10.0, -20.0], [-10.1, -19.8], [100, 100])
    assert abs(m["energy_mae_meV_atom"] - 1.5) < 1e-9
    assert m["n_frames"] == 2


def test_rel_rmse_handles_constant_reference():
    assert np.isnan(rel_rmse(1.0, np.ones(10)))


def test_spread_vs_error_shapes_and_rho_range():
    rng = np.random.default_rng(2)
    ref = rng.normal(size=(100, 3))
    stack = np.stack([ref + rng.normal(scale=s, size=ref.shape) for s in (0.05, 0.1, 0.2)])
    out = spread_vs_error(stack, ref)
    assert out["spread_eV_A"].shape == (100,)
    assert -1.0 <= out["spearman_rho"] <= 1.0


def test_bootstrap_rmse_pools_by_atom_count_and_brackets_the_value():
    from casebook.metrics import bootstrap_rmse

    pooled, lo, hi = bootstrap_rmse([1.0, 3.0], [100, 100], seed=0, draws=300)
    assert abs(pooled - np.sqrt(5.0)) < 1e-12  # sqrt((1 + 9) / 2)
    assert 1.0 <= lo <= pooled <= hi <= 3.0
    same, lo2, hi2 = bootstrap_rmse([2.0, 2.0, 2.0], [10, 20, 30], draws=50)
    assert abs(same - 2.0) < 1e-12 and abs(lo2 - 2.0) < 1e-12 and abs(hi2 - 2.0) < 1e-12

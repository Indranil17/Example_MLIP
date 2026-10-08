import numpy as np
from ase.build import bulk

from casebook.io import system_label
from casebook.structure import coordination_numbers, density_g_cm3, pair_distance_cdf


def test_diamond_silicon_descriptors():
    si = bulk("Si", "diamond", a=5.431).repeat((2, 2, 2))
    assert system_label(si) == "a-Si"  # composition label; the test cell happens to be crystalline
    rho = density_g_cm3(si)
    assert abs(rho - 2.33) < 0.02
    cn = coordination_numbers(si, {"Si-Si": 2.6})
    assert np.all(cn == 4)
    grid = np.linspace(2.2, 2.6, 5)
    cdf = pair_distance_cdf(si, "Si", "Si", grid)
    assert np.all(np.diff(cdf) >= 0)
    assert abs(cdf[-1] - 1.0) < 1e-9  # all pairs inside the window are counted by the top of the grid
    assert cdf[0] == 0.0  # no Si-Si pair below 2.2 A in diamond silicon


def test_si_bridges_count_in_quartz_like_chain():
    from ase import Atoms
    from casebook.structure import si_bridges

    # Si-O-Si-O-Si along x in a big box: two bridges, (0,2) and (2,4)
    a = Atoms("SiOSiOSi", positions=[[0, 0, 0], [1.6, 0, 0], [3.2, 0, 0], [4.8, 0, 0], [6.4, 0, 0]], cell=[30, 30, 30], pbc=True)
    assert si_bridges(a) == {(0, 2), (2, 4)}


def test_bonded_only_filter():
    # two Si and four O on a line: Si-O at 1.6 A counts, Si-Si at 3.2 A does not
    from ase import Atoms

    a = Atoms("SiOOSiOO", positions=[[0, 0, 0], [1.6, 0, 0], [-1.6, 0, 0], [3.2, 0, 0], [4.8, 0, 0], [6.4, 0, 0]],
              cell=[20, 20, 20], pbc=True)
    cn = coordination_numbers(a, {"Si-O": 2.0, "O-Si": 2.0, "Si-Si": 3.6}, bonded_only={"Si": "O", "O": "Si"})
    assert cn[0] == 2 and cn[3] == 2

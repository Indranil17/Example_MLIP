from ase import Atoms

from casebook.io import group_from, repo_relative, system_label


def test_group_from_reads_the_am26_quench_rate_and_density():
    assert group_from("silica-mq_10-13_7", r"10-(\d+)") == 13.0
    assert group_from("carbon-mq_2.5_3", r"mq_([0-9.]+)") == 2.5
    # without the regex the first number in the label is taken, which is the trap the README describes
    assert group_from("silica-mq_10-13_7") == 10.0
    assert group_from(None, r"10-(\d+)") is None


def test_system_label_by_composition_and_repo_relative():
    assert system_label(Atoms("SiO2")) == "a-SiO2"
    assert system_label(Atoms("C2")) == "a-C"
    assert system_label(Atoms("GeTe")).startswith("unknown")
    assert repo_relative("/no/such/place/file.xyz") == "file.xyz"

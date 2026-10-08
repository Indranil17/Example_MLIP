# M3. Vitriflow's silica example, run as shipped

Status: not started. Everything below is the plan.

A calibrated classical melt-quench population of amorphous silica, generated with the
Vitriflow package as its authors ship it. It is the baseline a fine-tuned machine-learned
potential would replace, and a second set of silica cells, without DFT labels, on which the
M0 foundation models can be compared with each other.

The package: Cottom, Delhomme and Olsson, https://github.com/Olsson-Materials-Modelling/Vitriflow,
MIT, Python 3.10 or newer, LAMMPS as the primary engine, CP2K and QUIP/GAP optional. The
version used in their paper is 0.4.37.0, archived at https://doi.org/10.5281/zenodo.21111823.
Four stages: numerical preflight, protocol calibration (`autotune`), population generation
(`run`), analysis (`analyze-output`, `plot-production`). The shipped examples include BKS
silica production YAMLs that need no OpenKIM install.

Planned commands, to be checked against their README at install time; none has been run:

```bash
git clone https://github.com/Olsson-Materials-Modelling/Vitriflow
cd Vitriflow && conda env create -f environment.yml && conda activate vitriflow && pip install -e .
vitriflow run vitriflow/examples/sio2_bks_zbl_smoke.yaml              # smoke test
vitriflow run vitriflow/examples/sio2_bks_packmol_production.yaml     # production, as independent jobs
vitriflow analyze-output <output directory>
vitriflow plot-production <output directory>
```

One member of the shipped silica protocol is a few nanoseconds of classical MD in a
192-atom cell. That should take minutes on a handful of cores, so an overnight batch should
give of the order of a hundred cells with Vitriflow's own audit JSON. This is an estimate, not
a timing.

What goes here afterwards: the configuration as used, the audit JSON, the `analyze-output`
summary, the numbers of cells generated and retained by their screen, and the M0 table on a
subsample of the retained cells (model-to-model spread only, since there are no labels).
The structures themselves are not committed.

Wording: "Vitriflow's published BKS silica example, run as shipped, N cells", and not
"reproduced the paper". Their silica population used the SHIK potential and the full
calibration chain.

Next step, outside this module: driving Vitriflow with a MACE potential through LAMMPS'
explicit pair-style commands. The package demonstrates GAP through QUIP; a MACE engine path
is something to propose to the authors, not something to claim here.

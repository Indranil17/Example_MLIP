# M0. Foundation models on amorphous silica and carbon, no fine-tuning

Status: done, 7 October 2026. The tables are in the Results block of the top-level README.

Two questions. How far do current MACE foundation models get on the AM26 silica and carbon
cells as they come? And when the models disagree with each other on a force, is that where
they are also wrong? The second matters because a committee of foundation models is the
cheapest uncertainty estimate available before any training.

Models, from `models.yaml`: MACE-MPA-0 and MACE-MP-0b3 (MIT) and MACE-MH-1 with its
`omat_pbe` head (Academic Software License, evaluated only). No dispersion correction is
applied. The low-density carbon cells are where that would matter most.

## What came out

- **Silica is close already.** The force RMSE on all 80 silica cells is 49 meV/Å for
  MPA-0, 56 for MP-0b3 and 33 for MH-1, about 3 to 6 percent of the spread of the reference
  forces. MPA-0 is also the closest on energy, at 0.42 meV/atom.
- **Carbon is not.** The force RMSE is 410 to 760 meV/Å, 30 to 56 percent of the spread of
  the reference forces. The energy errors of the two MIT models are almost entirely a
  constant offset: MPA-0 sits 99 meV/atom below the labels on average and MP-0b3 128 below.
  With each model's mean offset removed the energy MAE drops to 13 and 16 meV/atom, against
  10 for MH-1, whose offset is only 6. The forces carry no such offset, so the force error on
  carbon is real.
- **Disagreement tracks error, loosely.** Over all atoms, the Spearman coefficient between
  the force spread of the three models and the error of their mean is 0.51 for silica and
  0.41 for carbon. With the two MIT models only, on the other systems, it is 0.51 for a-LiPS,
  0.22 for a-GST and 0.21 for a-Si. A large spread marks atoms worth looking at. It is not an
  error bar.

MPA-0 is the base model for every fine-tune in M1. It is MIT, so a fine-tuned checkpoint
could be released, and on silica its zero-shot force error is 1.5 times that of MH-1.

## Running it

```bash
python modules/M0_foundation_triage/run.py --systems a-SiO2 a-C --device cpu
python modules/M0_foundation_triage/run.py --systems a-Si a-LiPS a-GST --device cpu \
    --only MACE-MPA-0 MACE-MP-0b3 --out modules/M0_foundation_triage/outputs_other_systems
python modules/M0_foundation_triage/run.py --systems a-SiO2 --max-frames 5 --only MACE-MPA-0   # smoke test
```

The remaining AM26 systems are run with the two MIT models into `outputs_other_systems/`,
kept separate so the main table stays about the two systems the rest of the repository uses.

Outputs in `outputs/`: `metrics.csv` with one row per system and model (force RMSE, MAE,
median and 95th percentile of the absolute component error, relative RMSE, per-atom energy
MAE and RMSE, per-element force RMSE, seconds per frame); `run_summary.json` with the model
URLs, the versions and the Spearman coefficients; `per_frame_*.csv` with the force RMSE and
the energy error of every cell; `spread_vs_error_<system>.csv` and `.png`.

## Reading the table

The relative RMSE here is 100 × RMSE / σ, with σ the standard deviation of the reference
force components of that system. AM26 also quotes relative errors. I have not checked that
their definition is the same, so the two should not be put side by side. The energy offset
and the MAE after removing it are computed from the per-frame files by
`tools/make_results_table.py`.

Wall-clock and hardware: one CPU node, 8 cores, float64. The silica and carbon triage took
25 minutes together with the first M2 pass, at 1.9 to 3.3 s per frame depending on the model.

What this does not show: accuracy against experiment (labels and MIT models are both PBE)
and stability in dynamics (that is the MD check in M1).

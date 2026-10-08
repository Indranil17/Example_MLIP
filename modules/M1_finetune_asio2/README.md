# M1. Fine-tuning MACE-MPA-0 on AM26 silica and carbon

Status: done, 8 October 2026, except the full-length multihead replay run. Its 12 h walltime
stopped it at epoch 32 of 150, before the evaluation step. A 30-epoch replay run with a
1000-sample replay set was evaluated instead; it is not comparable with the 150-epoch naive
runs and is reported as a smoke test of the replay set-up.

## Results in brief

All numbers come from the CSVs in `outputs/` and are tabulated, with bootstrap intervals, in
the Results block of the top-level README. Force errors are RMSE per Cartesian component.

- **Fit on a random split.** Fine-tuning on 51 random silica cells takes the force RMSE on the
  20 held-out cells from 50 to 15 meV/Å. Three seeds give 15.2, 15.3 and 15.3.
- **Learning curve.** Ten or twenty training cells give 26 meV/Å, forty give 17, all 51 give 15.
- **Corrupted labels.** Gaussian noise on a tenth of the training frames raises the force RMSE
  from 15.3 to 16.4 meV/Å and the energy MAE from 0.45 to 1.12 meV/atom. The bootstrap
  intervals overlap because they measure the scatter of a test set that all these runs share.
  The seed-to-seed difference, 0.1 meV/Å, is the better yardstick, and the noise costs ten
  times that.
- **Quench-rate transfer.** Holding out the fastest rate, 10¹⁴ K/s, leaves 29 meV/Å (zero-shot
  67). Holding out the slowest, 10¹¹ K/s, leaves 10.5 (zero-shot 34). Fast-quenched cells are
  harder for any model: at 10¹⁴ K/s 0.75 percent of the silicons are not four-coordinated,
  at 10¹¹ K/s none are.
- **Carbon.** Training on the 64 denser cells and testing on the 25 most porous ones cuts the
  force RMSE by about a third, from 777 to 486 meV/Å. That is still thirty times the silica
  error, so dense carbon does not teach porous carbon. The energy MAE falls from 91 to 13
  meV/atom, mostly by removing the constant offset seen in M0.
- **MD check.** The zero-shot and the fine-tuned model both hold a held-out silica cell for
  20 ps at 300 K and at 1500 K. Every silicon is still four-coordinated at the end.
- **Melt-quench at 5000 K.** The fine-tuned model melts and re-forms the network in five
  independent runs without a crash. Its glass matches AM26's own 10¹⁴ K/s cells within the
  scatter of five cells. Details below.

## The melt-quench, and what went wrong first

**The first protocol did not melt anything, and I first read it wrongly.** Four runs, NVT,
10 ps at 3000 K, then 10¹⁴ K/s down to 300 K. They started from the first four silica cells
in the file, which are 10¹¹ K/s cells, because the script ran before the rate label was
parsed. Against those true parents, 98.5 to 100 percent of the Si–O–Si bridges survive to the
end, while two unrelated AM26 cells share 1.5 to 5.5 percent by coincidence. The run with
saved frames, started from a 10¹⁴ K/s cell, keeps 93 percent. During its hold the mean squared
displacement of silicon sits on a plateau at 0.66 Å², against (V/N)^(2/3) = 6.1 Å². The atoms
rattle in their cages and the network never breaks up. Those cells are AM26 cells heated and
cooled, and their agreement with AM26 shows only that the fine-tuned model keeps a silica
network intact at 3000 K. My first check compared the final cells with the wrong parents
and reported that the network had been erased. That was wrong, and it is corrected here and
in `modules/story/rings_zipper_crosscheck.md`. Vitriflow's hold-time gate exists to catch
this failure. `melt_quench.py` now records, for every run, the mean squared displacement at
the end of the hold, the fraction of bridges kept and a `network_reset` flag.

**The 5000 K protocol did melt.** Five runs, each started from a different AM26 10¹⁴ K/s
cell: NVT at that cell's density, 20 ps at 5000 K, 10¹⁴ K/s down to 300 K, 5 ps at 300 K. The
fifth also saved a frame every 250 steps for the video. At the end of the hold the mean
squared displacement of silicon was 85 to 103 Å², against (V/N)^(2/3) = 6.1 Å², and 2.5 to
4.5 percent of the starting bridges were left, the level two unrelated cells share by
coincidence. The fine-tuned model ran all 72 000 steps of every run, at 45 to 56 ms per step,
and the shortest Si–O distance at the end is 1.53 to 1.55 Å. AM26's silica set contains no
liquid configurations, so this is the model working outside its training data.

**The glass the model made.** Against AM26's twenty 10¹⁴ K/s cells
(`outputs/melt_quench_5000K/melt_quench_vs_am26.csv`):

| | model, 5 cells | AM26, 20 cells |
|---|---|---|
| Si four-coordinated | 99.4 % | 99.25 % |
| O two-coordinated | 99.5 % | 99.7 % |
| mean ring length, atoms (matscipy) | 11.0 ± 0.4 | 11.0 ± 0.2 |
| mean ring size, Si (ZIPPER) | 6.74 ± 0.12 | 6.60 ± 0.16 |
| three-membered rings (ZIPPER) | 5.1 % | 5.2 % |
| nine- to twelve-membered rings (ZIPPER) | 23 % | 18 % |

The Si–O partial RDFs (computed with CRISP, `modules/story/fig_A_prdf.png`) peak at the same
1.625 Å. The Si–Si peak of the model's glass is one 0.05 Å bin further out, 3.075 against
3.025 Å. Of the 120 bins up to 6 Å, 18 for Si–O and 10 for Si–Si lie more than one AM26
standard deviation from the AM26 mean. The density is not a result, because the volume was
fixed at the starting cell's. Within five cells the model's glass is consistent with AM26 at
the same cooling rate, with a possible excess of large rings that five cells cannot settle.

**The liquid.** In the 5000 K liquid of the fifth run, about 42 percent of the silicons have
exactly three oxygens within 2.0 Å at any instant, and 49 percent have four
(`modules/story/coordination_along_quench.csv`). At least 99 percent are four-coordinated
again from 51 ps on, when the thermostat is at 1900 K. The three-coordinated fraction depends
on the fixed 2.0 Å cutoff, which is too tight for a liquid this hot. It also comes from a
region the model never saw in training, so it is not a statement about the chemistry of
liquid silica.

**The committee.** Three seeds of the naive fine-tune were evaluated on the saved frames of
every run, the committee check of Schran, Brezina and Marsalek (2020); Beck and co-workers
build the same idea into one foundation model with several heads. The spread is the norm of
the per-atom force standard deviation across the three models, so it is compared with the
per-atom error norm on the held-out glass, √3 × 15.3 = 26.5 meV/Å. In the 5000 K liquid the
spread averages 10 meV/Å with a 95th percentile of 21. In the final glass it is 1.3 meV/Å,
twenty times below the true error. Along the 3000 K run it was 3.4 meV/Å at 3000 K and 0.9 at
300 K. The spread rises eightfold where the model extrapolates, which is the useful part. As
an error bar it is too small everywhere: a committee of seeds from one fine-tuned foundation
model shares its training data and its base, so the seeds agree with each other more than
with DFT.

## Arms

The question is transfer, not fit. The arms differ in what is held out.

| Arm | YAML and split | Train on | Test on |
|---|---|---|---|
| A naive | `finetune_naive.yaml`, `a-SiO2` | 51 silica cells, random | 20 silica cells, random 25 percent |
| A repeat | `finetune_naive_holdmin.yaml`, `a-SiO2_holdmin` | same split as A | same 20 cells |
| B replay | `finetune_replay.yaml`, `a-SiO2` | as A, plus multihead replay on a subsample of the pretraining data | same 20 cells |
| rate14 | `finetune_naive.yaml`, `a-SiO2_rate14` | 60 cells quenched at 10¹¹, 10¹² and 10¹³ K/s (51 train, 9 valid) | the 20 cells quenched at 10¹⁴ K/s |
| rate11 | `finetune_naive.yaml`, `a-SiO2_rate11` | 60 cells at 10¹², 10¹³ and 10¹⁴ K/s (51 train, 9 valid) | the 20 cells at 10¹¹ K/s |
| D carbon | `finetune_naive_aC.yaml`, `a-C` | 64 carbon cells at 2.0 to 3.5 g/cm³ | the 25 most porous cells, 1.5 to 2.0 g/cm³ |
| E learning curve | arm A with `train_n10.xyz`, `train_n20.xyz`, `train_n40.xyz` from `data/subset_train.py` | 10, 20 or 40 of the 51 | same 20 cells |
| F noisy labels | arm A with `train_noisy_f10.xyz` from `data/inject_noise.py` | 51 cells, a random tenth with Gaussian noise of 300 meV/Å on every force component and 50 meV/atom on the energy | same 20 cells |

Arm A is also run with seeds 1 and 2, so the difference between arms can be read against the
difference between repeats. The AM26 metadata has no key named after the quench rate, so the
first split fell back to a random hold-out, and the run labelled `holdmin` reused that split.
It is a repeat of arm A, not a reverse hold-out. Once the rate was found inside the `label`
field, the two rate arms were made with `--group-key label --group-regex '10-(\d+)'`. Arm F is
the baseline that on-the-fly down-weighting of bad labels (Lam, O'Neill, Schran and Schaaf,
arXiv:2602.08849) is meant to recover. The down-weighting itself is not implemented here.

## Methods

Why MPA-0: it is MIT, so a fine-tuned checkpoint can be released, and Tompa and co-workers
found that the strength of the base model matters more than the fine-tuning method. The
recipe is theirs too: `E0s: estimated` (model-aware re-estimation of the atomic reference
energies), the Huber-type universal loss, zero weight decay, learning rate 5e-4, batch 2, 150
epochs, EMA 0.99, float64. The early-stopping patience of 40 epochs never triggered. Nothing
was tuned on a test set. The split manifests under `data/splits/` list every index and can be
verified.

`md_stability.py` runs 20 ps of Langevin NVT at 300 K and 1500 K on the first held-out cell.
`melt_quench.py` starts from an AM26 10¹⁴ K/s cell, holds it at 5000 K for 20 ps in NVT at
that cell's density, cools it at 100 K/ps (10¹⁴ K/s, the fastest rate in AM26) to 300 K, holds
5 ps, and compares the final cells with AM26's twenty 10¹⁴ K/s cells. The time step is 1 fs.
The first attempt is reproduced with `--t-melt 3000 --melt-ps 10`.

```bash
python -c "from casebook.models import ensure_file; ensure_file('https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model', 'models_cache')"
mace_run_train --config modules/M1_finetune_asio2/finetune_naive.yaml
python modules/M1_finetune_asio2/evaluate.py --test data/splits/a-SiO2/test.xyz \
    --model zero-shot=models_cache/mace-mpa-0-medium.model \
    --model naive=modules/M1_finetune_asio2/checkpoints/am26_asio2_mpa0_naive.model

# a rate arm: the same YAML with the split files overridden
S=data/splits/a-SiO2_rate14
mace_run_train --config modules/M1_finetune_asio2/finetune_naive.yaml --name am26_asio2_mpa0_naive_rate14 \
    --train_file $S/train.xyz --valid_file $S/valid.xyz --test_file $S/test.xyz

python modules/M1_finetune_asio2/md_stability.py --test data/splits/a-SiO2/test.xyz --frame 0 --temps 300 1500 --ps 20 \
    --model zero-shot=models_cache/mace-mpa-0-medium.model \
    --model naive=modules/M1_finetune_asio2/checkpoints/am26_asio2_mpa0_naive.model

# one 5000 K melt-quench run from the first 10^14 K/s cell, with the seed committee
python modules/M1_finetune_asio2/melt_quench.py --device cuda --seeds 1 --first-seed 0 \
    --out modules/M1_finetune_asio2/outputs/melt_quench_5000K \
    --model naive=modules/M1_finetune_asio2/checkpoints/am26_asio2_mpa0_naive.model \
    --model s1=modules/M1_finetune_asio2/checkpoints/am26_asio2_mpa0_naive_s1.model \
    --model s2=modules/M1_finetune_asio2/checkpoints/am26_asio2_mpa0_naive_s2.model
```

`hpc/README.md` lists the exact submissions for every arm.

## Outputs

`outputs/heldout_metrics.csv` holds arm A, and `outputs/<name>/heldout_metrics.csv` every
other run. Each has the force RMSE, MAE, median and 95th percentile, relative RMSE, per-element
force RMSE and per-atom energy errors on the held-out set, zero-shot and fine-tuned side by
side. `outputs/md_stability/md_stability_summary.csv` gives, per model and temperature, the
picoseconds completed, the potential-energy drift over the second half of the run, the
shortest interatomic distance at start and end, and the coordination fractions at start and
end. `outputs/melt_quench_5000K/` holds `melt_quench_summary.csv` (per run: finished or
crashed, steps, ms per step, the melt diagnostics, the shortest distance and the descriptors
of the final cell), `melt_quench_vs_am26.csv`, the temperature series and
`committee_along_quench_seed*.csv`. The final cells and the saved trajectory are written as
extxyz and kept out of git. `melt_quench_naive/`, `melt_quench_zeroshot/` and
`melt_quench_traj/` hold the 3000 K attempt. Checkpoints and training logs are not committed.

The acceptance line was modest: the fine-tuned force RMSE on the held-out set below the
zero-shot value of the same base model. Every run passes it. A fine-tune that fitted the
training cells and then collapsed at 1500 K would fail the MD check whatever its RMSE, which
is why the MD check is part of the module.

## Wall-clock and hardware

From the PBS job records, one GPU per job:

| Job | GPU | Wall-clock |
|---|---|---|
| naive fine-tune, 51 cells, 150 epochs, then evaluation | L40S / A100 | 50 / 25 min |
| learning-curve fine-tune, 10 to 40 cells | A100 or H100 | 14 to 17 min |
| carbon fine-tune, 64 cells | A100 | 42 min |
| full replay fine-tune | not recorded | 12 h walltime, stopped at epoch 32 |
| MD check, two models, two temperatures, float64 | L40S | 3 h 52 min |
| MD check, fine-tuned only, float32 | H100 | 22 min |
| 3000 K melt-quench, 4 runs × 2 models | L40S | 4 h 23 min |
| 5000 K melt-quench, one run with committee | L40S, L40 or A100 | 56 to 71 min |

## What this does not show

Transfer to other chemistries or to experiment. There are two materials, three silica
hold-outs (random, fastest rate, slowest rate) and one carbon hold-out, all against PBE labels.

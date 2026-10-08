# HPC templates

PBS templates with `<<PLACEHOLDER>>` fields. I fill them into copies named `*.local.pbs`,
which git ignores, so no queue name, account or personal path is committed. Every job name
begins with `am26_`.

| Template | Runs | Fill in |
|---|---|---|
| `job_triage_cpu.pbs` | M0 triage on a-SiO2 and a-C, then M2 on a-SiO2 per quench rate | CPU queue, cores, memory, walltime |
| `job_extra_cpu.pbs` | M4 net-force audit, M2 on a-C per density, M0 on the remaining systems with the MIT models | CPU queue, cores, memory, walltime |
| `job_finetune_gpu.pbs` | one M1 fine-tune on the random split, `ARM=naive` or `replay`, then held-out evaluation | GPU queue, one GPU, memory, walltime |
| `job_finetune_general.pbs` | any M1 YAML and split, with optional `SEED`, `TRAIN` and `OPTS` overrides; evaluation into `outputs/<NAME>/` | GPU queue, one GPU, memory, walltime |
| `job_md_stability.pbs` | 20 ps NVT at 300 and 1500 K, zero-shot against arm A, float64 | GPU queue, one GPU, memory, walltime |
| `job_md_one.pbs` | the same MD check for one model, float32 | GPU queue, one GPU, memory, walltime |
| `job_mq_melt.pbs` | one melt-quench run: 5000 K for 20 ps, 10¹⁴ K/s to 300 K, 5 ps hold; `SEED` picks the starting AM26 cell, `TRAJ=1` saves frames for the video | GPU queue, one GPU, memory, walltime |
| `job_melt_quench.pbs` | the first melt-quench attempt, 3000 K for 10 ps, kept so it can be reproduced; it did not melt the network | GPU queue, one GPU, memory, walltime |

How the M1 runs in the results table were submitted with `job_finetune_general.pbs`:

```bash
J=hpc/job_finetune_general.local.pbs
Y=modules/M1_finetune_asio2/finetune_naive.yaml
qsub -N am26_ft_s1 -v CONFIG=$Y,SPLIT=data/splits/a-SiO2,NAME=am26_asio2_mpa0_naive_s1,SEED=1 $J
qsub -N am26_ft_n10 -v CONFIG=$Y,SPLIT=data/splits/a-SiO2,NAME=am26_asio2_mpa0_naive_n10,TRAIN=data/splits/a-SiO2/train_n10.xyz $J
qsub -N am26_ft_noisy -v CONFIG=$Y,SPLIT=data/splits/a-SiO2,NAME=am26_asio2_mpa0_naive_noisy10,TRAIN=data/splits/a-SiO2/train_noisy_f10.xyz $J
qsub -N am26_ft_rate14 -v CONFIG=$Y,SPLIT=data/splits/a-SiO2_rate14,NAME=am26_asio2_mpa0_naive_rate14,OPTS=train_file=data/splits/a-SiO2_rate14/train.xyz:valid_file=data/splits/a-SiO2_rate14/valid.xyz:test_file=data/splits/a-SiO2_rate14/test.xyz $J
qsub -N am26_ft_replay_short -v CONFIG=modules/M1_finetune_asio2/finetune_replay.yaml,SPLIT=data/splits/a-SiO2,NAME=am26_asio2_mpa0_replay_short,OPTS=max_num_epochs=30:num_samples_pt=1000 $J
qsub -N am26_ft_aC -v CONFIG=modules/M1_finetune_asio2/finetune_naive_aC.yaml,SPLIT=data/splits/a-C,NAME=am26_aC_mpa0_naive $J
```

Seed 2, the 20- and 40-cell subsets and the rate11 split follow the same pattern. The
melt-quench runs were one job per starting cell:

```bash
for s in 0 1 2 3; do qsub -N am26_melt_s$s -v SEED=$s hpc/job_mq_melt.local.pbs; done
qsub -N am26_melt_traj -v SEED=4,TRAJ=1 hpc/job_mq_melt.local.pbs
```

All templates expect the conda environment from `environment.yml` and the repository at
`<<REPO>>`. The GPU templates install nothing, so the CUDA build of torch has to be in the
environment already. Jobs run in place on the shared filesystem because the IO is small.
They request no scratch and delete nothing.

The job output files, kept locally in `job_outputs/`, are the record of wall-clock time and
hardware quoted in the module READMEs. `tools/track_jobs.sh` prints the queue state and what
each job has written so far.

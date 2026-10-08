# Ring statistics and network memory, checked with ZIPPER

The ring PMF in M2 and in the melt-quench comparison comes from matscipy, which counts
shortest-path rings on the full Si–O graph with lengths in atoms, capped at 16 atoms. As a
cross-check I ran the same cells through the periodic shortest-path ring enumerator of my
zeolite package ZIPPER (`zeozipper.topology.TNet`, `ring_counts`). It contracts the oxygens
away, walks the quotient graph with explicit lattice shifts and counts rings in silicons up to
12. Its topology is graded against IZA's published coordination sequences (265 of 265
framework types) and long vertex symbols (1145 T-sites, two of which differ in a multiplicity
only). Each n-ring passes through n silicons, so per-site counts are divided by n. The Si–O
cutoff is 2.0 Å.

ZIPPER is not public yet, so the scripts and their raw output sit in `zipper_crosscheck/`,
which is kept out of git until it is: `rings.py` with `rings_output.txt`,
`network_memory.py` with `network_memory_output.txt`, `melt_check.py` with
`melt_check_output.txt`, and `melt5000_check.py` with `melt5000_check_output.txt`. All were
run on 8 October 2026.

## AM26 by quench rate

| | cells | mean ring size, Si | Si–Si degree | 3-rings | 9- to 12-rings |
|---|---|---|---|---|---|
| AM26, 10¹⁴ K/s | 20 | 6.60 ± 0.16 | 4.013 | 5.2 % | 18.4 % |
| AM26, 10¹¹ K/s | 20 | 6.49 ± 0.10 | 4.000 | 3.4 % | 12.4 % |

Cooling a thousand times more slowly narrows the ring distribution towards five-, six- and
seven-membered rings. It removes about a third of the strained three-rings and a third of the
large rings, and it removes the coordination defects. At 10¹⁴ K/s, 0.5 percent of the
silicons are five-coordinated and 0.25 percent three-coordinated, with 0.2 percent
three-coordinated oxygen, which together lift the mean Si–Si degree to 4.013. At 10¹¹ K/s
there are none. This is consistent with the cooling-rate dependence of ring statistics that
Vollmayr, Kob and Binder described for simulated silica in 1996, here on AM26's DFT-labelled
cells.

## The 3000 K run did not melt the network

The four runs of the first melt-quench were started from the first four silica cells in the
file, which are 10¹¹ K/s cells. Bridge graph of each final cell against its true parent, by
atom index (indices are conserved in MD):

| run | parent | bridges kept |
|---|---|---|
| 0 | silica-mq_10-11_1 | 1.000 |
| 1 | silica-mq_10-11_10 | 1.000 |
| 2 | silica-mq_10-11_11 | 0.985 |
| 3 | silica-mq_10-11_12 | 1.000 |

Two unrelated AM26 cells share 1.5 to 5.5 percent of their bridges by coincidence. The ring
statistics agree: the four parents give a mean ring size of 6.43 ± 0.09 and the four cells
after the run 6.43 ± 0.08, with the same distribution size by size. In the run with saved
frames, started from a 10¹⁴ K/s cell, 93 percent of bridges survive the 10 ps hold, and the
mean squared displacement of silicon sits on a plateau at 0.66 Å² (oxygen 1.08 Å²) against
(V/N)^(2/3) = 6.1 Å². That is caged vibration with no diffusion. At 3000 K, on a 10 ps
timescale, this model's silica is a hot glass, not a liquid.

An earlier version of this note compared the final cells with the wrong starting cells and
concluded the opposite. That was wrong. The consequence for the other results: the
"generated" cells of that run are AM26 parents heated and cooled, so their agreement with
AM26 shows that the fine-tuned model keeps a silica network intact through 3000 K and back,
not that it makes a glass. A hotter, longer hold was then run with `hpc/job_mq_melt.pbs`,
and `melt_quench.py` now reports the mean squared displacement and the fraction of bridges
kept for every run.

## The 5000 K run, and the model's own glass

At 5000 K for 20 ps the network did reset. Only 2.5 to 4.5 percent of the starting bridges
were left at the end of the hold, against 1.5 to 5.5 percent by coincidence, and the mean
squared displacement of silicon was 85 to 103 Å² (per-run summaries in
`modules/M1_finetune_asio2/outputs/melt_quench_5000K/`). The five final cells, against AM26 at
the same cooling rate:

| | cells | mean ring size, Si | 3-rings | 9- to 12-rings | Si–Si degree |
|---|---|---|---|---|---|
| AM26, 10¹⁴ K/s | 20 | 6.60 ± 0.16 | 5.2 % | 18.4 % | 4.013 |
| fine-tuned model, 5000 K then 10¹⁴ K/s | 5 | 6.74 ± 0.12 | 5.1 % | 23.1 % | 4.016 |

Small rings and coordination agree. The model's glass may carry more nine- to twelve-membered
rings, which five cells cannot settle.

The two ring codes count differently (atoms against silicons, different caps), so their
absolute means are not comparable. Both rank the populations the same way.

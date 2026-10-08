# M2. How many amorphous cells are enough?

Status: done, 7 October 2026, for a-SiO2 per quench rate and a-C per density.

## What came out

Twenty cells per group are far from enough for any distribution-valued descriptor. At every
quench rate the Si–Si pair-distance CDF sits at R = 0.8 to 1.0 against the 0.2 target, and
the 1/√n estimate puts the number of cells needed at roughly 450 to 650. That is the same
order as the 420 cells Vitriflow needed for the Si–Si CDF of its own SHIK silica. The density
of the silica cells needs 170 to 980 cells by the same estimate. Coordination is the
exception. At 10¹¹ and 10¹² K/s every silicon in every cell is four-coordinated, so the width
across cells is zero and R = 0 from two cells on. At 10¹³ and 10¹⁴ K/s, where a few
defects appear, the Si coordination fraction needs 16 cells. For carbon the density is the
grouping variable itself, so its R = 0 says nothing. The sp² and sp³ fractions and the C–C
CDF stay at R = 0.5 to 1.3 in every density group.

In an amorphous host there is no unit cell to average over. Pore size, coordination and ring
statistics are distributions over an ensemble of cells, and the width of that ensemble is
part of the physics. The practical question is how many independent cells it takes before
each observable is pinned down to a tolerance I am willing to declare, and how far the AM26
silica set, with its handful of cells per quench rate, is from that point.

The criterion is Vitriflow's (Cottom, Delhomme and Olsson, Comput. Mater. Sci. 275, 115098,
2026). For a descriptor y measured on n cells, R_y(n) = h_n(y) / τ_y, with h_n the 95 percent
confidence half-width across cells and τ_y a declared resolution scale; the population is
supported when Q(n) = max_y R_y(n) ≤ 0.2. The tolerances follow their silica settings:
density to 1 percent with a floor of 0.001 g/cm³, pair-distance and coordination CDFs to
2 percentage points, ring-size PMF to 5 points. For their own SHIK silica population, density
needed 560 cells and the Si–Si pair-distance CDF 420 cells to reach Q ≤ 0.2. That is the
scale to keep in mind when reading what 20 cells per rate can and cannot resolve.

Descriptors computed per cell: density; fraction of four-coordinated Si and two-coordinated
O (Si–O cutoff 2.0 Å); Si–O, Si–Si and O–O first-shell pair-distance CDFs on fixed grids,
normalised to the pairs inside the window so that the two-point tolerance applies to a
probability; ring-size PMF from matscipy on the Si–O network, with ring length counted in
atoms, so a six-membered Si ring has length 12. These are ASE-based so that the module runs
with the core dependencies only; the story figures use CRISP for the same quantities.

AM26 stores the quench rate inside the `label` field, as `silica-mq_10-13_7` for 10¹³ K/s,
and the carbon density as `carbon-mq_2.5_3`, so the groups are read with a regex:

```bash
python modules/M2_ensemble_support/run.py --system a-SiO2 --group-key label --group-regex '10-(\d+)'
python modules/M2_ensemble_support/run.py --system a-C --group-key label --group-regex 'mq_([0-9.]+)'
```

The first pass, before the label format was known, pooled all cells of a system into one
group; its `*_all.*` files are kept for the record and are superseded by the per-group runs.

Outputs: `outputs/support_summary.csv` with, per group and descriptor, the cells available,
R at that n, whether q_conv was reached, the n from which it stays reached, and an estimate of
the n needed under a 1/√n assumption; `outputs/support_curve_<system>_<group>.csv` and
`.png` with R_y(n) for every descriptor, Q(n) in black and the 0.2 line.

Reading the plot: a curve still above the dashed line at the largest n is not resolved with
the cells available. The extrapolated n assumes R falls as 1/√n and is labelled as an
estimate. `n_required` is the smallest n from which the median R stays at or below 0.2 for
every larger n, so a curve that dips below the line once and comes back up does not count.
Subsets are drawn without replacement from the same cells, so the curves are a resampling
estimate, not independent populations.

What this does not show: it is not a Vitriflow run (no calibration, no screening, no
melt-quench), and it says nothing about which quench rate is right, only how many cells each
rate's distribution needs.

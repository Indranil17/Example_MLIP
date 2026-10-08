# M4. How clean are the reference forces?

Status: done, 7 October 2026. Result: the net force per atom, |ΣF|/N, has a median of about
3 × 10⁻⁷ meV/Å in every AM26 system. AM26 stores forces with eight decimals in eV/Å, and
rounding the forces of a 300-atom cell to that precision alone gives a net force of that size. So the
net force is at the rounding level of the stored forces, which is consistent with forces that
were drift-corrected before they were written. This test therefore cannot see the SCF noise
in AM26. The module stays because the check costs seconds and is the first thing to run on
any new reference set. On raw DFT output it measures the noise; here it shows only that the
net force was removed.

For a periodic DFT cell the forces must sum to zero. Whatever remains is numerical noise
in the labels, from the SCF threshold, the grid and the Pulay terms, and no model trained
on those labels can be checked below that level. Kuryla, Berger, Csányi and Michaelides
used this net-force check to show that several widely used molecular datasets carry much
larger force uncertainties than assumed (J. Chem. Phys. 163, 224313, 2025). Here I run the
same check on the AM26 periodic cells, system by system.

```bash
python modules/M4_label_quality/run.py            # all five systems, seconds on one core
```

Outputs in `outputs/`: `net_force_per_frame.csv` with |ΣF| and |ΣF|/N for every cell,
`net_force_summary.csv` with the median, 95th percentile and maximum of |ΣF|/N per system
and the fraction of cells above 1 meV/Å, and a histogram per system.

How to read it: on raw DFT forces, compare the per-system median of |ΣF|/N with the force
RMSE of the models in M0. If the two are within a factor of a few, the label noise sets the
floor for the comparison. If the net force was removed before the forces were written, as
the AM26 numbers suggest, |ΣF|/N is close to zero by construction and says nothing about the
SCF noise. AM26 does not document which was done.

What this does not show: the accuracy of the labels against a tighter reference, only
their internal consistency.

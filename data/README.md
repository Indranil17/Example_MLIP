# Data

Only scripts, `CHECKSUMS.json` and the split manifests are committed here. Structures are
downloaded or written when the scripts run.

## AM26

Fragapane and Deringer, arXiv:2607.11384 (2026), https://github.com/vldgroup/AM26. One
extxyz file, `data/am26.extxyz`, with 934 melt-quenched cells of a-C (densities 1.5 to
3.5 g/cm³), a-SiO2 (quench rates 10¹¹ to 10¹⁴ K/s), a-Si, a-LiPS and a-GST, labelled with
PBE energies and forces. The repository README does not document the metadata keys or the
counts per system, so both are read off the file before anything else:

```bash
python data/get_am26.py          # download and check the SHA-256 against CHECKSUMS.json
python data/inspect_am26.py      # systems by composition, info keys, candidate group keys
```

`CHECKSUMS.json` holds the file used here: its SHA-256, size, and the AM26 commit on `main`
at download time, `e88472a`. If upstream changes the file, the check fails. The exact file can
then still be fetched from that commit with
`--url https://raw.githubusercontent.com/vldgroup/AM26/e88472a56614eaa24d9b4b839f2543249baf6674/data/am26.extxyz`.

Systems are labelled from composition (Si and O only is a-SiO2, and so on), which needs no
metadata. There is no key named after the quench rate. The rate sits inside the `label`
field, as `silica-mq_10-13_7` for a 10¹³ K/s cell, and the carbon density as
`carbon-mq_2.5_3`. The energies and forces are stored on the ASE calculator, and every
written frame carries them as `REF_energy` and `REF_forces`.

## Splits

```bash
# random 25 percent hold-out (20 of the 80 silica cells): arm A, its seeds, the learning curve, the noisy run
python data/make_splits.py --system a-SiO2
# a whole quench rate held out: the fastest (10^14 K/s) or the slowest (10^11 K/s)
python data/make_splits.py --system a-SiO2 --group-key label --group-regex '10-(\d+)' --holdout max --out data/splits/a-SiO2_rate14
python data/make_splits.py --system a-SiO2 --group-key label --group-regex '10-(\d+)' --holdout min --out data/splits/a-SiO2_rate11
# carbon: the lowest-density quarter held out (density computed from each cell, 1.5 to 2.0 g/cm^3)
python data/make_splits.py --system a-C --group-key density --holdout min --holdout-fraction 0.25
python data/make_splits.py --verify data/splits/a-SiO2

# training-set variants on the random split
python data/subset_train.py --split data/splits/a-SiO2 --n 10      # also --n 20 and --n 40
python data/inject_noise.py --split data/splits/a-SiO2 --fraction 0.10 --sigma-force-meV-A 300 --sigma-energy-meV-atom 50
```

`data/splits/a-SiO2_holdmin/` was written by `--holdout min` before the rate key was known.
With no group key the command fell back to the random split, so this folder holds the same
split as `a-SiO2/` and the run trained on it is a repeat of arm A.

Every written frame carries `am26_index`, its position in the source file. The manifest lists
every index per split, and `--verify` re-reads the three files and fails if an index appears
twice.

## Licence

The AM26 repository had no licence file on 7 October 2026. The data are cited and downloaded
by whoever runs the scripts, and the labels themselves are not redistributed here. The
per-frame CSVs in the module outputs store prediction errors, not the reference energies.

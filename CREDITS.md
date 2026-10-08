# Credits

Every dataset, model and method used here, with source and licence. The code in this
repository is MIT. Nothing listed below is redistributed unless its licence allows it and
the entry says so.

## Data

- **AM26.** N. L. Fragapane and V. L. Deringer, *Amorphous materials as a frontier challenge
  for universal interatomic potentials*, arXiv:2607.11384 (2026). Data:
  https://github.com/vldgroup/AM26, file `data/am26.extxyz`. The repository carried no
  licence file on 7 October 2026; the data are downloaded at run time and not committed.
- **Vitriflow silica population (M3, when run).** Generated locally with the Vitriflow package
  and its shipped BKS silica example. No DFT labels.

## Models

- **MACE-MP-0b3** and **MACE-MPA-0.** I. Batatia et al., *A foundation model for atomistic
  materials chemistry*, J. Chem. Phys. 163, 184110 (2025); arXiv:2401.00096. MIT.
  https://github.com/ACEsuit/mace-foundations
- **MACE-MH-1.** Cross-domain multihead model, Academic Software License. Evaluated in M0
  only. No model derived from it is trained or redistributed; its predictions appear here only
  as evaluation results. https://github.com/ACEsuit/mace-foundations
- **MACE.** I. Batatia, D. P. Kovács, G. N. C. Simm, C. Ortner, G. Csányi, *MACE: Higher order
  equivariant message passing neural networks for fast and accurate force fields*, NeurIPS
  2022. mace-torch 0.3.16. https://github.com/ACEsuit/mace

## Methods followed

- **Fine-tuning recipe and E0 re-estimation (M1).** T. L. Tompa, E. Varga-Umbrich,
  I. Batatia, A. M. Elena, N. Bernstein, G. Csányi, *Fine-tuning MLIP foundation models:
  strategies for accuracy and transferability*, arXiv:2606.12704 (2026). `E0s: estimated` in
  mace-torch 0.3.16 is the model-aware re-estimation they describe.
- **Support criterion (M2) and calibrated melt-quench (M3).** J. Cottom, R. Delhomme,
  E. Olsson, *Vitriflow: calibrated amorphous structure ensembles from melt-quench
  simulation*, Comput. Mater. Sci. 275, 115098 (2026); arXiv:2607.01407. Code:
  https://github.com/Olsson-Materials-Modelling/Vitriflow (MIT). Data and archived version:
  https://doi.org/10.5281/zenodo.21111823. M2 re-implements their support condition
  Q(n) = max_y h_n(y)/τ_y ≤ 0.2 on AM26 cells and does not use or modify their code.
- **Spread between models against error (M0) and the seed committee (M1).** C. Schran, K. Brezina, O. Marsalek,
  *Committee neural network potentials control generalization errors and enable active
  learning*, J. Chem. Phys. 153, 104105 (2020); H. Beck, P. Simko, L. L. Schaaf, O. Marsalek,
  C. Schran, *Multi-head committees enable direct uncertainty prediction for atomistic
  foundation models*, J. Chem. Phys. 163, 234103 (2025).
- **Net-force audit (M4).** D. Kuryla, F. Berger, G. Csányi, A. Michaelides, *How accurate are
  DFT forces? Unexpectedly large uncertainties in molecular datasets*, J. Chem. Phys. 163,
  224313 (2025).
- **Noisy-label baseline (M1, arm F).** Lam, O'Neill, Schran and Schaaf, arXiv:2602.08849
  (2026). Arm F trains on corrupted labels without their down-weighting, as the baseline that
  method is meant to improve on.
- **Cooling-rate dependence of silica rings.** K. Vollmayr, W. Kob, K. Binder, *Cooling-rate
  effects in amorphous silica: A computer-simulation study*, Phys. Rev. B 54, 15808 (1996).
- **Partial radial distribution functions in the story figure.** CRISP, I. Saha, D. Willimetz,
  L. Grajciar, *CRISP: Enhancing ASE Workflows with Advanced Molecular Simulation
  Post-Processing*, J. Comput. Chem. 47, e70384 (2026); `pip install crisp-ase`.
  https://github.com/Indranil17/CRISP. Licence: CC BY 4.0 in the repository's LICENSE file;
  the PyPI metadata of crisp-ase 1.1.4 states CC BY-NC-SA 4.0. CRISP is not redistributed
  here; the scripts load it from a separate checkout or installation. `tools/story_plots.py`,
  `tools/story_dynamic.py` and `tools/story_video.py` call its `compute_pairwise_rdf` on each
  cell or frame, and its `calculate_coordination` and `calculate_avg_percentages` for the
  coordination fractions. The coordination module comes from the development version on
  GitHub and is not in crisp-ase 1.1.4 on PyPI.
- **Ring statistics.** matscipy, `matscipy.rings.ring_statistics`, LGPL-2.1.
  https://github.com/libAtoms/matscipy
- **Ring cross-check.** ZIPPER (`zeozipper`), my zeolite analysis package, not public yet.
  Used only for the cross-check in `modules/story/rings_zipper_crosscheck.md`; its scripts and
  output stay out of this repository until it is released. Its topology is graded against the
  coordination sequences and vertex symbols of the IZA Database of Zeolite Structures
  (Ch. Baerlocher and L. B. McCusker, https://www.iza-structure.org/databases/).

## Software

- **ASE.** A. H. Larsen et al., *The atomic simulation environment: a Python library for
  working with atoms*, J. Phys.: Condens. Matter 29, 273002 (2017). LGPL-2.1. Structures,
  neighbour lists and MD.
- **PyTorch** (BSD-3-Clause), **NumPy**, **SciPy**, **pandas** (BSD-3-Clause), **matplotlib**
  (matplotlib licence, BSD-style), **PyYAML** (MIT).
- **imageio-ffmpeg** (BSD-2-Clause) with its bundled **FFmpeg** binary (LGPL/GPL) for
  `modules/story/quench_video.mp4`, and **Pillow** (HPND licence) for the GIF.

## Learning

- ML4CHEM 2026 Summer School, Leipzig, 3 to 5 August 2026, co-organised and funded by COST
  Action DAEMON (CA22154). MACE notebooks by Ilyes Batatia (MIT,
  https://github.com/ilyes319/mace-tutorials): practice I and II, theory, fine-tuning. The
  lecture material is the lecturers' and is not redistributed.

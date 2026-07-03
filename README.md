# EEG Microstate Aggregate Conditional Entropy — an AD-Specific Biomarker

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20756169-blue)](https://doi.org/10.5281/zenodo.20756169)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Platform: Google Colab](https://img.shields.io/badge/Platform-Google%20Colab-orange)](https://colab.research.google.com/)

**Author:** Hassan Farooq — Sargodha Medical College, Sargodha, Pakistan
**Contact:** hasanfarooq.edu@gmail.com

A fully reproducible Python pipeline that computes the **Aggregate
Conditional Entropy, ACE(T)**, of EEG microstate transition sequences —
an interpretable, MRI-free, deep-learning-free biomarker that
differentiates Alzheimer's disease (AD) from healthy controls (HC) in
resting-state EEG.

---

## Overview

The resting brain sustains spontaneous spatiotemporal fluctuations whose
transition structure can be summarized by **EEG microstates** — quasi-stable
scalp topographies that persist for 60–120 ms before switching to a new
configuration. This pipeline models the sequence of microstates as a
first-order Markov chain and computes the **row-averaged conditional
entropy of the empirical transition matrix**, ACE(T), as a scalar index of
transition diversity:

```
ACE(T) = (1/K) * Σᵢ [ -Σⱼ Tᵢⱼ log₂(Tᵢⱼ) ]
```

where `T` is the `K × K` empirical transition matrix (`K = 4` canonical
microstate classes) and the theoretical maximum is `log₂(K) = 2.0 bits`.
Lower ACE(T) indicates a more constrained, stereotyped transition
repertoire; higher ACE(T) indicates a more flexible one.

Using resting-state EEG from the OpenNeuro **ds004504** dataset
(Alzheimer's disease, frontotemporal dementia (FTD), and healthy controls),
this pipeline shows that ACE(T) is significantly reduced in AD relative to
HC (**AUC = 0.691, 95% CI 0.555–0.819; Holm-corrected permutation
p = 0.021**), while FTD does not differ significantly from HC in this
underpowered subsample. Full statistical and methodological detail is in
the accompanying manuscript (in preparation).

**This is a research pipeline for exploratory biomarker development, not a
validated diagnostic tool.** See [Limitations](#limitations).

---

## What the pipeline does

`reproduce.py` is a single, self-contained script that performs the entire
analysis end to end:

1. **Downloads** resting-state EEG from OpenNeuro `ds004504` via the public,
   no-login AWS S3 mirror.
2. **Preprocesses** each recording: 1–40 Hz zero-phase FIR bandpass filter,
   common-average reference, FastICA (95% explained variance) artefact
   rejection with automated EOG-component detection.
3. **Extracts microstates**: modified K-means clustering (`K = 4` by
   default) on GFP-peak topographies, using a polarity-invariant spatial
   correlation distance.
4. **Computes ACE(T)** per participant from the empirical first-order
   Markov transition matrix, along with two control metrics used in the
   manuscript to rule out confounds:
   - the stationary-distribution-weighted entropy rate, `H_rate(T)`
     (Cover & Thomas), and
   - the microstate occupancy entropy, `H(π)`.
5. **Runs the statistical battery**: Kruskal-Wallis with η², pairwise
   permutation tests (10,000 permutations) with Holm-Bonferroni correction,
   bootstrap ROC/AUC with 95% confidence intervals, and a sex-stratified
   sensitivity check.
6. **Generates figures** — group ACE(T) distributions and ROC curves —
   built directly from the statistics actually computed in that run (not
   hard-coded illustrative values).

## Repository contents

```
.
├── src/                        Modular pipeline functions
│   ├── preprocessing.py        Filtering, referencing, ICA
│   ├── microstate_extraction.py   K-means microstate extraction
│   └── entropy_rate.py         Transition matrix and entropy metrics
├── scripts/
│   ├── run_full_pipeline.py    Runs src/ modules over all subjects
│   └── generate_figures.py     Regenerates figures from results/
├── notebooks/
│   └── exploratory_analysis.ipynb   Step-by-step interactive walkthrough
├── reproduce.py                 ★ Single-file, fully self-contained
│                                  reproduction of the whole pipeline —
│                                  the recommended entry point
├── requirements.txt
├── LICENSE
└── README.md
```

`reproduce.py` does not import from `src/`; it inlines everything needed so
that it can be copied and run on its own. It also implements the full
statistics and figure-generation steps that `scripts/` and `notebooks/`
only partially cover, and it computes ACE(T) exactly as defined in the
manuscript's Methods section. If you only run one thing in this repo, run
`reproduce.py`.

## Getting started

### Prerequisites

- Python ≥ 3.9 (developed on 3.10)
- ~16 GB RAM recommended (Google Colab or a local machine)
- ~5 GB free disk for the dataset
- AWS CLI (`pip install awscli`) *or* `openneuro-py`, for the initial
  dataset download

### Installation

```bash
git clone https://github.com/hadbierox196/EEG-Microstate-Entropy-Rate.git
cd EEG-Microstate-Entropy-Rate
pip install -r requirements.txt
```

### Run the full reproduction (recommended)

```bash
python reproduce.py
```

This downloads `ds004504` into `./data_mci_ad` (skipped automatically if
already present), runs the full pipeline, prints a statistical summary,
and writes:

```
results/
├── ace_results.csv          Per-subject ACE(T), H_rate, H(π), metadata
├── statistics_summary.json  All test statistics, effect sizes, CIs
└── figures/
    ├── figure_group_ACE.png / .pdf
    └── figure_roc.png / .pdf
```

### Useful flags

```bash
# Use a dataset copy you already downloaded elsewhere
python reproduce.py --data-root /path/to/ds004504 --skip-download

# Re-run only the statistics/figures from an existing results/ace_results.csv
python reproduce.py --stats-only

# Smoke-test the statistics and figure code on synthetic data —
# no EEG data, mne, or pycrostates required
python reproduce.py --demo

# Match a different number of microstate classes or permutations
python reproduce.py --k 5 --n-perm 5000
```

Run `python reproduce.py --help` for the full option list.

### Alternative entry points

- `scripts/run_full_pipeline.py` + `scripts/generate_figures.py` mirror the
  original modular pipeline built on `src/`.
- `notebooks/exploratory_analysis.ipynb` walks through the same steps
  interactively, useful for inspecting intermediate outputs (a single
  subject's transition matrix, GFP peaks, etc.) on Google Colab.

## Data

The dataset is [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504)
(AHEPA Hospital EEG dataset; Miltiadous et al., 2023), comprising
resting-state, eyes-closed, 19-channel (10–20 system) EEG from 88
participants (HC n = 27, AD n = 35, FTD n = 23 after quality control). It
is publicly available, fully de-identified, and downloads automatically via
the no-login AWS S3 mirror — no data use agreement or account is required.

## Key findings

| Group | N  | Mean ACE(T) [bits] | SD     |
|-------|----|--------------------|--------|
| HC    | 27 | 0.521              | 0.091  |
| FTD   | 23 | 0.520              | 0.110  |
| AD    | 35 | 0.454              | 0.098  |

- **Kruskal-Wallis (HC/AD/FTD):** H(2) = 7.464, p = 0.024, η² = 0.067
- **AD vs. HC:** ΔACE = 0.067 bits, Cohen's d = 0.705, Holm-corrected
  permutation p = 0.021, **AUC = 0.691** (95% CI 0.555–0.819)
- **AD vs. FTD:** ΔACE = −0.066 bits, Holm-corrected p = 0.042
- **FTD vs. HC:** not significant (p = 0.963, AUC = 0.517) — underpowered
  at n = 23, not evidence of disease non-specificity
- The theta/alpha spectral power ratio outperforms ACE(T) as a univariate
  classifier (AUC = 0.765 vs. 0.691); ACE(T)'s contribution is
  interpretability and theoretical grounding rather than peak accuracy.

See the manuscript for the full results, including per-row entropy
decomposition, sex-stratified analysis, K-sensitivity sweep, and control
analyses ruling out an occupancy-imbalance confound.

## Limitations

- Single-site, cross-sectional, in-sample discovery cohort (N = 85); no
  external validation cohort has been evaluated yet.
- The FTD group (n = 23) is underpowered (minimum detectable d ≈ 0.811);
  the null FTD-vs-HC result should not be read as confirmed AD
  specificity.
- No medication data are available for this cohort; cholinesterase
  inhibitors and memantine can independently affect EEG dynamics.
- ACE(T) is not the best-performing univariate metric on this cohort (the
  theta/alpha ratio is); its value is interpretability and a transparent
  information-theoretic derivation, not maximal discriminative power.

This is preliminary, exploratory research and is **not validated for
clinical or diagnostic use**.

## Dependencies

See `requirements.txt`. Key packages: `mne`, `pycrostates`,
`scikit-learn`, `scipy`, `numpy`, `pandas`, `matplotlib`, `seaborn`,
`tqdm`. Downloading the dataset additionally needs the AWS CLI or
`openneuro-py`.

## Citation

If you use this code or these results, please cite:

```bibtex
@software{farooq2025ace,
  author  = {Farooq, Hassan},
  title   = {EEG Microstate Aggregate Conditional Entropy (ACE) Pipeline},
  year    = {2025},
  url     = {https://doi.org/10.5281/zenodo.20756169},
  doi     = {10.5281/zenodo.20756169}
}
```

Please also cite the source dataset:

```bibtex
@article{miltiadous2023dataset,
  author  = {Miltiadous, A. and Zyberaj, M. and Tzimourta, K. D. and
             Terzidou, E. and Giannakeas, N. and Tzallas, A. T.},
  title   = {A Dataset of Scalp EEG Recordings of Alzheimer's Disease,
             Frontotemporal Dementia and Healthy Subjects},
  journal = {Data},
  volume  = {8},
  number  = {6},
  pages   = {95},
  year    = {2023},
  doi     = {10.3390/data8060095}
}
```

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

We thank the OpenNeuro platform and the AHEPA Hospital dataset
contributors (Miltiadous et al., 2023) for making this data publicly
available. All development and analysis were performed on Google Colab
infrastructure.

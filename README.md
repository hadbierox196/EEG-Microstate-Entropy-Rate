# EEG Microstate Entropy Rate as a Transdiagnostic Biomarker

[![DOI](https://zenodo.org/badge/1273985303.svg)](https://doi.org/10.5281/zenodo.20756169)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Platform: Google Colab](https://img.shields.io/badge/Platform-Google%20Colab-orange)](https://colab.research.google.com/)

**Authors:** [Your Name], [Co-author]  
**Affiliation:** [Your Institution]  
**Contact:** [Your Email]

## Overview
This repository provides a fully reproducible pipeline to compute **entropy rate** of EEG microstate sequences as a criticality‑derived biomarker for neuropsychiatric disorders. Using resting‑state EEG from the OpenNeuro **ds004504** dataset (Alzheimer’s disease, Frontotemporal dementia, and healthy controls), we show that entropy rate distinguishes Alzheimer’s disease from controls (AUC = 0.691, p = 0.007) with medium effect size (η² = 0.078).

The pipeline:
- Downloads public EEG data (AWS S3)
- Preprocesses (filter, re-reference, ICA artifact removal)
- Extracts microstates via modified K‑means on GFP peaks
- Computes maximum‑entropy Markov transition matrices
- Calculates entropy rate per subject
- Performs group statistics and ROC analysis
- Generates publication‑ready figures

## Key Findings
| Group | N | Mean H (nats) | SD |
|-------|---|--------------|-----|
| HC    | 27| 0.5210       | 0.0909 |
| FTD   | 23| 0.5198       | 0.1098 |
| AD    | 35| 0.4538       | 0.0984 |

- **AD vs HC**: ΔH = 0.067 ± 0.095, permutation p = 0.007, AUC = 0.691 (95% CI: 0.549–0.819)
- **FTD vs HC**: no significant difference (AUC = 0.517)

## Repository Contents
- `src/` – modular Python functions for each processing step
- `scripts/` – executable scripts to run the full pipeline
- `notebooks/` – Jupyter notebook for interactive exploration
- `results/` – output CSV, figures, and tables
- `paper/` – abstract PDF

## Getting Started

### Prerequisites
- Python ≥ 3.9
- Google Colab or local machine with 16GB RAM (recommended)

### Installation (local)
```bash
git clone https://github.com/YOUR_USERNAME/EEG-Microstate-Entropy-Rate.git
cd EEG-Microstate-Entropy-Rate
pip install -r requirements.txt
```

Run the full pipeline

```bash
python scripts/run_full_pipeline.py
```

This will download the data, process all subjects, and save results to results/.

Reproduce figures

```bash
python scripts/generate_figures.py
```

## Dependencies

See requirements.txt. Key packages:

· mne
· pycrostates
· scikit-learn
· scipy
· numpy
· pandas
· matplotlib
· seaborn
· tqdm

## Data

The dataset is OpenNeuro ds004504 (AHEPA Hospital EEG). It is automatically downloaded via AWS S3 (no login required). For details, see the dataset paper.

## Citation

If you use this code or results, please cite:

```
[Your Paper] (in preparation)
```

Also cite the dataset:

```
[Data descriptor citation]
```

## License

This project is licensed under the MIT License – see the LICENSE file.

## Acknowledgements

We thank the OpenNeuro team and the AHEPA Hospital for making the data publicly available.

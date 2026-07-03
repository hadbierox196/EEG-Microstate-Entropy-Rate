#!/usr/bin/env python3
"""
reproduce.py
============

Fully self-contained, end-to-end reproduction pipeline for:

    "EEG Microstate Aggregate Conditional Entropy Derived from Markov
    Modelling as an AD-Specific Biomarker: Differentiating Alzheimer's
    Disease from Frontotemporal Dementia and Healthy Controls"
    Hassan Farooq, Sargodha Medical College.

This single script reproduces the entire analysis described in the
manuscript, from raw EEG through to the final statistics and figures:

    1.  Download resting-state EEG (OpenNeuro ds004504) if not present.
    2.  Preprocess each recording (1-40 Hz FIR bandpass, average
        reference, FastICA artefact rejection).
    3.  Extract 4-class EEG microstates via modified K-means on GFP peaks.
    4.  Build each participant's empirical first-order Markov transition
        matrix and compute the Aggregate Conditional Entropy, ACE(T) --
        the row-averaged conditional entropy in bits (log base 2), which
        is the paper's primary, discriminative biomarker.
    5.  Also compute the stationary-distribution-weighted entropy rate,
        H_rate(T) (Cover & Thomas), as the paper's non-discriminative
        control metric, and the occupancy entropy H(pi) control.
    6.  Run the full statistical battery: Kruskal-Wallis + eta-squared,
        Holm-Bonferroni-corrected pairwise permutation tests, bootstrap
        ROC/AUC with 95% CIs, and a sex-stratified sensitivity check.
    7.  Generate the group-comparison and ROC figures from the *actual*
        computed results (not illustrative placeholder numbers).

Everything needed is in this one file -- no imports from a local `src/`
package are required. It is safe to copy this file on its own into an
empty directory and run it there.

Usage
-----
    python reproduce.py                       # full pipeline
    python reproduce.py --data-root ./data    # custom data location
    python reproduce.py --skip-download       # data already present
    python reproduce.py --k 4 --n-perm 10000  # match paper's parameters
    python reproduce.py --demo                # synthetic smoke test,
                                               # no data / no MNE needed

Requirements
------------
See requirements.txt (mne, pycrostates, scikit-learn, scipy, numpy,
pandas, matplotlib, seaborn, tqdm). The heavy EEG dependencies (mne,
pycrostates) are imported lazily, inside the functions that need them,
so `--demo` mode and `--stats-only` mode work even if they are not
installed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants matching the manuscript (Methods, Sections 2.1-2.7)
# ---------------------------------------------------------------------------

DATASET_ID = "ds004504"
S3_URI = f"s3://openneuro.org/{DATASET_ID}"

DEFAULT_K = 4                 # canonical 4-microstate solution
BANDPASS_LOW_HZ = 1.0
BANDPASS_HIGH_HZ = 40.0
ICA_VARIANCE = 0.95            # FastICA, 95% explained variance retained
ICA_MAX_EOG_COMPONENTS = 2
RANDOM_STATE = 42
N_PERMUTATIONS = 10_000
N_BOOTSTRAP = 10_000
ALPHA = 0.05

GROUP_LETTER_MAP = {"C": "HC", "A": "AD", "F": "FTD"}
GROUP_ORDER = ["HC", "AD", "FTD"]
GROUP_COLORS = {"HC": "#0072B2", "AD": "#D55E00", "FTD": "#009E73"}  # Wong palette


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class SubjectResult:
    subject_id: str
    group: str
    sex: Optional[str] = None
    age: Optional[float] = None
    ACE: float = np.nan            # primary biomarker, bits, log2
    H_rate: float = np.nan         # stationary-weighted control, bits, log2
    H_occupancy: float = np.nan    # occupancy entropy control, bits, log2
    row_entropy: list = field(default_factory=list)  # per-row H(row_i), len K
    n_gfp_peaks: int = 0
    status: str = "ok"


# ---------------------------------------------------------------------------
# Step 1: data acquisition
# ---------------------------------------------------------------------------

def ensure_dataset(data_root: Path, skip_download: bool) -> None:
    """Download ds004504 from the public OpenNeuro S3 bucket if missing."""
    participants_tsv = data_root / "participants.tsv"
    if participants_tsv.exists():
        print(f"[data] Found existing dataset at {data_root}")
        return
    if skip_download:
        raise FileNotFoundError(
            f"--skip-download was set but {participants_tsv} does not exist. "
            "Either remove --skip-download or point --data-root at a valid "
            "local copy of OpenNeuro ds004504."
        )

    data_root.mkdir(parents=True, exist_ok=True)
    print(f"[data] Downloading {DATASET_ID} ({S3_URI}) -> {data_root}")
    print("[data] This is ~5 GB and can take 5-20 minutes.")

    if _which("aws") is not None:
        cmd = ["aws", "s3", "sync", "--no-sign-request", S3_URI, str(data_root)]
    else:
        # Fall back to openneuro-py if the AWS CLI isn't available.
        try:
            import openneuro  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Neither the AWS CLI ('aws') nor the 'openneuro-py' package "
                "is available. Install one of them, e.g.:\n"
                "    pip install awscli --break-system-packages\n"
                "or\n"
                "    pip install openneuro-py --break-system-packages"
            ) from exc
        cmd = [sys.executable, "-m", "openneuro", "download",
               "--dataset", DATASET_ID, "--target-dir", str(data_root)]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0 or not participants_tsv.exists():
        raise RuntimeError(
            f"Dataset download appears to have failed (exit code "
            f"{result.returncode}). Try running manually:\n  "
            f"aws s3 sync --no-sign-request {S3_URI} {data_root}"
        )


def _which(exe: str) -> Optional[str]:
    from shutil import which
    return which(exe)


def load_participants(data_root: Path) -> pd.DataFrame:
    participants = pd.read_csv(data_root / "participants.tsv", sep="\t")
    if "Group" not in participants.columns:
        raise ValueError(
            "participants.tsv does not contain a 'Group' column; is "
            "--data-root really pointing at ds004504?"
        )
    participants["group"] = participants["Group"].map(GROUP_LETTER_MAP)
    return participants


# ---------------------------------------------------------------------------
# Step 2: preprocessing (Methods 2.2)
# ---------------------------------------------------------------------------

def preprocess_subject(raw_path: Path):
    """Bandpass filter, average reference, FastICA artefact removal.

    Matches manuscript Section 2.2: zero-phase FIR bandpass 1-40 Hz,
    common-average reference, FastICA (95% explained variance) with
    automated EOG-correlation component rejection (<=2 components).
    """
    import mne
    import warnings
    warnings.filterwarnings("ignore")

    raw = mne.io.read_raw_eeglab(str(raw_path), preload=True, verbose=False)
    raw.pick_types(eeg=True, eog=False, stim=False)

    # NOTE: method='fir' (zero-phase), matching the manuscript's stated
    # filter design. Do not silently switch to 'iir' -- that changes the
    # phase response and is not what was reported.
    raw.filter(l_freq=BANDPASS_LOW_HZ, h_freq=BANDPASS_HIGH_HZ,
               method="fir", phase="zero", verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)

    ica = mne.preprocessing.ICA(
        n_components=ICA_VARIANCE,
        method="fastica",
        random_state=RANDOM_STATE,
        max_iter=800,
        verbose=False,
    )
    ica.fit(raw, verbose=False)
    try:
        eog_indices, _ = ica.find_bads_eog(raw, verbose=False)
        ica.exclude = eog_indices[:ICA_MAX_EOG_COMPONENTS]
    except Exception:
        ica.exclude = []

    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose=False)
    return raw_clean


# ---------------------------------------------------------------------------
# Step 3: microstate extraction (Methods 2.3)
# ---------------------------------------------------------------------------

def extract_microstates(raw_clean, K: int = DEFAULT_K):
    from pycrostates.cluster import ModKMeans
    from pycrostates.preprocessing import extract_gfp_peaks

    gfp_peaks = extract_gfp_peaks(raw_clean, picks="eeg")
    modk = ModKMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=100)
    modk.fit(gfp_peaks, verbose=False)
    segmentation = modk.predict(raw_clean, picks="eeg", verbose=False)
    labels = np.asarray(segmentation.labels)
    return labels, modk, len(gfp_peaks)


# ---------------------------------------------------------------------------
# Step 4: transition matrix + entropy metrics (Methods 2.4-2.6)
# ---------------------------------------------------------------------------

def empirical_transition_matrix(labels: np.ndarray, K: int) -> np.ndarray:
    """First-order empirical transition matrix T_ij = P(s_{t+1}=j | s_t=i)."""
    labels = labels[labels >= 0]  # drop unlabeled/rejected samples if any
    T = np.zeros((K, K))
    for t in range(len(labels) - 1):
        i, j = labels[t], labels[t + 1]
        T[i, j] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return T / row_sums


def stationary_distribution(T: np.ndarray) -> np.ndarray:
    """Left eigenvector of T for eigenvalue 1, normalised to sum to 1."""
    eigvals, eigvecs = np.linalg.eig(T.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    pi = np.real(eigvecs[:, idx])
    pi = np.abs(pi)
    total = pi.sum()
    return pi / total if total > 0 else np.full(T.shape[0], 1.0 / T.shape[0])


def per_row_entropy_bits(T: np.ndarray) -> np.ndarray:
    """H(row_i) = -sum_j T_ij * log2(T_ij), one value per source state i."""
    T_safe = np.clip(T, 1e-12, 1.0)
    return -np.sum(T_safe * np.log2(T_safe), axis=1)


def aggregate_conditional_entropy(T: np.ndarray) -> float:
    """ACE(T): the manuscript's PRIMARY, discriminative biomarker.

    Row-averaged Shannon conditional entropy of the transition matrix,
    in bits (log base 2):

        ACE(T) = (1/K) * sum_i [ -sum_j T_ij * log2(T_ij) ]

    Theoretical maximum = log2(K) (2.0 bits for K=4). This metric is
    NOT weighted by the stationary distribution -- see
    stationary_weighted_entropy_rate() below for that (non-discriminative,
    per manuscript Section 2.5) alternative.
    """
    K = T.shape[0]
    return float(np.mean(per_row_entropy_bits(T)))


def stationary_weighted_entropy_rate(T: np.ndarray, pi: np.ndarray) -> float:
    """H_rate(T): Cover & Thomas stationary-weighted entropy rate, bits.

    H_rate(T) = -sum_i pi_i * sum_j T_ij * log2(T_ij)

    Reported in the manuscript (Section 2.5) as a NON-discriminative
    control metric (Kruskal-Wallis p = 0.403 across HC/AD/FTD) -- included
    here for completeness / to allow the same sensitivity comparison, not
    as the paper's headline biomarker.
    """
    T_safe = np.clip(T, 1e-12, 1.0)
    return float(-np.sum(pi[:, None] * T_safe * np.log2(T_safe)))


def occupancy_entropy(pi: np.ndarray) -> float:
    """H(pi) = -sum_i pi_i * log2(pi_i), the occupancy-imbalance control."""
    pi_safe = np.clip(pi, 1e-12, 1.0)
    return float(-np.sum(pi_safe * np.log2(pi_safe)))


# ---------------------------------------------------------------------------
# Step 5: run the pipeline over all subjects
# ---------------------------------------------------------------------------

def run_pipeline(data_root: Path, K: int, task_label: str = "eyesclosed"
                  ) -> pd.DataFrame:
    participants = load_participants(data_root)
    rows = []

    try:
        from tqdm import tqdm
        iterator = tqdm(list(participants.iterrows()), total=len(participants))
    except ImportError:
        iterator = participants.iterrows()

    for _, row in iterator:
        subj = row["participant_id"]
        group = row.get("group")
        if group not in GROUP_ORDER:
            continue

        eeg_path = data_root / subj / "eeg" / f"{subj}_task-{task_label}_eeg.set"
        if not eeg_path.exists():
            rows.append(SubjectResult(subj, group, status="missing_file").__dict__)
            continue

        try:
            raw_clean = preprocess_subject(eeg_path)
            labels, _, n_peaks = extract_microstates(raw_clean, K=K)
            T = empirical_transition_matrix(labels, K=K)
            pi = stationary_distribution(T)

            result = SubjectResult(
                subject_id=subj,
                group=group,
                sex=row.get("Gender", row.get("Sex")),
                age=row.get("Age"),
                ACE=aggregate_conditional_entropy(T),
                H_rate=stationary_weighted_entropy_rate(T, pi),
                H_occupancy=occupancy_entropy(pi),
                row_entropy=per_row_entropy_bits(T).tolist(),
                n_gfp_peaks=n_peaks,
                status="ok",
            )
            rows.append(result.__dict__)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[pipeline] {subj}: FAILED ({exc})", file=sys.stderr)
            rows.append(SubjectResult(subj, group, status=f"error: {exc}").__dict__)

    df = pd.DataFrame(rows)
    n_ok = (df["status"] == "ok").sum() if len(df) else 0
    print(f"[pipeline] Completed: {n_ok}/{len(df)} subjects processed successfully.")
    return df


# ---------------------------------------------------------------------------
# Step 6: statistics (Methods 2.6-2.7, Results 3.1-3.6)
# ---------------------------------------------------------------------------

def permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int = N_PERMUTATIONS,
                      seed: int = RANDOM_STATE):
    """Two-tailed permutation test on the difference of means."""
    rng = np.random.default_rng(seed)
    obs = float(np.mean(a) - np.mean(b))
    combined = np.concatenate([a, b])
    n1 = len(a)
    diffs = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(combined)
        diffs[i] = shuffled[:n1].mean() - shuffled[n1:].mean()
    p = float(np.mean(np.abs(diffs) >= np.abs(obs)))
    return obs, p


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    pooled_sd = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1))
                         / (n1 + n2 - 2))
    return float((a.mean() - b.mean()) / pooled_sd) if pooled_sd > 0 else np.nan


def holm_bonferroni(pvals: dict) -> dict:
    """Holm-Bonferroni step-down correction. Returns {label: corrected_p}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    corrected = {}
    running_max = 0.0
    for rank, (label, p) in enumerate(items):
        adj = min((m - rank) * p, 1.0)
        running_max = max(running_max, adj)
        corrected[label] = running_max
    return corrected


def bootstrap_auc_ci(y_true: np.ndarray, scores: np.ndarray,
                      n_boot: int = N_BOOTSTRAP, seed: int = RANDOM_STATE):
    from sklearn.metrics import roc_auc_score
    from sklearn.utils import resample

    auc = roc_auc_score(y_true, scores)
    rng_seed = seed
    aucs = []
    for i in range(n_boot):
        idx = resample(np.arange(len(y_true)), random_state=rng_seed + i)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], scores[idx]))
    ci_low, ci_high = np.percentile(aucs, [2.5, 97.5])
    return float(auc), float(ci_low), float(ci_high)


def run_statistics(df: pd.DataFrame, n_perm: int = N_PERMUTATIONS) -> dict:
    from scipy.stats import kruskal, shapiro

    ok = df[df["status"] == "ok"].copy()
    groups = {g: ok.loc[ok["group"] == g, "ACE"].to_numpy() for g in GROUP_ORDER}
    n = {g: len(v) for g, v in groups.items()}
    print(f"[stats] N per group: {n}")

    stats: dict = {"n_per_group": n}

    # Shapiro-Wilk normality per group
    stats["shapiro"] = {
        g: {"W": float(shapiro(v)[0]), "p": float(shapiro(v)[1])}
        for g, v in groups.items() if len(v) >= 3
    }

    # Kruskal-Wallis + eta-squared
    H_stat, p_kw = kruskal(*groups.values())
    k = len(groups)
    N = sum(n.values())
    eta_sq = (H_stat - k + 1) / (N - k)
    stats["kruskal_wallis"] = {"H": float(H_stat), "p": float(p_kw),
                                "eta_sq": float(eta_sq)}

    # Pairwise permutation tests + Holm-Bonferroni + Cohen's d
    pairs = [("HC", "AD"), ("HC", "FTD"), ("AD", "FTD")]
    raw_p, deltas, d_vals = {}, {}, {}
    for a_name, b_name in pairs:
        delta, p = permutation_test(groups[a_name], groups[b_name], n_perm=n_perm)
        label = f"{a_name}-{b_name}"
        raw_p[label] = p
        deltas[label] = delta
        d_vals[label] = cohens_d(groups[a_name], groups[b_name])
    corrected_p = holm_bonferroni(raw_p)
    stats["pairwise"] = {
        label: {"delta_ACE": deltas[label], "p_raw": raw_p[label],
                "p_holm": corrected_p[label], "cohens_d": d_vals[label]}
        for label in raw_p
    }

    # ROC / AUC (AD lower ACE than HC, so flip sign for scoring)
    for target in ["AD", "FTD"]:
        y_true = np.concatenate([np.zeros(n["HC"]), np.ones(n[target])])
        scores = -np.concatenate([groups["HC"], groups[target]])
        auc, lo, hi = bootstrap_auc_ci(y_true, scores)
        stats[f"auc_{target}_vs_HC"] = {"auc": auc, "ci_low": lo, "ci_high": hi}

    # Occupancy-entropy control: should be NON-significant
    occ = {g: ok.loc[ok["group"] == g, "H_occupancy"].to_numpy() for g in GROUP_ORDER}
    H_occ, p_occ = kruskal(*occ.values())
    stats["occupancy_control_kruskal_wallis"] = {"H": float(H_occ), "p": float(p_occ)}

    # Stationary-weighted entropy rate control: should be NON-significant
    hrate = {g: ok.loc[ok["group"] == g, "H_rate"].to_numpy() for g in GROUP_ORDER}
    H_hr, p_hr = kruskal(*hrate.values())
    stats["hrate_control_kruskal_wallis"] = {"H": float(H_hr), "p": float(p_hr)}

    # Sex-stratified sensitivity check (if sex data available)
    if "sex" in ok.columns and ok["sex"].notna().any():
        stats["sex_stratified"] = {}
        for sex_val in sorted(ok["sex"].dropna().unique()):
            sub = ok[ok["sex"] == sex_val]
            hc = sub.loc[sub["group"] == "HC", "ACE"].to_numpy()
            ad = sub.loc[sub["group"] == "AD", "ACE"].to_numpy()
            if len(hc) >= 3 and len(ad) >= 3:
                delta, p = permutation_test(hc, ad, n_perm=n_perm)
                stats["sex_stratified"][str(sex_val)] = {
                    "n_HC": len(hc), "n_AD": len(ad),
                    "delta_ACE": delta, "p": p,
                }

    return stats


def print_stats_report(stats: dict) -> None:
    print("\n" + "=" * 72)
    print("STATISTICAL SUMMARY")
    print("=" * 72)
    kw = stats["kruskal_wallis"]
    print(f"Kruskal-Wallis (HC/AD/FTD): H(2)={kw['H']:.3f}, p={kw['p']:.4f}, "
          f"eta^2={kw['eta_sq']:.3f}")
    print("\nPairwise permutation tests (Holm-Bonferroni corrected):")
    for label, d in stats["pairwise"].items():
        print(f"  {label}: delta_ACE={d['delta_ACE']:+.4f} bits, "
              f"d={d['cohens_d']:+.3f}, p_raw={d['p_raw']:.4f}, "
              f"p_holm={d['p_holm']:.4f}")
    for target in ["AD", "FTD"]:
        a = stats[f"auc_{target}_vs_HC"]
        print(f"\nAUC ({target} vs HC): {a['auc']:.3f} "
              f"(95% CI {a['ci_low']:.3f}-{a['ci_high']:.3f})")
    occ = stats["occupancy_control_kruskal_wallis"]
    print(f"\nOccupancy-entropy control: H(2)={occ['H']:.3f}, p={occ['p']:.4f} "
          "(expected: non-significant)")
    hr = stats["hrate_control_kruskal_wallis"]
    print(f"Stationary-weighted H_rate control: H(2)={hr['H']:.3f}, "
          f"p={hr['p']:.4f} (expected: non-significant)")
    if "sex_stratified" in stats:
        print("\nSex-stratified HC vs AD:")
        for sex_val, d in stats["sex_stratified"].items():
            print(f"  sex={sex_val}: n_HC={d['n_HC']}, n_AD={d['n_AD']}, "
                  f"delta_ACE={d['delta_ACE']:+.4f}, p={d['p']:.4f}")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Step 7: figures, generated from the ACTUAL computed results
# ---------------------------------------------------------------------------

def make_figures(df: pd.DataFrame, stats: dict, out_dir: Path, K: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    out_dir.mkdir(parents=True, exist_ok=True)
    ok = df[df["status"] == "ok"]
    data = {g: ok.loc[ok["group"] == g, "ACE"].to_numpy() for g in GROUP_ORDER}

    # --- Figure: group distributions (violin + box) -----------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    parts = ax.violinplot([data[g] for g in GROUP_ORDER],
                           positions=range(len(GROUP_ORDER)),
                           showmeans=False, showextrema=False)
    for pc, g in zip(parts["bodies"], GROUP_ORDER):
        pc.set_facecolor(GROUP_COLORS[g])
        pc.set_alpha(0.6)
    for i, g in enumerate(GROUP_ORDER):
        jitter = np.random.default_rng(RANDOM_STATE).normal(i, 0.05, len(data[g]))
        ax.scatter(jitter, data[g], color=GROUP_COLORS[g], s=18, alpha=0.7,
                   edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels([f"{g}\n(n={len(data[g])})" for g in GROUP_ORDER])
    ax.set_ylabel("ACE(T) [bits]")
    ax.axhline(np.log2(K), color="gray", ls="--", lw=1.5, alpha=0.7,
               label=f"Max = log2({K})")
    kw = stats["kruskal_wallis"]
    ax.set_title(f"Group ACE(T)\nKruskal-Wallis p={kw['p']:.4f}, "
                  f"eta^2={kw['eta_sq']:.3f}")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    bp = ax.boxplot([data["HC"], data["AD"]], positions=[1, 2], widths=0.5,
                     patch_artist=True, medianprops=dict(color="k", lw=2))
    for patch, g in zip(bp["boxes"], ["HC", "AD"]):
        patch.set_facecolor(GROUP_COLORS[g])
        patch.set_alpha(0.7)
    hc_ad = stats["pairwise"]["HC-AD"]
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"HC\n(n={len(data['HC'])})", f"AD\n(n={len(data['AD'])})"])
    ax.set_ylabel("ACE(T) [bits]")
    ax.set_title(f"HC vs AD: delta={hc_ad['delta_ACE']:.3f} bits, "
                  f"d={hc_ad['cohens_d']:.3f}\n"
                  f"p_holm={hc_ad['p_holm']:.4f}")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("EEG Microstate Aggregate Conditional Entropy, ACE(T)",
                  fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "figure_group_ACE.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "figure_group_ACE.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- Figure: ROC curve, AD vs HC and FTD vs HC -------------------------
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for target, color in [("AD", GROUP_COLORS["AD"]), ("FTD", GROUP_COLORS["FTD"])]:
        y_true = np.concatenate([np.zeros(len(data["HC"])), np.ones(len(data[target]))])
        scores = -np.concatenate([data["HC"], data[target]])
        fpr, tpr, _ = roc_curve(y_true, scores)
        a = stats[f"auc_{target}_vs_HC"]
        ax.plot(fpr, tpr, color=color, lw=2.2,
                label=f"{target} vs HC (AUC={a['auc']:.3f}, "
                      f"CI {a['ci_low']:.2f}-{a['ci_high']:.2f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC: ACE(T) as a univariate classifier")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_roc.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "figure_roc.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[figures] Saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Demo mode: synthetic data so the statistics/figures/CLI can be smoke-tested
# without EEG data or mne/pycrostates installed.
# ---------------------------------------------------------------------------

def make_demo_dataframe(seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    specs = [("HC", 27, 0.521, 0.091), ("AD", 35, 0.454, 0.098),
             ("FTD", 23, 0.520, 0.110)]
    sexes = ["M", "F"]
    i = 0
    for group, n, mean, sd in specs:
        for _ in range(n):
            i += 1
            T = rng.dirichlet(np.ones(DEFAULT_K), size=DEFAULT_K)
            pi = stationary_distribution(T)
            rows.append(SubjectResult(
                subject_id=f"sub-{i:03d}",
                group=group,
                sex=sexes[rng.integers(0, 2)],
                age=float(rng.normal(66, 7)),
                ACE=float(np.clip(rng.normal(mean, sd), 0, np.log2(DEFAULT_K))),
                H_rate=stationary_weighted_entropy_rate(T, pi),
                H_occupancy=occupancy_entropy(pi),
                row_entropy=per_row_entropy_bits(T).tolist(),
                n_gfp_peaks=int(rng.integers(2000, 6000)),
                status="ok",
            ).__dict__)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reproduce the EEG microstate ACE(T) Alzheimer's biomarker pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=Path, default=Path("./data_mci_ad"),
                   help="Local path for / to OpenNeuro ds004504")
    p.add_argument("--results-dir", type=Path, default=Path("./results"),
                   help="Where to write results CSV and figures")
    p.add_argument("--k", type=int, default=DEFAULT_K,
                   help="Number of microstate classes")
    p.add_argument("--n-perm", type=int, default=N_PERMUTATIONS,
                   help="Number of permutations for permutation tests")
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--skip-download", action="store_true",
                   help="Do not attempt to download data; require it to exist")
    p.add_argument("--stats-only", action="store_true",
                   help="Skip EEG processing; recompute stats from an "
                        "existing results CSV in --results-dir")
    p.add_argument("--demo", action="store_true",
                   help="Run on synthetic data (no EEG data or mne/pycrostates "
                        "required) to smoke-test statistics and figures")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    np.random.seed(args.seed)
    global RANDOM_STATE, N_PERMUTATIONS
    RANDOM_STATE = args.seed
    N_PERMUTATIONS = args.n_perm

    args.results_dir.mkdir(parents=True, exist_ok=True)
    results_csv = args.results_dir / "ace_results.csv"

    if args.demo:
        print("[mode] DEMO: using synthetic data, no EEG dependencies required.")
        df = make_demo_dataframe(seed=args.seed)
    elif args.stats_only:
        print(f"[mode] STATS-ONLY: loading existing results from {results_csv}")
        if not results_csv.exists():
            print(f"ERROR: {results_csv} not found. Run without --stats-only first.",
                  file=sys.stderr)
            return 1
        df = pd.read_csv(results_csv)
        df["row_entropy"] = df["row_entropy"].apply(
            lambda s: json.loads(s) if isinstance(s, str) else s)
    else:
        ensure_dataset(args.data_root, args.skip_download)
        df = run_pipeline(args.data_root, K=args.k)
        df.to_csv(results_csv, index=False)
        print(f"[pipeline] Results written to {results_csv}")

    ok = df[df["status"] == "ok"]
    for g in GROUP_ORDER:
        n = (ok["group"] == g).sum()
        if n < 3:
            print(f"ERROR: group '{g}' has only {n} usable subjects; need >= 3 "
                  "per group for statistics. Check data availability / "
                  "preprocessing failures.", file=sys.stderr)
            return 1

    print("\n[summary] Descriptive statistics (ACE, bits):")
    print(ok.groupby("group")["ACE"].agg(["count", "mean", "std"]).round(4)
          .reindex(GROUP_ORDER))

    stats = run_statistics(df, n_perm=args.n_perm)
    print_stats_report(stats)

    stats_json = args.results_dir / "statistics_summary.json"
    with open(stats_json, "w") as f:
        json.dump(stats, f, indent=2, default=float)
    print(f"[stats] Full statistics saved to {stats_json}")

    make_figures(df, stats, args.results_dir / "figures", K=args.k)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

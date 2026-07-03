#!/usr/bin/env python
"""Generate all publication-ready figures from the results CSV."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import resample


def kruskal_wallis_with_eta(groups_dict):
    stat, p = kruskal(*groups_dict.values())
    k = len(groups_dict)
    N = sum(len(v) for v in groups_dict.values())
    eta_sq = (stat - k + 1) / (N - k)
    return stat, p, eta_sq


def permutation_test(a, b, n_perm=10000, seed=42):
    rng = np.random.default_rng(seed)
    obs = np.mean(a) - np.mean(b)
    combined = np.concatenate([a, b])
    n1 = len(a)
    diffs = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(combined)
        diffs[i] = shuffled[:n1].mean() - shuffled[n1:].mean()
    p = np.mean(np.abs(diffs) >= np.abs(obs))
    return obs, p


def bootstrap_auc_ci(y_true, scores, n_boot=10000, seed=42):
    auc = roc_auc_score(y_true, scores)
    aucs = []
    for i in range(n_boot):
        idx = resample(np.arange(len(y_true)), random_state=seed + i)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], scores[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return auc, lo, hi


def make_dual_panel_figure(df, K=4, save_dir='results/figures'):
    """
    Dual-panel figure computed from the ACTUAL results DataFrame:
      (A) Violin plot of ACE(T) across groups with real Kruskal-Wallis stats.
      (B) Box plot HC vs AD with real permutation-test delta/p and AUC.
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.8, rc={"lines.linewidth": 2.5})
    colors = {"HC": "#0072B2", "FTD": "#009E73", "AD": "#D55E00"}

    groups = ['HC', 'FTD', 'AD']
    H_values = {g: df[df['group'] == g]['ACE'].values for g in groups}

    stat, p_kw, eta_sq = kruskal_wallis_with_eta(H_values)
    delta_hc_ad, p_hc_ad = permutation_test(H_values['HC'], H_values['AD'])

    y_true = np.concatenate([np.zeros(len(H_values['HC'])), np.ones(len(H_values['AD']))])
    scores = -np.concatenate([H_values['HC'], H_values['AD']])
    auc, ci_lo, ci_hi = bootstrap_auc_ci(y_true, scores)

    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

    # ---- Panel A: group violin plot ----
    ax1 = fig.add_subplot(gs[0])
    for i, g in enumerate(groups):
        parts = ax1.violinplot(H_values[g], positions=[i], widths=0.6,
                                showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(colors[g])
            pc.set_alpha(0.6)
        jitter = np.random.default_rng(42).normal(i, 0.05, len(H_values[g]))
        ax1.scatter(jitter, H_values[g], s=40, color=colors[g], alpha=0.7,
                    edgecolor='white', linewidth=0.5)
    ax1.axhline(y=np.log2(K), color='gray', linestyle='--', lw=2, alpha=0.6,
                label=f'Max = log2({K})')
    ax1.set_xticks(range(3))
    ax1.set_xticklabels([f"{g}\n(n={len(H_values[g])})" for g in groups], fontsize=13)
    ax1.set_ylabel('ACE(T) [bits]', fontsize=14, weight='bold')
    ax1.set_title(f'Kruskal-Wallis H(2)={stat:.3f}, p={p_kw:.4f}, η²={eta_sq:.3f}',
                  fontsize=12)
    ax1.legend(fontsize=9)
    ax1.spines[['top', 'right']].set_visible(False)

    # ---- Panel B: HC vs AD box plot ----
    ax2 = fig.add_subplot(gs[1])
    bp = ax2.boxplot([H_values['HC'], H_values['AD']], positions=[0, 1], widths=0.4,
                      patch_artist=True, showfliers=False,
                      medianprops=dict(color='black', linewidth=2))
    for patch, g in zip(bp['boxes'], ['HC', 'AD']):
        patch.set_facecolor(colors[g])
        patch.set_alpha(0.7)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([f"HC\n(n={len(H_values['HC'])})", f"AD\n(n={len(H_values['AD'])})"],
                         fontsize=13)
    ax2.set_ylabel('ACE(T) [bits]', fontsize=14, weight='bold')
    ax2.set_title(f'Δ={delta_hc_ad:.3f} bits, p={p_hc_ad:.4f}, AUC={auc:.3f} '
                  f'[{ci_lo:.2f}-{ci_hi:.2f}]', fontsize=12)
    ax2.spines[['top', 'right']].set_visible(False)

    fig.suptitle("EEG Microstate ACE(T): A Physics-Grounded Biomarker for Alzheimer's Disease",
                 fontsize=16, weight='bold', y=1.05)

    plt.savefig(os.path.join(save_dir, 'figure_panel_A_B.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'figure_panel_A_B.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_roc_curve(df, save_dir='results/figures'):
    """Standalone ROC curve for AD vs HC, computed from real results."""
    os.makedirs(save_dir, exist_ok=True)
    H_HC = df[df['group'] == 'HC']['ACE'].values
    H_AD = df[df['group'] == 'AD']['ACE'].values

    y_true = np.concatenate([np.zeros(len(H_HC)), np.ones(len(H_AD))])
    scores = -np.concatenate([H_HC, H_AD])
    auc, ci_lo, ci_hi = bootstrap_auc_ci(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#0072B2', lw=2.5,
             label=f'AUC = {auc:.3f} [{ci_lo:.2f}-{ci_hi:.2f}]')
    plt.plot([0, 1], [0, 1], '--', color='gray', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC – AD vs HC')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.pdf'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300)
    plt.close()


def main():
    results_path = 'results/entropy_rate_results.csv'
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Run the pipeline first.")
        return
    df = pd.read_csv(results_path)
    print("Generating figures...")
    make_dual_panel_figure(df, K=4)
    generate_roc_curve(df)
    print("Figures saved to results/figures/")


if __name__ == '__main__':
    main()

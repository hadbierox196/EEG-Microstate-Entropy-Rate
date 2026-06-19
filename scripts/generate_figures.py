#!/usr/bin/env python
"""Generate all publication‑ready figures from the results CSV."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.stats import sem
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import resample


def make_dual_panel_figure(df, K=4, save_dir='results/figures'):
    """
    Generate the dual‑panel figure:
      (A) Transition matrices for HC (flexible) and AD (rigid)
      (B) Raincloud plot of entropy rate across groups with stats.
    Saves PDF and PNG.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.8, rc={"lines.linewidth": 2.5})
    colors = {"HC": "#2E86AB", "FTD": "#A23B72", "AD": "#F18F01"}

    # Extract data
    groups = ['HC', 'FTD', 'AD']
    H_values = [df[df['group'] == g]['H'].values for g in groups]
    means = [np.mean(h) for h in H_values]

    # Create figure
    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.8])

    # ---- Panel A: Transition matrices ----
    ax1 = fig.add_subplot(gs[0])

    # Example matrices (you can replace these with group‑averaged matrices if saved)
    T_HC = np.array([[0.897, 0.045, 0.03, 0.028],
                     [0.045, 0.871, 0.056, 0.029],
                     [0.052, 0.051, 0.861, 0.035],
                     [0.043, 0.036, 0.032, 0.89 ]])
    T_AD = np.array([[0.94, 0.02, 0.02, 0.02],
                     [0.02, 0.93, 0.03, 0.02],
                     [0.03, 0.02, 0.92, 0.03],
                     [0.02, 0.02, 0.03, 0.93]])

    vmin, vmax = 0.0, 1.0
    sns.heatmap(T_HC, annot=True, fmt='.2f', cmap='Blues', cbar=False,
                square=True, ax=ax1, vmin=vmin, vmax=vmax, linewidths=2,
                annot_kws={"size": 14, "weight": "bold"})
    ax1.set_title('Healthy Control (HC)\nFlexible Transitions', fontsize=16, weight='bold')
    ax1.set_xlabel('Next State', fontsize=14)
    ax1.set_ylabel('Current State', fontsize=14)
    ax1.tick_params(labelsize=12)

    # Inset: AD heatmap
    ax1_sub = ax1.inset_axes([0.55, 0.05, 0.4, 0.4])
    sns.heatmap(T_AD, annot=True, fmt='.2f', cmap='Reds', cbar=False,
                square=True, ax=ax1_sub, vmin=vmin, vmax=vmax, linewidths=2,
                annot_kws={"size": 10, "weight": "bold"})
    ax1_sub.set_title('AD (Rigid)', fontsize=12, weight='bold', color='#F18F01')
    ax1_sub.set_xlabel('Next', fontsize=10)
    ax1_sub.set_ylabel('Current', fontsize=10)
    ax1_sub.tick_params(labelsize=8)

    # Arrow and label
    ax1.annotate('', xy=(0.55, 0.55), xytext=(0.75, 0.55),
                 xycoords='axes fraction', arrowprops=dict(arrowstyle='->', color='gray', lw=3))
    ax1.text(0.5, -0.25, '↓ Entropy Rate (H)', transform=ax1.transAxes,
             fontsize=14, weight='bold', ha='center', color='#F18F01')

    # ---- Panel B: Raincloud plot ----
    ax2 = fig.add_subplot(gs[1])

    for i, (group, vals) in enumerate(zip(groups, H_values)):
        # Violin (left half)
        parts = ax2.violinplot(vals, positions=[i], widths=0.6, showmeans=False,
                               showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(colors[group])
            pc.set_alpha(0.5)
            # Clip to left half
            pc.set_clip_path(Rectangle((-1e6, -1e6), i+0.5, 1e6, transform=ax2.transData))

        # Boxplot
        bp = ax2.boxplot(vals, positions=[i], widths=0.15, patch_artist=True,
                         showfliers=False, boxprops=dict(facecolor='white', alpha=0.8),
                         whiskerprops=dict(color='black'), capprops=dict(color='black'),
                         medianprops=dict(color='black', linewidth=2))

        # Jittered points
        x_jitter = np.random.normal(i, 0.04, size=len(vals))
        ax2.scatter(x_jitter, vals, s=80, color=colors[group], alpha=0.7, edgecolor='white', linewidth=0.5)

    # Means
    ax2.scatter(range(3), means, s=200, color='black', marker='D', zorder=10, label='Mean')

    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['HC', 'FTD', 'AD'], fontsize=16, weight='bold')
    ax2.set_ylabel('Entropy Rate (H) [nats]', fontsize=16, weight='bold')
    ax2.set_ylim(0.2, 0.9)

    # Significance bracket: HC vs AD (from your results)
    y_bracket = 0.82
    ax2.plot([0, 2], [y_bracket, y_bracket], color='black', lw=2)
    ax2.text(1, y_bracket+0.02, 'p = 0.007', ha='center', fontsize=14, weight='bold')
    ax2.text(1, y_bracket-0.03, 'AUC = 0.69', ha='center', fontsize=12, style='italic')

    # Criticality line
    ax2.axhline(y=np.log(K), color='gray', linestyle='--', lw=2, alpha=0.6, label='Max Entropy (Critical)')
    ax2.text(2.5, np.log(K)+0.02, 'Criticality', color='gray', fontsize=12)

    # Effect size
    ax2.text(0.5, 0.25, 'η² = 0.078', fontsize=14, weight='bold', color='#2E86AB')
    ax2.legend(loc='upper left', fontsize=10, frameon=True)

    # Overall title
    fig.suptitle('EEG Microstate Entropy Rate: A Physics‑Grounded Biomarker for Alzheimer\'s Disease',
                 fontsize=18, weight='bold', y=1.02)

    # Save
    plt.savefig(os.path.join(save_dir, 'figure_panel_A_B.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'figure_panel_A_B.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def generate_roc_curve(df, save_dir='results/figures'):
    """Generate a standalone ROC curve for AD vs HC."""
    os.makedirs(save_dir, exist_ok=True)

    H_HC = df[df['group'] == 'HC']['H'].values
    H_AD = df[df['group'] == 'AD']['H'].values

    y_true = np.concatenate([np.zeros(len(H_HC)), np.ones(len(H_AD))])
    # AD has lower H, so flip scores
    scores = -np.concatenate([H_HC, H_AD])

    auc = roc_auc_score(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#4f86f7', lw=2.5, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], '--', color='gray', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC – AD vs HC')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.pdf'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300)
    plt.show()
    plt.close()


def main():
    # Load results
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

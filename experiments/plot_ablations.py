"""
LST Ablation Plotting Suite
=============================
Generates all figures needed for the paper ablation section:

  Fig A: Loss vs Wall-Clock Time (THE key plot)
  Fig B: GA Ablation — speedup and degradation bar charts
  Fig C: Acceptance rate comparison (K=5 vs K=20) over training
  Fig D: Hybrid schedule timeline — phase boundaries + loss
  Fig E: Long training (10K) acceptance trajectory
  Fig F: Combined results table (printed, not plotted)

Usage:
    python experiments/plot_ablations.py
    # Or from Colab:
    %run experiments/plot_ablations.py

Reads checkpoints from checkpoints/ directory.
Outputs to paper/figures/ablation_*.pdf
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Consistent style for publication
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.8,
})

COLORS = {
    'lst': '#2196F3',       # Blue
    'baseline': '#FF5722',  # Orange-red
    'hybrid': '#4CAF50',    # Green
    'quality': '#9C27B0',   # Purple
    'ga1': '#FF9800',       # Orange
    'ga2': '#00BCD4',       # Teal
    'ga4': '#2196F3',       # Blue
}

OUT_DIR = os.path.join('paper', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)


def load_checkpoint(name):
    """Load a checkpoint pickle if it exists, else return None."""
    path = os.path.join('checkpoints', name)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def smooth(values, window=20):
    """Exponential moving average smoothing."""
    if len(values) == 0:
        return values
    smoothed = []
    val = values[0]
    alpha = 2.0 / (window + 1)
    for v in values:
        val = alpha * v + (1 - alpha) * val
        smoothed.append(val)
    return smoothed


# ====================================================================
#  FIG A: LOSS VS WALL-CLOCK TIME (Key Plot)
# ====================================================================
def plot_loss_vs_walltime():
    """
    THE most important plot. Shows loss (Y) vs cumulative wall-clock seconds (X)
    for LST and baseline. This is the fair comparison because LST steps are
    cheaper than baseline steps.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False

    for name, label, color, ls in [
        ('quality_focused', 'LST (K=5, quality)', COLORS['quality'], '-'),
        ('hybrid', 'LST Hybrid (LST 80%→Std 20%)', COLORS['hybrid'], '-'),
        ('original', 'LST (K=20, original)', COLORS['lst'], '--'),
    ]:
        data = load_checkpoint(f'lst_{name}.pkl')
        if data is None:
            continue

        cumtime = np.cumsum(data['step_times'])
        losses = smooth(data['losses'], window=30)
        ax.plot(cumtime, losses, color=color, linestyle=ls, label=label, alpha=0.9)
        plotted = True

    # Baseline (use quality_focused baseline, or original)
    for name in ['quality_focused', 'original']:
        data = load_checkpoint(f'baseline_{name}.pkl')
        if data is None:
            continue
        cumtime = np.cumsum(data['baseline_times'])
        losses = smooth(data['baseline_losses'], window=30)
        ax.plot(cumtime, losses, color=COLORS['baseline'], linestyle='-',
                label='Standard Training (Baseline)', alpha=0.9)
        plotted = True
        break  # Only plot one baseline

    if not plotted:
        print("  [SKIP] loss_vs_walltime — no data found")
        plt.close(fig)
        return

    ax.set_xlabel('Wall-Clock Time (seconds)')
    ax.set_ylabel('Training Loss')
    ax.set_title('Loss vs. Wall-Clock Time')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Secondary x-axis in minutes
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/60:.0f}'))
    ax2.set_xlabel('Time (minutes)')

    path = os.path.join(OUT_DIR, 'ablation_loss_vs_walltime.pdf')
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)

    # Also save PNG for Colab preview
    path_png = path.replace('.pdf', '.png')
    fig2, ax = plt.subplots(figsize=(8, 5))
    # Re-plot for PNG (matplotlib can't save after close)
    for name, label, color, ls in [
        ('quality_focused', 'LST (K=5, quality)', COLORS['quality'], '-'),
        ('hybrid', 'LST Hybrid (LST 80%→Std 20%)', COLORS['hybrid'], '-'),
        ('original', 'LST (K=20, original)', COLORS['lst'], '--'),
    ]:
        data = load_checkpoint(f'lst_{name}.pkl')
        if data is None:
            continue
        cumtime = np.cumsum(data['step_times'])
        losses_s = smooth(data['losses'], window=30)
        ax.plot(cumtime, losses_s, color=color, linestyle=ls, label=label, alpha=0.9)

    for name in ['quality_focused', 'original']:
        data = load_checkpoint(f'baseline_{name}.pkl')
        if data is None:
            continue
        cumtime = np.cumsum(data['baseline_times'])
        losses_s = smooth(data['baseline_losses'], window=30)
        ax.plot(cumtime, losses_s, color=COLORS['baseline'], linestyle='-',
                label='Standard Training', alpha=0.9)
        break

    ax.set_xlabel('Wall-Clock Time (seconds)')
    ax.set_ylabel('Training Loss')
    ax.set_title('Loss vs. Wall-Clock Time')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig2.savefig(path_png)
    print(f"  Saved: {path_png}")
    plt.close(fig2)


# ====================================================================
#  FIG B: GA ABLATION BAR CHART
# ====================================================================
def plot_ga_ablation():
    """
    Bar chart: speedup and loss degradation for GA=1, GA=2, GA=4.
    This addresses the concern that GA, not the draft model, drives speedup.
    """
    ga_configs = [
        ('ga1', 'GA=1'),
        ('ga2', 'GA=2'),
        ('quality_focused', 'GA=4'),  # quality_focused uses GA=4
    ]

    speedups = []
    degradations = []
    labels = []

    for name, label in ga_configs:
        lst = load_checkpoint(f'lst_{name}.pkl')
        base = load_checkpoint(f'baseline_{name}.pkl')
        if lst is None or base is None:
            continue

        lst_time = lst['total_time']
        base_time = base['baseline_total_time']
        lst_loss = np.mean(lst['losses'][-50:])
        base_loss = np.mean(base['baseline_losses'][-50:])

        speedups.append(base_time / lst_time)
        degradations.append((lst_loss - base_loss) / base_loss * 100)
        labels.append(label)

    if not speedups:
        print("  [SKIP] ga_ablation — no data found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    x = np.arange(len(labels))
    bars1 = ax1.bar(x, speedups, 0.6, color=[COLORS['ga1'], COLORS['ga2'], COLORS['ga4']])
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Speedup (×)')
    ax1.set_title('Wall-Clock Speedup')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No speedup')
    ax1.set_ylim(bottom=0)
    for bar, val in zip(bars1, speedups):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.2f}×', ha='center', va='bottom', fontweight='bold')

    bars2 = ax2.bar(x, degradations, 0.6, color=[COLORS['ga1'], COLORS['ga2'], COLORS['ga4']])
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Loss Degradation (%)')
    ax2.set_title('Quality Impact')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=5, color='red', linestyle=':', alpha=0.5, label='5% threshold')
    for bar, val in zip(bars2, degradations):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

    fig.suptitle('Gradient Accumulation Ablation', fontsize=14, fontweight='bold')
    fig.tight_layout()

    path = os.path.join(OUT_DIR, 'ablation_ga.pdf')
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


# ====================================================================
#  FIG C: ACCEPTANCE RATE COMPARISON (K=5 vs K=20)
# ====================================================================
def plot_acceptance_comparison():
    """
    Line plot of acceptance rate over training steps for different configs.
    Shows how K=5 with stricter tolerance maintains or differs from K=20.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False

    for name, label, color, ls in [
        ('quality_focused', 'K=5, tol_floor=0.02', COLORS['quality'], '-'),
        ('hybrid', 'Hybrid (K=20 → standard)', COLORS['hybrid'], '-'),
        ('original', 'K=20, original', COLORS['lst'], '--'),
    ]:
        data = load_checkpoint(f'lst_{name}.pkl')
        if data is None:
            continue
        if not data.get('acceptance_rates'):
            continue

        ar = data['acceptance_rates']
        steps = np.arange(1, len(ar) + 1)
        ar_smooth = smooth(ar, window=50)
        ax.plot(steps, [a * 100 for a in ar_smooth], color=color, linestyle=ls,
                label=label, alpha=0.9)
        plotted = True

    if not plotted:
        print("  [SKIP] acceptance_comparison — no data found")
        plt.close(fig)
        return

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Acceptance Rate (%)')
    ax.set_title('Acceptance Rate Over Training')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    path = os.path.join(OUT_DIR, 'ablation_acceptance.pdf')
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


# ====================================================================
#  FIG D: HYBRID SCHEDULE TIMELINE
# ====================================================================
def plot_hybrid_timeline():
    """
    Shows hybrid training: LST phase → standard phase with clear boundary.
    Overlays loss curve with phase annotation.
    """
    data = load_checkpoint('lst_hybrid.pkl')
    if data is None:
        print("  [SKIP] hybrid_timeline — no data found")
        return

    fig, ax1 = plt.subplots(figsize=(9, 4.5))

    steps = np.arange(1, len(data['losses']) + 1)
    losses = smooth(data['losses'], window=30)
    ax1.plot(steps, losses, color=COLORS['hybrid'], label='Hybrid LST Loss', alpha=0.9)

    # Baseline overlay
    for name in ['quality_focused', 'original']:
        base = load_checkpoint(f'baseline_{name}.pkl')
        if base is None:
            continue
        base_losses = smooth(base['baseline_losses'], window=30)
        base_steps = np.arange(1, len(base_losses) + 1)
        ax1.plot(base_steps, base_losses, color=COLORS['baseline'], linestyle='--',
                label='Standard Baseline', alpha=0.7)
        break

    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Training Loss')

    # Phase boundary
    switch_step = data['config'].get('lst_hybrid_switch_step', 1600)
    ax1.axvline(x=switch_step, color='red', linestyle=':', alpha=0.7, linewidth=2)
    ylim = ax1.get_ylim()
    mid_y = (ylim[0] + ylim[1]) * 0.5

    ax1.annotate('LST Phase\n(speculative)', xy=(switch_step * 0.5, mid_y),
                fontsize=11, ha='center', color=COLORS['hybrid'], fontweight='bold')
    ax1.annotate('Standard Phase\n(fine-tuning)', xy=(switch_step + (data['steps_done'] - switch_step) * 0.5, mid_y),
                fontsize=11, ha='center', color=COLORS['baseline'], fontweight='bold')

    ax1.set_title('Hybrid Schedule: LST → Standard Training')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    path = os.path.join(OUT_DIR, 'ablation_hybrid.pdf')
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


# ====================================================================
#  FIG E: LONG TRAINING ACCEPTANCE TRAJECTORY
# ====================================================================
def plot_long_training():
    """
    10K-step acceptance trajectory + loss curve.
    Answers the question: does acceptance rate stabilize or collapse?
    """
    data = load_checkpoint('lst_long_10k.pkl')
    if data is None:
        print("  [SKIP] long_training — no data found")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={'height_ratios': [1, 1]})

    steps = np.arange(1, len(data['losses']) + 1)

    # Loss curve
    losses = smooth(data['losses'], window=50)
    ax1.plot(steps, losses, color=COLORS['lst'], label='LST Loss', alpha=0.9)

    base = load_checkpoint('baseline_long_10k.pkl')
    if base is not None:
        base_losses = smooth(base['baseline_losses'], window=50)
        base_steps = np.arange(1, len(base_losses) + 1)
        ax1.plot(base_steps, base_losses, color=COLORS['baseline'], linestyle='--',
                label='Baseline Loss', alpha=0.7)

    ax1.set_ylabel('Training Loss')
    ax1.set_title('10K-Step Long Training')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Acceptance rate
    if data.get('acceptance_rates'):
        ar = data['acceptance_rates']
        ar_steps = np.arange(1, len(ar) + 1)
        ar_smooth = smooth(ar, window=100)
        ax2.plot(ar_steps, [a * 100 for a in ar_smooth], color=COLORS['lst'], alpha=0.9)
        ax2.fill_between(ar_steps, [a * 100 for a in ar_smooth], alpha=0.15, color=COLORS['lst'])

    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Acceptance Rate (%)')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'ablation_long_training.pdf')
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


# ====================================================================
#  TABLE: COMBINED RESULTS
# ====================================================================
def print_results_table():
    """Print a publication-ready results table."""
    configs = [
        ('original', 'Original (K=20, GA=4)'),
        ('quality_focused', 'Quality (K=5, tol≥0.02)'),
        ('hybrid', 'Hybrid (LST→Std)'),
        ('ga1', 'GA=1, K=20'),
        ('ga2', 'GA=2, K=20'),
        ('long_10k', '10K Steps'),
    ]

    print(f"\n{'='*100}")
    print(f"{'Config':<25} {'Time(s)':>9} {'Loss':>7} {'Accept%':>8} {'Base Time':>10} {'Base Loss':>10} {'Speed':>7} {'Degrad':>8}")
    print(f"{'='*100}")

    for name, label in configs:
        lst = load_checkpoint(f'lst_{name}.pkl')
        if lst is None:
            continue

        lst_time = lst['total_time']
        lst_loss = np.mean(lst['losses'][-50:])
        accept = lst['acceptance_rates'][-1] * 100 if lst.get('acceptance_rates') else 0

        base = load_checkpoint(f'baseline_{name}.pkl')
        if base is not None:
            bt = base['baseline_total_time']
            bl = np.mean(base['baseline_losses'][-50:])
            speedup = f'{bt/lst_time:.2f}×'
            degrad = f'{(lst_loss - bl) / bl * 100:+.1f}%'
            base_info = f'{bt:>10.0f} {bl:>10.2f}'
        else:
            speedup = '--'
            degrad = '--'
            base_info = f'{"--":>10} {"--":>10}'

        print(f"  {label:<23} {lst_time:>9.0f} {lst_loss:>7.2f} {accept:>7.1f}% {base_info} {speedup:>7} {degrad:>8}")

    print(f"{'='*100}")


# ====================================================================
#  MAIN
# ====================================================================
def plot_all():
    """Generate all ablation figures from available checkpoints."""
    print("Generating ablation figures...")
    print(f"Reading from: checkpoints/")
    print(f"Writing to:   {OUT_DIR}/\n")

    plot_loss_vs_walltime()
    plot_ga_ablation()
    plot_acceptance_comparison()
    plot_hybrid_timeline()
    plot_long_training()
    print_results_table()

    print(f"\nDone! Figures saved to {OUT_DIR}/")


if __name__ == '__main__':
    plot_all()

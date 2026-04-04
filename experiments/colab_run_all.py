"""
LST Ablation Experiment Notebook — Colab Ready
=================================================
Copy this entire file into a single Colab cell, or split at the marked
"# === CELL ===" boundaries.

Requirements: A100 40GB GPU runtime.
Estimated total time: ~3 hours for Exp 1-3, ~9 hours with Exp 4.
Each experiment saves independently — safe to interrupt.

Run order (priority):
  1. quality_focused (K=5, tol_floor=0.02) — fixes the paper's biggest flaw
  2. hybrid (LST 80% → standard 20%) — targets ~0% degradation
  3. ga1 + ga2 (GA ablation) — isolates GA contribution
  4. long_10k (10K steps) — only if time permits
"""

# === CELL 1: Setup ===
# !pip install transformers datasets
# %cd /content
# !git clone https://github.com/RAVINDRA8008/Learned-Speculative-Training-LST-.git LST
# %cd LST
# !pip install -e .

# === CELL 2: Verify GPU ===
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
assert torch.cuda.is_available(), "Need GPU!"

# === CELL 3: Run Experiment 1 — Quality-Focused LST ===
import sys, os
sys.path.insert(0, '/content/LST')
os.chdir('/content/LST')

from experiments.run_ablations import run_experiment

print("=" * 70)
print("EXP 1: Quality-Focused LST (K=5, tol_min=0.02)")
print("Expected time: ~75 min (LST + baseline)")
print("=" * 70)
lst_qf, base_qf = run_experiment('quality_focused')

# Quick summary
import numpy as np
lst_loss = np.mean(lst_qf['losses'][-50:])
base_loss = np.mean(base_qf['baseline_losses'][-50:])
speedup = base_qf['baseline_total_time'] / lst_qf['total_time']
degrad = (lst_loss - base_loss) / base_loss * 100
print(f"\n>>> RESULT: Speedup={speedup:.2f}x, Degradation={degrad:.1f}%, Accept={lst_qf['acceptance_rates'][-1]:.1%}")

# === CELL 4: Run Experiment 2 — Hybrid Schedule ===
from experiments.run_ablations import run_experiment

print("=" * 70)
print("EXP 2: Hybrid LST (LST 80% → Standard 20%)")
print("Expected time: ~30 min (no separate baseline needed)")
print("=" * 70)
lst_hybrid, _ = run_experiment('hybrid', skip_baseline=True)

lst_loss_h = np.mean(lst_hybrid['losses'][-50:])
degrad_h = (lst_loss_h - base_loss) / base_loss * 100
speedup_h = base_qf['baseline_total_time'] / lst_hybrid['total_time']
print(f"\n>>> RESULT: Speedup={speedup_h:.2f}x, Degradation={degrad_h:.1f}%")

# === CELL 5: Run Experiment 3a — GA=1 Ablation ===
from experiments.run_ablations import run_experiment

print("=" * 70)
print("EXP 3a: GA=1 Ablation")
print("Expected time: ~24 min")
print("=" * 70)
lst_ga1, base_ga1 = run_experiment('ga1')

lst_loss_ga1 = np.mean(lst_ga1['losses'][-50:])
base_loss_ga1 = np.mean(base_ga1['baseline_losses'][-50:])
speedup_ga1 = base_ga1['baseline_total_time'] / lst_ga1['total_time']
degrad_ga1 = (lst_loss_ga1 - base_loss_ga1) / base_loss_ga1 * 100
print(f"\n>>> RESULT: GA=1 Speedup={speedup_ga1:.2f}x, Degradation={degrad_ga1:.1f}%")

# === CELL 6: Run Experiment 3b — GA=2 Ablation ===
from experiments.run_ablations import run_experiment

print("=" * 70)
print("EXP 3b: GA=2 Ablation")
print("Expected time: ~42 min")
print("=" * 70)
lst_ga2, base_ga2 = run_experiment('ga2')

lst_loss_ga2 = np.mean(lst_ga2['losses'][-50:])
base_loss_ga2 = np.mean(base_ga2['baseline_losses'][-50:])
speedup_ga2 = base_ga2['baseline_total_time'] / lst_ga2['total_time']
degrad_ga2 = (lst_loss_ga2 - base_loss_ga2) / base_loss_ga2 * 100
print(f"\n>>> RESULT: GA=2 Speedup={speedup_ga2:.2f}x, Degradation={degrad_ga2:.1f}%")

# === CELL 7: Generate All Plots ===
from experiments.plot_ablations import plot_all
plot_all()

# === CELL 8: Display Results Table ===
from experiments.plot_ablations import print_results_table
print_results_table()

# === CELL 9: Show Key Plots Inline ===
from IPython.display import Image, display
import os

for fig_name in [
    'ablation_loss_vs_walltime.png',
    'ablation_ga.pdf',
    'ablation_acceptance.pdf',
    'ablation_hybrid.pdf',
]:
    path = os.path.join('paper', 'figures', fig_name)
    if os.path.exists(path):
        print(f"\n{'='*50}")
        print(f"  {fig_name}")
        print(f"{'='*50}")
        if path.endswith('.png'):
            display(Image(filename=path))
        else:
            print(f"  (PDF saved to {path})")

# === CELL 10 (OPTIONAL): Run Experiment 4 — Long Training 10K Steps ===
# WARNING: This takes ~6 hours. Only run if you have the GPU time.
# from experiments.run_ablations import run_experiment
# lst_10k, base_10k = run_experiment('long_10k')
# from experiments.plot_ablations import plot_long_training
# plot_long_training()

# === CELL 11: Quick LaTeX Table for Paper ===
import pickle, numpy as np
print("\n% === LaTeX table for paper (copy-paste) ===")
print("\\begin{tabular}{@{}lcccc@{}}")
print("\\toprule")
print("\\textbf{Config} & \\textbf{Speedup} & \\textbf{Accept\\%} & \\textbf{Degrad\\%} & \\textbf{Wall Time} \\\\")
print("\\midrule")

for name, label in [
    ('quality_focused', 'Quality ($K{=}5$)'),
    ('hybrid', 'Hybrid'),
    ('ga1', 'GA=1'),
    ('ga2', 'GA=2'),
]:
    lst_path = f'checkpoints/lst_{name}.pkl'
    base_path = f'checkpoints/baseline_{name}.pkl'
    if not os.path.exists(lst_path):
        continue
    with open(lst_path, 'rb') as f:
        lst = pickle.load(f)
    lst_loss = np.mean(lst['losses'][-50:])
    accept = lst['acceptance_rates'][-1] * 100 if lst.get('acceptance_rates') else 0

    if os.path.exists(base_path):
        with open(base_path, 'rb') as f:
            base = pickle.load(f)
        base_loss = np.mean(base['baseline_losses'][-50:])
        base_time = base['baseline_total_time']
        speedup = base_time / lst['total_time']
        degrad = (lst_loss - base_loss) / base_loss * 100
        print(f"{label} & {speedup:.2f}$\\times$ & {accept:.1f}\\% & {degrad:+.1f}\\% & {lst['total_time']:.0f}s \\\\")

print("\\bottomrule")
print("\\end{tabular}")

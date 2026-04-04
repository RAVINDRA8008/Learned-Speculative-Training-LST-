"""
LST Comprehensive Experiment Suite
===================================
Run ALL ablations needed for SOTA-level paper:

  Exp 1: Quality-focused LST (K=5, tol_floor=0.02) — target <5% loss degradation
  Exp 2: Hybrid LST (80% speculative → 20% standard) — target ~0% degradation
  Exp 3: Gradient Accumulation ablation (GA=1, GA=2, GA=4)
  Exp 4: Long training (10K steps) — acceptance trajectory stability
  Exp 5: Loss-vs-walltime curve (from all runs)

Usage on Colab:
    !pip install transformers datasets
    %cd /content
    !git clone https://github.com/RAVINDRA8008/Learned-Speculative-Training-LST-.git LST
    %cd LST
    !pip install -e .

    # Then run individual experiments:
    from experiments.run_ablations import run_experiment, CONFIGS
    results = run_experiment('quality_focused')   # or 'hybrid', 'ga1', 'ga2', 'long_10k'

    # Or run all:
    from experiments.run_ablations import run_all
    run_all()

Each experiment saves a checkpoint .pkl file. Plotting scripts load from these.
Estimated GPU time (A100 40GB):
  Exp 1 (quality_focused):   ~25 min LST + ~50 min baseline = ~75 min
  Exp 2 (hybrid):            ~30 min (no separate baseline — uses Exp 1 baseline)
  Exp 3a (ga1):              ~12 min LST + ~12 min baseline = ~24 min
  Exp 3b (ga2):              ~18 min LST + ~24 min baseline = ~42 min
  Exp 4 (long_10k):          ~125 min LST + ~250 min baseline = ~375 min
  Total:                     ~9 hours (run Exp 1-3 first, Exp 4 if time permits)
"""

import os
import sys
import gc
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, IterableDataset
from dataclasses import dataclass, field
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ====================================================================
#  CONFIG DATACLASS
# ====================================================================
@dataclass
class ExperimentConfig:
    """Configuration for one experiment run."""
    name: str = "default"

    # Model
    model_name: str = "gpt2"
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12

    # Data
    max_seq_len: int = 1024
    batch_size: int = 16  # micro-batch
    grad_accum_steps: int = 4

    # Training
    total_steps: int = 2000
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    use_amp: bool = True
    seed: int = 42

    # LST
    lst_enabled: bool = True
    lst_warmup_steps: int = 50
    lst_K: int = 20
    lst_tolerance: float = 0.015
    lst_rank: int = 8
    lst_proj_dim: int = 32
    lst_grad_history: int = 4
    lst_draft_d_model: int = 256
    lst_draft_n_heads: int = 4
    lst_draft_n_blocks: int = 2
    lst_draft_lr: float = 3e-4
    lst_adaptive_tol: bool = True
    lst_draft_layer_fraction: float = 0.25
    lst_draft_max_elements: int = 4096
    lst_draft_train_every: int = 2
    lst_tol_min: float = 0.005
    lst_tol_max: float = 0.05
    lst_hybrid_switch_step: Optional[int] = None

    # Logging
    log_interval: int = 50


# ====================================================================
#  EXPERIMENT CONFIGS
# ====================================================================

# Original run (for reference / re-run)
ORIGINAL = ExperimentConfig(
    name="original",
    lst_K=20,
    lst_tolerance=0.015,
    lst_tol_min=0.005,
    lst_tol_max=0.05,
)

# Exp 1: Quality-focused — tighter supervision, higher tolerance floor
QUALITY_FOCUSED = ExperimentConfig(
    name="quality_focused",
    lst_K=5,              # 4x more supervision (every 5 steps instead of 20)
    lst_tolerance=0.02,   # start slightly higher
    lst_tol_min=0.02,     # FLOOR at 0.02 — never accept >2% worse than baseline
    lst_tol_max=0.05,
    lst_draft_train_every=1,  # Train draft every supervision (more data at K=5)
)

# Exp 2: Hybrid — LST for first 80%, standard for last 20%
HYBRID = ExperimentConfig(
    name="hybrid",
    lst_K=20,
    lst_tolerance=0.015,
    lst_tol_min=0.005,
    lst_tol_max=0.05,
    lst_hybrid_switch_step=1600,  # Switch to standard at step 1600 (80% of 2000)
)

# Exp 3a: GA=1 (no accumulation, micro-batch=16, effective batch=16)
GA1 = ExperimentConfig(
    name="ga1",
    grad_accum_steps=1,
    batch_size=16,  # effective batch = 16
)

# Exp 3b: GA=2 (micro-batch=16, effective batch=32)
GA2 = ExperimentConfig(
    name="ga2",
    grad_accum_steps=2,
    batch_size=16,  # effective batch = 32
)

# Exp 4: Long training (10K steps)
LONG_10K = ExperimentConfig(
    name="long_10k",
    total_steps=10000,
    lst_warmup_steps=200,  # longer warmup proportional to total
    log_interval=100,
)

# All configs
CONFIGS = {
    'original': ORIGINAL,
    'quality_focused': QUALITY_FOCUSED,
    'hybrid': HYBRID,
    'ga1': GA1,
    'ga2': GA2,
    'long_10k': LONG_10K,
}

BASELINE_CONFIGS = {
    'original': ORIGINAL,
    'quality_focused': QUALITY_FOCUSED,  # same baseline needed
    'ga1': GA1,
    'ga2': GA2,
    'long_10k': LONG_10K,
}


# ====================================================================
#  DATA PIPELINE (same as notebook)
# ====================================================================
class StreamingTextDataset(IterableDataset):
    """Streams tokenized text chunks. Cycles infinitely."""

    def __init__(self, tokenizer, seq_len=1024, split='train'):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dataset = None
        self.text_key = 'text'

        for name, config_name in [
            ('wikitext-103', 'wikitext-103-raw-v1'),
            ('wikitext-2', 'wikitext-2-raw-v1'),
        ]:
            try:
                from datasets import load_dataset
                self.dataset = load_dataset(
                    'wikitext', config_name,
                    split=split,
                )
                print(f"Loaded {name} ({split}, {len(self.dataset)} examples)")
                break
            except Exception as e:
                print(f"{name} failed: {e}")

        if self.dataset is None:
            print("Generating synthetic data (testing only)...")
            self.dataset = [
                {'text': f"The quick brown fox jumps over the lazy dog. Sentence {i}. " * 10}
                for i in range(10000)
            ]

    def __iter__(self):
        while True:
            buffer = []
            for example in self.dataset:
                text = example[self.text_key]
                if not text or len(text.strip()) < 10:
                    continue
                tokens = self.tokenizer.encode(text)
                buffer.extend(tokens)
                while len(buffer) >= self.seq_len + 1:
                    chunk = buffer[:self.seq_len + 1]
                    buffer = buffer[self.seq_len:]
                    input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                    labels = torch.tensor(chunk[1:], dtype=torch.long)
                    attention_mask = torch.ones_like(input_ids)
                    yield {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'labels': labels,
                    }


# ====================================================================
#  MODEL / OPTIMIZER / SCHEDULER SETUP
# ====================================================================
def setup_model(config: ExperimentConfig, device='cuda'):
    """Create fresh GPT-2 model + optimizer + scheduler from config."""
    from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    torch.manual_seed(config.seed)
    model_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=config.max_seq_len,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(model_config).to(device)

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr,
        weight_decay=config.weight_decay, betas=(0.9, 0.999),
    )

    warmup_steps = int(config.total_steps * config.warmup_ratio)
    warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=config.total_steps - warmup_steps, eta_min=config.lr * 0.1)
    lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])

    return model, optimizer, lr_scheduler, tokenizer, model_config


# ====================================================================
#  TRAINING LOOPS
# ====================================================================
def run_lst_training(config: ExperimentConfig, model, optimizer, lr_scheduler, tokenizer, device='cuda'):
    """Run LST training and return results dict."""
    from lst.trainer import LSTTrainer

    dataset = StreamingTextDataset(tokenizer, seq_len=config.max_seq_len, split='train')
    loader = DataLoader(dataset, batch_size=config.batch_size)

    lst_trainer = LSTTrainer(
        model=model,
        optimizer=optimizer,
        K=config.lst_K,
        tolerance=config.lst_tolerance,
        warmup_steps=config.lst_warmup_steps,
        grad_history_len=config.lst_grad_history,
        proj_dim=config.lst_proj_dim,
        rank=config.lst_rank,
        total_steps=config.total_steps,
        draft_lr=config.lst_draft_lr,
        d_model=config.lst_draft_d_model,
        n_heads=config.lst_draft_n_heads,
        n_blocks=config.lst_draft_n_blocks,
        adaptive_tolerance=config.lst_adaptive_tol,
        max_grad_norm=config.max_grad_norm,
        use_amp=config.use_amp,
        draft_layer_fraction=config.lst_draft_layer_fraction,
        draft_max_elements=config.lst_draft_max_elements,
        draft_train_every=config.lst_draft_train_every,
        tol_min=config.lst_tol_min,
        tol_max=config.lst_tol_max,
        hybrid_switch_step=config.lst_hybrid_switch_step,
    )

    losses, step_times, acceptance_rates = [], [], []
    accepted_flags, draft_losses, tolerances = [], [], []
    steps_done = 0

    model.train()
    micro_batches = []
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"LST Training: {config.name} | K={config.lst_K} | tol_min={config.lst_tol_min} | GA={config.grad_accum_steps}")
    if config.lst_hybrid_switch_step:
        print(f"  Hybrid switch at step {config.lst_hybrid_switch_step}")
    print(f"{'='*60}")

    for batch in loader:
        if steps_done >= config.total_steps:
            break

        micro_batches.append(batch)
        if len(micro_batches) < config.grad_accum_steps:
            continue

        step_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        result = lst_trainer.step_batch(micro_batches, lr=current_lr)
        micro_batches = []

        if result.get('accepted') is not True:
            lr_scheduler.step()

        step_time = time.time() - step_start
        steps_done += 1

        losses.append(result['loss'])
        step_times.append(step_time)

        if result.get('accepted') is not None:
            accepted_flags.append(1 if result['accepted'] else 0)
        if result.get('acceptance_rate') is not None:
            acceptance_rates.append(result['acceptance_rate'])
        if result.get('draft_loss') is not None:
            draft_losses.append(result['draft_loss'])
        if result.get('tolerance') is not None:
            tolerances.append(result['tolerance'])

        if steps_done % config.log_interval == 0:
            avg_loss = np.mean(losses[-config.log_interval:])
            avg_time = np.mean(step_times[-config.log_interval:])
            phase = result.get('phase', '?')
            msg = f"  Step {steps_done}/{config.total_steps} | Loss {avg_loss:.4f} | {avg_time*1000:.0f}ms/step | {phase}"
            if acceptance_rates:
                msg += f" | Accept {acceptance_rates[-1]:.1%}"
            print(msg)

    total_time = time.time() - start_time

    results = {
        'name': config.name,
        'type': 'lst',
        'config': config.__dict__,
        'total_time': total_time,
        'losses': losses,
        'step_times': step_times,
        'acceptance_rates': acceptance_rates,
        'accepted_flags': accepted_flags,
        'draft_losses': draft_losses,
        'tolerances': tolerances,
        'steps_done': steps_done,
        'lst_total_accepted': lst_trainer.verifier.total_accepted,
        'lst_total_speculative': lst_trainer.verifier.total_speculative,
        'lst_final_tolerance': lst_trainer.verifier.tolerance,
        'lst_draft_params': lst_trainer.draft.count_parameters(),
    }

    print(f"\n  Done: {steps_done} steps, {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Final loss: {np.mean(losses[-50:]):.4f}")
    print(f"  Avg ms/step: {np.mean(step_times)*1000:.1f}")
    if acceptance_rates:
        print(f"  Acceptance: {acceptance_rates[-1]:.1%} ({results['lst_total_accepted']}/{results['lst_total_speculative']})")

    return results


def run_baseline_training(config: ExperimentConfig, model_config, tokenizer, device='cuda'):
    """Run standard baseline training and return results dict."""
    from transformers import GPT2LMHeadModel

    torch.manual_seed(config.seed)
    model = GPT2LMHeadModel(model_config).to(device)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr,
        weight_decay=config.weight_decay, betas=(0.9, 0.999),
    )
    warmup_steps = int(config.total_steps * config.warmup_ratio)
    warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=config.total_steps - warmup_steps, eta_min=config.lr * 0.1)
    lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])

    dataset = StreamingTextDataset(tokenizer, seq_len=config.max_seq_len, split='train')
    loader = DataLoader(dataset, batch_size=config.batch_size)

    losses, step_times = [], []
    steps_done = 0

    model.train()
    micro_batches = []
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"Baseline Training: {config.name} | GA={config.grad_accum_steps}")
    print(f"{'='*60}")

    for batch in loader:
        if steps_done >= config.total_steps:
            break

        micro_batches.append(batch)
        if len(micro_batches) < config.grad_accum_steps:
            continue

        step_start = time.time()
        batches_gpu = [{k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)} for b in micro_batches]
        optimizer.zero_grad()
        n_micro = len(batches_gpu)
        total_loss = 0.0
        for mb in batches_gpu:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=config.use_amp):
                output = model(**mb)
                scaled_loss = output.loss / n_micro
            scaled_loss.backward()
            total_loss += output.loss.item()
        avg_loss = total_loss / n_micro

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        micro_batches = []

        step_time = time.time() - step_start
        steps_done += 1
        losses.append(avg_loss)
        step_times.append(step_time)

        if steps_done % config.log_interval == 0:
            avg_l = np.mean(losses[-config.log_interval:])
            avg_t = np.mean(step_times[-config.log_interval:])
            print(f"  Baseline Step {steps_done}/{config.total_steps} | Loss {avg_l:.4f} | {avg_t*1000:.0f}ms/step")

    total_time = time.time() - start_time

    results = {
        'name': config.name,
        'type': 'baseline',
        'config': config.__dict__,
        'baseline_total_time': total_time,
        'baseline_losses': losses,
        'baseline_times': step_times,
        'baseline_steps': steps_done,
    }

    print(f"\n  Done: {steps_done} steps, {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Final loss: {np.mean(losses[-50:]):.4f}")
    print(f"  Avg ms/step: {np.mean(step_times)*1000:.1f}")

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ====================================================================
#  EXPERIMENT RUNNER
# ====================================================================
def save_checkpoint(results, filename):
    """Save results to pickle."""
    os.makedirs('checkpoints', exist_ok=True)
    path = os.path.join('checkpoints', filename)
    with open(path, 'wb') as f:
        pickle.dump(results, f)
    print(f"  Saved: {path}")
    return path


def run_experiment(name: str, skip_baseline: bool = False, device: str = 'cuda'):
    """
    Run a single named experiment (LST + baseline).

    Args:
        name: One of 'quality_focused', 'hybrid', 'ga1', 'ga2', 'long_10k', 'original'
        skip_baseline: If True, only run LST (e.g., hybrid reuses existing baseline)
        device: 'cuda' or 'cpu'

    Returns:
        (lst_results, baseline_results) tuple
    """
    config = CONFIGS[name]
    print(f"\n{'#'*70}")
    print(f"# EXPERIMENT: {name}")
    print(f"# K={config.lst_K}, tol_min={config.lst_tol_min}, GA={config.grad_accum_steps}, steps={config.total_steps}")
    if config.lst_hybrid_switch_step:
        print(f"# Hybrid switch at step {config.lst_hybrid_switch_step}")
    print(f"{'#'*70}")

    # --- LST Run ---
    model, optimizer, lr_scheduler, tokenizer, model_config = setup_model(config, device)
    lst_results = run_lst_training(config, model, optimizer, lr_scheduler, tokenizer, device)
    save_checkpoint(lst_results, f'lst_{name}.pkl')

    # Free memory
    del model, optimizer, lr_scheduler
    gc.collect()
    torch.cuda.empty_cache()

    # --- Baseline Run ---
    baseline_results = None
    if not skip_baseline:
        baseline_results = run_baseline_training(config, model_config, tokenizer, device)
        save_checkpoint(baseline_results, f'baseline_{name}.pkl')

    return lst_results, baseline_results


def run_priority_experiments(device='cuda'):
    """
    Run experiments in priority order (most impactful first).
    Each experiment saves independently — safe to interrupt.
    """
    print("=" * 70)
    print("LST COMPREHENSIVE ABLATION SUITE")
    print("=" * 70)
    all_results = {}

    # Priority 1: Quality-focused (K=5, tol_floor=0.02)
    # This is the most important — if it works, it fixes the paper's biggest weakness
    print("\n\n>>> Priority 1: Quality-focused LST (K=5, tol_floor=0.02)")
    lst_r, base_r = run_experiment('quality_focused', device=device)
    all_results['quality_focused'] = {'lst': lst_r, 'baseline': base_r}

    # Priority 2: Hybrid schedule (LST 80% → standard 20%)
    # Reuses the quality_focused baseline (same GA=4 config)
    print("\n\n>>> Priority 2: Hybrid LST (switch at step 1600)")
    lst_r, _ = run_experiment('hybrid', skip_baseline=True, device=device)
    all_results['hybrid'] = {'lst': lst_r, 'baseline': all_results['quality_focused']['baseline']}

    # Priority 3: GA ablation (GA=1 and GA=2 — we already have GA=4 from quality_focused)
    print("\n\n>>> Priority 3a: GA=1 ablation")
    lst_r, base_r = run_experiment('ga1', device=device)
    all_results['ga1'] = {'lst': lst_r, 'baseline': base_r}

    print("\n\n>>> Priority 3b: GA=2 ablation")
    lst_r, base_r = run_experiment('ga2', device=device)
    all_results['ga2'] = {'lst': lst_r, 'baseline': base_r}

    # Priority 4: Long training (10K steps) — only if time permits
    print("\n\n>>> Priority 4: Long training (10K steps)")
    print("    WARNING: This takes ~6 hours. Ctrl+C to skip.")
    try:
        lst_r, base_r = run_experiment('long_10k', device=device)
        all_results['long_10k'] = {'lst': lst_r, 'baseline': base_r}
    except KeyboardInterrupt:
        print("    Skipped long_10k (interrupted)")

    # Save combined results
    save_checkpoint(all_results, 'all_experiments.pkl')
    print("\n\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)

    return all_results


def run_all(device='cuda'):
    """Alias for run_priority_experiments."""
    return run_priority_experiments(device)


# ====================================================================
#  QUICK SUMMARY
# ====================================================================
def print_summary():
    """Load all checkpoints and print a comparison table."""
    results = {}
    for name in CONFIGS:
        lst_path = f'checkpoints/lst_{name}.pkl'
        base_path = f'checkpoints/baseline_{name}.pkl'
        if os.path.exists(lst_path):
            with open(lst_path, 'rb') as f:
                results[f'lst_{name}'] = pickle.load(f)
        if os.path.exists(base_path):
            with open(base_path, 'rb') as f:
                results[f'baseline_{name}'] = pickle.load(f)

    print(f"\n{'='*90}")
    print(f"{'Experiment':<20} {'Wall Time':>10} {'Loss':>8} {'Accept%':>9} {'Speedup':>9} {'Degrad%':>9}")
    print(f"{'='*90}")

    for name in CONFIGS:
        lst_key = f'lst_{name}'
        base_key = f'baseline_{name}'
        if lst_key not in results:
            continue

        lst = results[lst_key]
        lst_time = lst['total_time']
        lst_loss = np.mean(lst['losses'][-50:])
        accept = lst['acceptance_rates'][-1] if lst['acceptance_rates'] else 0

        base_time = '--'
        speedup = '--'
        degrad = '--'

        if base_key in results:
            base = results[base_key]
            bt = base['baseline_total_time']
            bl = np.mean(base['baseline_losses'][-50:])
            base_time = f'{bt:.0f}s'
            speedup = f'{bt/lst_time:.2f}x'
            degrad = f'{(lst_loss - bl) / bl * 100:.1f}%'

        print(f"  LST {name:<16} {lst_time:>8.0f}s {lst_loss:>8.2f} {accept:>8.1%} {speedup:>9} {degrad:>9}")

    print(f"{'='*90}")


# ====================================================================
#  MAIN
# ====================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='LST Ablation Experiments')
    parser.add_argument('--experiment', '-e', type=str, default='all',
                        choices=['all', 'quality_focused', 'hybrid', 'ga1', 'ga2', 'long_10k', 'original', 'summary'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip-baseline', action='store_true')
    args = parser.parse_args()

    if args.experiment == 'summary':
        print_summary()
    elif args.experiment == 'all':
        run_all(args.device)
    else:
        run_experiment(args.experiment, skip_baseline=args.skip_baseline, device=args.device)
        print_summary()

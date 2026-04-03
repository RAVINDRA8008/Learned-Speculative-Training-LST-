# Learned Speculative Training (LST)

**Extending Speculative Weight Prediction into Chaotic Training Regimes via Online-Trained Draft Networks**

## Overview

Modern LLM training spends ~66% of each step's FLOPs on the backward pass. LST replaces analytic weight extrapolation with a small learned draft network (~3M parameters) trained online via K-step self-supervision on the target model's own gradient distribution.

Key results (targeted):
- **40-50% wall-clock reduction** in LLM pretraining
- **<1% loss degradation** vs standard training
- **~40-50% acceptance** in chaotic regimes (vs Leap+Verify's ~9%)
- **~85% acceptance** in stable regimes

## Quick Start

```bash
pip install -e .
```

### Run on Colab
Open `notebooks/lst_train_colab.ipynb` in Google Colab with a T4/A100 GPU.

### Run locally
```python
from lst.trainer import LSTTrainer

trainer = LSTTrainer(model, optimizer)
for batch in dataloader:
    result = trainer.step_batch(batch, lr=current_lr)
    print(f"Loss: {result['loss']:.4f}, Accepted: {result.get('accepted', 'N/A')}")
```

## Project Structure

```
lst/
├── __init__.py
├── draft_model.py          # GradientTransformer (3M param draft network)
├── feature_extraction.py   # Weight stats + gradient history compression
├── trainer.py              # Core LST training loop
├── verification.py         # Accept/reject logic + adaptive tolerance
└── utils.py                # Snapshot, restore, metrics tracking
notebooks/
└── lst_train_colab.ipynb   # Main Colab training notebook
```

## Method

1. **Draft Prediction** — A small transformer predicts weight updates from gradient history features
2. **Speculative Application** — Apply predicted update, snapshot weights for rollback
3. **Verification** — Forward pass only (33% cost) to check loss
4. **Accept/Reject** — Keep update (skip backward) or revert + real backward (train draft on error)

## Citation

```
@article{ravindra2026lst,
  title={Learned Speculative Training: Extending Weight Prediction into Chaotic Training Regimes},
  author={Ravindra},
  year={2026}
}
```

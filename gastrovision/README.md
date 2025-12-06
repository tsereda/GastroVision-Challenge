# GastroVision Challenge - Swin Transformer

Hyperparameter optimization for 4-class endoscopic image classification.

## Setup

```bash
pip install -r requirements.txt
```

## Data Preparation

Edit `train.py` and implement the `load_data()` function with your dataset paths.

## Run Sweep

```bash
# Initialize sweep
wandb sweep config/sweep.yaml

# Run agent (replace with your sweep ID)
wandb agent <your-entity>/<your-project>/<sweep-id> --count 50
```

## Key Features

- **Swin Transformer** baseline (proven for medical imaging)
- **3 loss functions**: Focal, Weighted CE, Focal+Weighted
- **Image size sweep**: 224 vs 384
- **Strong augmentations**: Color jitter, mixup, cutmix
- **Class imbalance handling**: Automatic class weighting

## Metrics Tracked

- Balanced accuracy (primary)
- F1-macro
- Per-class recall (especially erythema/polyps)
- Worst-class recall

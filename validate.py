"""
Validation script - evaluate model on validation set before submission
"""

import argparse
import torch
import wandb
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    balanced_accuracy_score, 
    f1_score, 
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

from src import GastroVisionDataset, SwinClassifier, get_val_transforms


def load_model_from_wandb(run_path, device='cuda'):
    """Download and load model from W&B"""
    print(f"🔄 Downloading model from W&B run: {run_path}")
    
    api = wandb.Api()
    run = api.run(run_path)
    config = run.config
    
    # Download checkpoint
    checkpoint_file = None
    for file in run.files():
        if file.name == 'best_model.pth':
            checkpoint_file = file
            break
    
    if checkpoint_file is None:
        raise ValueError(f"No best_model.pth found in run {run_path}")
    
    checkpoint_path = checkpoint_file.download(replace=True, root='./wandb_downloads')
    
    # Create and load model
    model = SwinClassifier(
        model_name=config.get('model_name', 'swin_base_patch4_window12_384'),
        num_classes=4,
        pretrained=False,
        dropout=config.get('dropout', 0.3),
        stochastic_depth=config.get('stochastic_depth', 0.2)
    )
    
    checkpoint = torch.load(checkpoint_path.name, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config


def load_validation_data():
    """Load validation split (same as training)"""
    data_root = Path('Gastrovision Challenge dataset/Training data')
    
    class_mapping = {
        'Normal mucosa and vascular pattern in the large bowel': 0,
        'Normal esophagus': 1,
        'Colon polyps': 2,
        'Erythema': 3
    }
    
    all_paths = []
    all_labels = []
    
    for class_name, label in class_mapping.items():
        class_dir = data_root / class_name
        image_paths = list(class_dir.glob('*.jpg'))
        all_paths.extend([str(p) for p in image_paths])
        all_labels.extend([label] * len(image_paths))
    
    # Same split as training
    _, val_paths, _, val_labels = train_test_split(
        all_paths, all_labels,
        test_size=0.2,
        stratify=all_labels,
        random_state=42
    )
    
    return val_paths, val_labels


def evaluate(model, val_loader, device='cuda'):
    """Run evaluation"""
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
    
    return np.array(all_preds), np.array(all_targets)


def print_metrics(preds, targets):
    """Print detailed metrics"""
    class_names = [
        'Normal_mucosa_large_bowel',
        'Normal_esophagus',
        'colon polyp',
        'Erythema'
    ]
    
    bal_acc = balanced_accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro')
    
    print(f"\n{'='*60}")
    print(f"📊 VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Macro F1 Score:    {f1:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(targets, preds, target_names=class_names, digits=4))
    
    print(f"\n🔢 Confusion Matrix:")
    cm = confusion_matrix(targets, preds)
    print("\nPredicted →")
    print(f"{'True ↓':<25}", end='')
    for name in class_names:
        print(f"{name[:10]:>12}", end='')
    print()
    
    for i, name in enumerate(class_names):
        print(f"{name:<25}", end='')
        for j in range(len(class_names)):
            print(f"{cm[i,j]:>12}", end='')
        print()
    
    # Per-class accuracy
    print(f"\n📈 Per-Class Recall:")
    for i, name in enumerate(class_names):
        recall = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  {name:<30} {recall:.4f} ({cm[i, i]}/{cm[i].sum()})")
    
    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Validate model on validation set')
    parser.add_argument(
        '--run',
        type=str,
        required=True,
        help='W&B run path or run ID'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device'
    )
    
    args = parser.parse_args()
    
    # Format run path
    if '/' not in args.run:
        run_path = f"timgsereda/gastrovision-challenge/{args.run}"
    else:
        run_path = args.run
    
    print("🔬 GastroVision Challenge - Validation")
    
    # Load model
    model, config = load_model_from_wandb(run_path, device=args.device)
    
    # Load validation data
    print("\n📁 Loading validation data...")
    val_paths, val_labels = load_validation_data()
    print(f"   {len(val_paths)} validation images")
    
    # Create dataset
    img_size = config.get('img_size', 384)
    transforms = get_val_transforms(img_size=img_size)
    val_dataset = GastroVisionDataset(val_paths, val_labels, transform=transforms)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate
    print("\n🔮 Running evaluation...")
    preds, targets = evaluate(model, val_loader, device=args.device)
    
    # Print results
    print_metrics(preds, targets)


if __name__ == '__main__':
    main()

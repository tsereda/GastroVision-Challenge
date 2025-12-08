#!/usr/bin/env python3
"""
Enhanced Inference Pipeline with Ensemble + TTA
Supports:
1. Single model inference
2. Ensemble of top-K models
3. Test-Time Augmentation (TTA)
4. Ensemble + TTA combined
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import wandb

from src import (
    SwinClassifier,
    get_val_transforms,
    ModelEnsemble,
    create_ensemble_from_wandb,
    get_top_runs_from_sweep,
    TTAWrapper,
    MultiScaleTTA,
    CombinedTTA
)
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class TestDataset(Dataset):
    """Simple test dataset"""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, Path(img_path).name


def predict_single_model(checkpoint_path, test_dir, output_csv='predictions.csv', 
                         batch_size=32, use_tta=False, tta_mode='standard'):
    """Predict with a single model (optionally with TTA)
    
    Args:
        checkpoint_path: Path to model checkpoint
        test_dir: Directory containing test images
        output_csv: Output CSV filename
        batch_size: Batch size for inference
        use_tta: Whether to use Test-Time Augmentation
        tta_mode: TTA mode ('standard', 'multiscale', 'combined')
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*80)
    print(f"SINGLE MODEL INFERENCE {'WITH TTA' if use_tta else ''}")
    print("="*80)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Create model
    model = SwinClassifier(
        model_name=config.get('model_name', 'swin_base_patch4_window12_384'),
        num_classes=4,
        pretrained=False,
        dropout=config.get('dropout_rate', 0.3),
        stochastic_depth=config.get('stochastic_depth', 0.2)
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Wrap with TTA if requested
    if use_tta:
        if tta_mode == 'standard':
            model = TTAWrapper(model, device=device)
            print("Using standard TTA (8 augmentations)")
        elif tta_mode == 'multiscale':
            model = MultiScaleTTA(model, device=device)
            print("Using multi-scale TTA")
        elif tta_mode == 'combined':
            model = CombinedTTA(model, device=device)
            print("Using combined TTA (geometric + multi-scale)")
    
    # Prepare test data
    test_path = Path(test_dir)
    test_images = sorted(list(test_path.glob('*.jpg')))
    
    print(f"Found {len(test_images)} test images")
    
    # Create dataset and loader
    img_size = config.get('image_size', 384)
    transform = get_val_transforms(img_size)
    
    test_dataset = TestDataset(test_images, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Run inference
    all_predictions = []
    all_image_names = []
    
    print("\nRunning inference...")
    with torch.no_grad():
        for images, image_names in tqdm(test_loader):
            images = images.to(device)
            
            if use_tta:
                preds, _ = model.predict(images)
            else:
                logits = model(images)
                preds = torch.argmax(logits, dim=1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_image_names.extend(image_names)
    
    # Convert to class labels
    label_mapping = {
        0: 'Normal mucosa and vascular pattern in the large bowel',
        1: 'Normal esophagus',
        2: 'Colon polyps',
        3: 'Erythema'
    }
    
    predictions_text = [label_mapping[p] for p in all_predictions]
    
    # Create DataFrame
    df = pd.DataFrame({
        'image_id': all_image_names,
        'prediction': predictions_text
    })
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Predictions saved to: {output_csv}")
    print(f"  - {len(df)} predictions")
    
    # Print distribution
    print("\nPrediction distribution:")
    for label, count in df['prediction'].value_counts().items():
        print(f"  {label}: {count}")
    
    return df


def predict_ensemble(sweep_id, entity, project, test_dir, output_csv='predictions_ensemble.csv',
                     top_k=5, batch_size=32, use_tta=False, weights=None):
    """Predict with ensemble of top-K models from sweep
    
    Args:
        sweep_id: W&B sweep ID
        entity: W&B entity
        project: W&B project
        test_dir: Directory containing test images
        output_csv: Output CSV filename
        top_k: Number of top models to ensemble
        batch_size: Batch size for inference
        use_tta: Whether to use TTA on ensemble
        weights: Optional custom weights for models
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*80)
    print(f"ENSEMBLE INFERENCE (Top-{top_k} models)" + (" WITH TTA" if use_tta else ""))
    print("="*80)
    
    # Get top runs from sweep
    top_run_ids = get_top_runs_from_sweep(
        sweep_id=sweep_id,
        entity=entity,
        project=project,
        top_k=top_k,
        metric='val/balanced_accuracy'
    )
    
    # Create ensemble
    ensemble = create_ensemble_from_wandb(
        run_ids=top_run_ids,
        entity=entity,
        project=project,
        weights=weights,
        device=device
    )
    
    # Wrap with TTA if requested
    if use_tta:
        print("\nWrapping ensemble with TTA...")
        # Apply TTA to each model in ensemble
        for i, model in enumerate(ensemble.models):
            ensemble.models[i] = TTAWrapper(model, device=device)
    
    # Prepare test data
    test_path = Path(test_dir)
    test_images = sorted(list(test_path.glob('*.jpg')))
    
    print(f"\nFound {len(test_images)} test images")
    
    # Create dataset and loader
    transform = get_val_transforms(384)  # Use standard size
    
    test_dataset = TestDataset(test_images, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Run inference
    all_predictions = []
    all_image_names = []
    
    print("\nRunning ensemble inference...")
    with torch.no_grad():
        for images, image_names in tqdm(test_loader):
            if use_tta:
                # TTA already applied to individual models
                preds, _ = ensemble.predict(images)
            else:
                preds, _ = ensemble.predict(images)
            
            all_predictions.extend(preds.cpu().numpy())
            all_image_names.extend(image_names)
    
    # Convert to class labels
    label_mapping = {
        0: 'Normal mucosa and vascular pattern in the large bowel',
        1: 'Normal esophagus',
        2: 'Colon polyps',
        3: 'Erythema'
    }
    
    predictions_text = [label_mapping[p] for p in all_predictions]
    
    # Create DataFrame
    df = pd.DataFrame({
        'image_id': all_image_names,
        'prediction': predictions_text
    })
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Ensemble predictions saved to: {output_csv}")
    print(f"  - {len(df)} predictions")
    print(f"  - {top_k} models ensembled")
    
    # Print distribution
    print("\nPrediction distribution:")
    for label, count in df['prediction'].value_counts().items():
        print(f"  {label}: {count}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Enhanced GastroVision Inference')
    parser.add_argument('--mode', choices=['single', 'ensemble'], default='single',
                       help='Inference mode')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint (for single mode)')
    parser.add_argument('--sweep-id', type=str, help='W&B sweep ID (for ensemble mode)')
    parser.add_argument('--entity', type=str, default='timgsereda', help='W&B entity')
    parser.add_argument('--project', type=str, default='gastrovision-challenge', help='W&B project')
    parser.add_argument('--test-dir', type=str, 
                       default='Gastrovision Challenge dataset/Test data',
                       help='Test data directory')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output CSV filename')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top models for ensemble')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--use-tta', action='store_true',
                       help='Use Test-Time Augmentation')
    parser.add_argument('--tta-mode', choices=['standard', 'multiscale', 'combined'],
                       default='combined', help='TTA mode')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.checkpoint:
            raise ValueError("--checkpoint required for single mode")
        
        predict_single_model(
            checkpoint_path=args.checkpoint,
            test_dir=args.test_dir,
            output_csv=args.output,
            batch_size=args.batch_size,
            use_tta=args.use_tta,
            tta_mode=args.tta_mode
        )
    
    elif args.mode == 'ensemble':
        if not args.sweep_id:
            raise ValueError("--sweep-id required for ensemble mode")
        
        predict_ensemble(
            sweep_id=args.sweep_id,
            entity=args.entity,
            project=args.project,
            test_dir=args.test_dir,
            output_csv=args.output,
            top_k=args.top_k,
            batch_size=args.batch_size,
            use_tta=args.use_tta
        )


if __name__ == '__main__':
    main()

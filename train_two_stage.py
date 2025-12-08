#!/usr/bin/env python3
"""
Enhanced training script with two-stage training and hierarchical loss
Supports both single-stage and two-stage training modes
"""
from pathlib import Path
import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split

from src import (
    GastroVisionDataset,
    SwinClassifier,
    get_loss_function,
    get_train_transforms,
    get_val_transforms,
)
from src.two_stage_trainer import TwoStageTrainer


def load_data():
    """Load GastroVision dataset from directory structure"""
    data_root = Path('Gastrovision Challenge dataset/Training data')
    
    # Class mapping
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
        print(f"{class_name}: {len(image_paths)} images (label={label})")
    
    # 80/20 train/val split with stratification
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, 
        test_size=0.2, 
        stratify=all_labels, 
        random_state=42
    )
    
    print(f"\nTrain: {len(train_paths)} images")
    print(f"Val: {len(val_paths)} images")
    
    return train_paths, train_labels, val_paths, val_labels


def train_two_stage(config=None):
    """Two-stage training with W&B integration"""
    with wandb.init(config=config) as run:
        config = wandb.config
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("\n" + "="*80)
        print("TWO-STAGE TRAINING MODE")
        print("="*80)
        
        # Load data
        train_paths, train_labels, val_paths, val_labels = load_data()
        
        # Create model
        model = SwinClassifier(
            model_name=config.model_name,
            num_classes=4,
            pretrained=True,
            dropout=config.dropout_rate,
            stochastic_depth=config.stochastic_depth
        ).to(device)
        
        # Compute class weights
        class_counts = np.bincount(train_labels)
        total_samples = len(train_labels)
        class_weights = total_samples / (len(class_counts) * class_counts)
        class_weights = class_weights * config.get('class_weight_boost', 2.0)
        
        print(f"\nClass weights: {class_weights}")
        
        # Create loss function
        criterion = get_loss_function(
            loss_name=config.loss_function,
            class_weights=class_weights,
            focal_gamma=config.get('focal_gamma', 2.0),
            label_smoothing=config.get('label_smoothing', 0.1),
            lambda_worst=config.get('lambda_worst', 0.3),
            poly_epsilon=config.get('poly_epsilon', 1.0),
            hierarchical_alpha=config.get('hierarchical_alpha', 0.3),
            device=device
        )
        
        # Create dataloaders for stage 1
        train_dataset = GastroVisionDataset(
            train_paths, train_labels,
            transform=get_train_transforms(config)
        )
        val_dataset = GastroVisionDataset(
            val_paths, val_labels,
            transform=get_val_transforms(config.image_size)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize two-stage trainer
        trainer = TwoStageTrainer(model, dict(config), device)
        
        # Stage 1 optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Stage 1 scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.get('stage1_epochs', 30)
        )
        
        # Run Stage 1
        stage1_acc = trainer.train_stage1(
            train_loader, val_loader, optimizer, scheduler, criterion
        )
        
        # Run Stage 2
        stage2_acc, stage2_worst_recall = trainer.train_stage2(
            train_paths, train_labels, val_loader, criterion
        )
        
        # Final logging
        wandb.summary['final/balanced_accuracy'] = stage2_acc
        wandb.summary['final/worst_class_recall'] = stage2_worst_recall
        wandb.summary['stage1/best_accuracy'] = stage1_acc
        
        # Save final model
        checkpoint_path = f'checkpoints/two_stage_{run.id}.pth'
        Path('checkpoints').mkdir(exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': dict(config),
            'final_val_acc': stage2_acc,
            'final_worst_recall': stage2_worst_recall,
        }, checkpoint_path)
        
        wandb.save(checkpoint_path)
        
        print(f"\n✓ Training complete!")
        print(f"Final balanced accuracy: {stage2_acc:.4f}")
        print(f"Final worst class recall: {stage2_worst_recall:.4f}")
        print(f"Model saved to: {checkpoint_path}")


if __name__ == '__main__':
    train_two_stage()

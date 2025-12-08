"""
Two-Stage Training Strategy for GastroVision

Stage 1: General feature learning on all classes
Stage 2: Fine-tuning on hard/minority classes with class-specific augmentations

This addresses the Erythema recall bottleneck by:
1. Learning general endoscopic features first
2. Then focusing specifically on underperforming classes
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, recall_score
import wandb

from .dataset import GastroVisionDataset
from .augmentations import get_train_transforms, get_val_transforms, get_erythema_augmentations
from .losses import get_loss_function
from .utils import mixup_data, cutmix_data, mixup_criterion


class TwoStageTrainer:
    """Two-stage training strategy
    
    Stage 1: Train on all data with standard augmentations (warm start)
    Stage 2: Fine-tune on hard classes with class-specific augmentations
    
    Args:
        model: PyTorch model
        config: Training configuration
        device: Device to train on
    """
    
    def __init__(self, model, config, device='cuda'):
        self.model = model
        self.config = config
        self.device = device
        
        # Identify hard classes (from analysis: Erythema is bottleneck)
        self.hard_classes = config.get('hard_classes', [3])  # Erythema
        
        print("\n" + "="*60)
        print("TWO-STAGE TRAINING INITIALIZED")
        print("="*60)
        print(f"Stage 1: General training ({config.get('stage1_epochs', 30)} epochs)")
        print(f"Stage 2: Hard class focus ({config.get('stage2_epochs', 20)} epochs)")
        print(f"Hard classes: {self.hard_classes}")
        print("="*60 + "\n")
    
    def train_stage1(self, train_loader, val_loader, optimizer, scheduler, criterion):
        """Stage 1: General feature learning"""
        print("\n🚀 STAGE 1: General Feature Learning")
        print("-" * 60)
        
        stage1_epochs = self.config.get('stage1_epochs', 30)
        best_val_acc = 0.0
        
        for epoch in range(stage1_epochs):
            # Training
            train_loss, train_acc = self._train_epoch(
                train_loader, criterion, optimizer, epoch, stage=1
            )
            
            # Validation
            val_metrics = self._validate(val_loader, criterion)
            
            # Update scheduler
            if scheduler is not None:
                scheduler.step()
            
            # Logging
            if wandb.run is not None:
                wandb.log({
                    'stage': 1,
                    'stage1/epoch': epoch,
                    'stage1/train_loss': train_loss,
                    'stage1/train_acc': train_acc,
                    'stage1/val_loss': val_metrics['val/loss'],
                    'stage1/val_balanced_accuracy': val_metrics['val/balanced_accuracy'],
                    'stage1/val_worst_class_recall': val_metrics['val/worst_class_recall'],
                })
            
            # Save best model
            if val_metrics['val/balanced_accuracy'] > best_val_acc:
                best_val_acc = val_metrics['val/balanced_accuracy']
                self.stage1_best_state = self.model.state_dict().copy()
            
            print(f"Stage 1 Epoch {epoch+1}/{stage1_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Acc: {val_metrics['val/balanced_accuracy']:.4f}, "
                  f"Worst Class: {val_metrics['val/worst_class_recall']:.4f}")
        
        print(f"\n✓ Stage 1 Complete - Best Val Acc: {best_val_acc:.4f}\n")
        return best_val_acc
    
    def train_stage2(self, train_paths, train_labels, val_loader, criterion):
        """Stage 2: Fine-tune on hard classes with specialized augmentations"""
        print("\n🎯 STAGE 2: Hard Class Fine-Tuning")
        print("-" * 60)
        
        # Load best model from stage 1
        if hasattr(self, 'stage1_best_state'):
            self.model.load_state_dict(self.stage1_best_state)
        
        # Create hard-class focused dataset
        hard_class_loader = self._create_hard_class_loader(
            train_paths, train_labels
        )
        
        # Stage 2 optimizer with lower learning rate
        stage2_lr = self.config.get('learning_rate', 1e-4) * 0.1
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=stage2_lr,
            weight_decay=self.config.get('weight_decay', 0.0003)
        )
        
        # Cosine annealing for stage 2
        stage2_epochs = self.config.get('stage2_epochs', 20)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=stage2_epochs
        )
        
        best_val_acc = 0.0
        best_worst_recall = 0.0
        
        for epoch in range(stage2_epochs):
            # Training with hard class focus
            train_loss, train_acc = self._train_epoch(
                hard_class_loader, criterion, optimizer, epoch, stage=2
            )
            
            # Validation
            val_metrics = self._validate(val_loader, criterion)
            
            # Update scheduler
            scheduler.step()
            
            # Logging
            if wandb.run is not None:
                wandb.log({
                    'stage': 2,
                    'stage2/epoch': epoch,
                    'stage2/train_loss': train_loss,
                    'stage2/train_acc': train_acc,
                    'stage2/val_loss': val_metrics['val/loss'],
                    'stage2/val_balanced_accuracy': val_metrics['val/balanced_accuracy'],
                    'stage2/val_worst_class_recall': val_metrics['val/worst_class_recall'],
                    'stage2/val_erythema_recall': val_metrics['val/recall_erythema'],
                })
            
            # Save best model based on worst class recall
            if val_metrics['val/worst_class_recall'] > best_worst_recall:
                best_worst_recall = val_metrics['val/worst_class_recall']
                best_val_acc = val_metrics['val/balanced_accuracy']
                self.stage2_best_state = self.model.state_dict().copy()
            
            print(f"Stage 2 Epoch {epoch+1}/{stage2_epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Acc: {val_metrics['val/balanced_accuracy']:.4f}, "
                  f"Worst Class: {val_metrics['val/worst_class_recall']:.4f}, "
                  f"Erythema: {val_metrics['val/recall_erythema']:.4f}")
        
        # Load best stage 2 model
        if hasattr(self, 'stage2_best_state'):
            self.model.load_state_dict(self.stage2_best_state)
        
        print(f"\n✓ Stage 2 Complete - Best Worst Recall: {best_worst_recall:.4f}\n")
        return best_val_acc, best_worst_recall
    
    def _create_hard_class_loader(self, train_paths, train_labels):
        """Create dataloader focused on hard classes
        
        Uses oversampling to balance hard classes with easy classes
        """
        # Calculate sample weights
        sample_weights = np.ones(len(train_labels))
        
        for i, label in enumerate(train_labels):
            if label in self.hard_classes:
                # Oversample hard classes (3x weight)
                sample_weights[i] = 3.0
        
        # Create sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_labels),
            replacement=True
        )
        
        # Use erythema-specific augmentations for stage 2
        if 3 in self.hard_classes:  # Erythema
            train_transform = get_erythema_augmentations(
                img_size=self.config.get('image_size', 384)
            )
        else:
            train_transform = get_train_transforms(self.config)
        
        dataset = GastroVisionDataset(
            train_paths, train_labels, transform=train_transform
        )
        
        loader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 32),
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
        
        return loader
    
    def _train_epoch(self, loader, criterion, optimizer, epoch, stage=1):
        """Train single epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        for images, targets in loader:
            images, targets = images.to(self.device), targets.to(self.device)
            
            # Apply mixup/cutmix (only in stage 1)
            if stage == 1:
                use_mixup = np.random.rand() < self.config.get('mixup_prob', 0.5)
                if use_mixup and self.config.get('mixup_alpha', 0) > 0:
                    images, targets_a, targets_b, lam = mixup_data(
                        images, targets, alpha=self.config.get('mixup_alpha', 0.2)
                    )
                    outputs = self.model(images)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                elif self.config.get('cutmix_alpha', 0) > 0:
                    images, targets_a, targets_b, lam = cutmix_data(
                        images, targets, alpha=self.config.get('cutmix_alpha', 1.0)
                    )
                    outputs = self.model(images)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    outputs = self.model(images)
                    loss = criterion(outputs, targets)
            else:
                # Stage 2: No mixup/cutmix for cleaner hard class learning
                outputs = self.model(images)
                loss = criterion(outputs, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Metrics
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        epoch_loss = running_loss / len(loader)
        epoch_acc = balanced_accuracy_score(all_targets, all_preds)
        
        return epoch_loss, epoch_acc
    
    def _validate(self, loader, criterion):
        """Validate model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in loader:
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Compute metrics
        val_loss = running_loss / len(loader)
        balanced_acc = balanced_accuracy_score(all_targets, all_preds)
        recall_per_class = recall_score(all_targets, all_preds, average=None)
        
        metrics = {
            'val/loss': val_loss,
            'val/balanced_accuracy': balanced_acc,
            'val/recall_normal_mucosa': recall_per_class[0],
            'val/recall_normal_esophagus': recall_per_class[1],
            'val/recall_polyps': recall_per_class[2],
            'val/recall_erythema': recall_per_class[3],
            'val/worst_class_recall': recall_per_class.min(),
        }
        
        return metrics

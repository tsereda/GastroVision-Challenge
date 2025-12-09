import time
import numpy as np
import torch
import wandb
from pathlib import Path
import numpy as np
import torch
import wandb
import time
import argparse
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src import (
    GastroVisionDataset,
    SwinClassifier,
    get_loss_function,
    get_train_transforms,
    get_val_transforms,
    mixup_data,
    cutmix_data,
    mixup_criterion
)


def load_data():
    """Load GastroVision dataset from directory structure
    
    Returns:
        train_paths, train_labels, val_paths, val_labels
    """
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


def train_epoch(model, loader, criterion, optimizer, device, config, epoch):
    """Single training epoch with mixup/cutmix"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        
        # Apply mixup or cutmix
        use_mixup = np.random.rand() < config.get('mixup_prob', 0.5)
        if use_mixup and config.get('mixup_alpha', 0) > 0:
            images, targets_a, targets_b, lam = mixup_data(
                images, targets, alpha=config['mixup_alpha']
            )
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        elif config.get('cutmix_alpha', 0) > 0:
            images, targets_a, targets_b, lam = cutmix_data(
                images, targets, alpha=config['cutmix_alpha']
            )
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = balanced_accuracy_score(all_targets, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Validation with detailed metrics"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Compute metrics
    val_loss = running_loss / len(loader)
    balanced_acc = balanced_accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average='macro')
    recall_per_class = recall_score(all_targets, all_preds, average=None)
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    metrics = {
        'val/loss': val_loss,
        'val/balanced_accuracy': balanced_acc,
        'val/f1_macro': f1_macro,
        'val/recall_normal_mucosa': recall_per_class[0],
        'val/recall_normal_esophagus': recall_per_class[1],
        'val/recall_polyps': recall_per_class[2],
        'val/recall_erythema': recall_per_class[3],
        'val/worst_class_recall': recall_per_class.min(),
    }
    
    return metrics, conf_matrix


def train(config=None):
    """Main training function for W&B sweep"""
    with wandb.init(config=config) as run:
        config = wandb.config
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load data
        train_paths, train_labels, val_paths, val_labels = load_data()
        
        # Calculate class weights with optional boost
        unique, counts = np.unique(train_labels, return_counts=True)
        class_weights = 1.0 / counts
        class_weights = class_weights / class_weights.sum() * len(unique)
        
        # UPDATED: Apply boost more strategically based on empirical analysis
        # Analysis shows: boost helps bal_acc (+0.29 correlation) but HURTS Erythema (-0.04)
        # Solution: Differential boost strategy
        boost = config.get('class_weight_boost', 1.0)
        
        # Boost strategy based on 992-run sweep analysis:
        # - Mucosa (class 0): Moderate boost (0.8×) - drives balanced accuracy
        # - Esophagus (class 1): No boost - already 99.23% avg recall
        # - Polyps (class 2): Moderate boost (0.8×) - solid 93.14% avg recall
        # - Erythema (class 3): Higher boost (1.2×) - hardest class at 66.77% avg recall
        class_weights[0] *= boost * 0.8      # Normal mucosa
        # class_weights[1] stays at base     # Normal esophagus - already excellent
        class_weights[2] *= boost * 0.8      # Polyps
        class_weights[3] *= boost * 1.2      # Erythema - needs disproportionate help
        
        class_weights = class_weights / class_weights.sum() * len(unique)  # Re-normalize
        
        print(f"\nClass weights (with {boost}x boost, Erythema×1.2):")
        print(f"  Class 0 (Mucosa):    {class_weights[0]:.3f}")
        print(f"  Class 1 (Esophagus): {class_weights[1]:.3f}")
        print(f"  Class 2 (Polyps):    {class_weights[2]:.3f}")
        print(f"  Class 3 (Erythema):  {class_weights[3]:.3f}")
        
        # Create datasets
        train_dataset = GastroVisionDataset(
            train_paths, train_labels, 
            transform=get_train_transforms(config)
        )
        val_dataset = GastroVisionDataset(
            val_paths, val_labels, 
            transform=get_val_transforms(config['image_size'])
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=config['batch_size'],
            shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config['batch_size'],
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        # Create model
        model = SwinClassifier(
            model_name=config['model_name'],
            num_classes=4,
            dropout=config.get('dropout_rate', 0.3),
            stochastic_depth=config.get('stochastic_depth', 0.2)
        ).to(device)
        
        # Loss, optimizer, scheduler
        criterion = get_loss_function(
            config['loss_function'],
            class_weights=class_weights,
            focal_gamma=config.get('focal_gamma', 2.0),
            label_smoothing=config.get('label_smoothing', 0.1),
            lambda_worst=config.get('lambda_worst', 0.3),  # For dynamic_worst_class loss
            poly_epsilon=config.get('poly_epsilon', 1.0),  # For poly loss
            device=device
        )
        
        optimizer = AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        total_epochs = config.get('epochs', 100)
        warmup_epochs = config.get('warmup_epochs', 10)
        
        if config.get('scheduler') == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
        else:
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs
        )
        
        # Training loop with early stopping
        best_balanced_acc = 0.0
        patience = 8
        patience_counter = 0
        
        for epoch in range(1, total_epochs + 1):
            epoch_start = time.time()
            
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, config, epoch
            )
            
            val_metrics, conf_matrix = validate(model, val_loader, criterion, device)
            
            if epoch <= warmup_epochs:
                warmup_scheduler.step()
            else:
                scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            val_acc = val_metrics['val/balanced_accuracy']
            val_loss = val_metrics['val/loss']
            epoch_time = time.time() - epoch_start
            
            # Per-class recall display
            class_names = ['Mucosa', 'Esoph', 'Polyps', 'Eryth']
            recall_per_class = [
                val_metrics['val/recall_normal_mucosa'],
                val_metrics['val/recall_normal_esophagus'],
                val_metrics['val/recall_polyps'],
                val_metrics['val/recall_erythema']
            ]
            worst_idx = np.argmin(recall_per_class)
            worst_class = class_names[worst_idx]
            worst_recall = recall_per_class[worst_idx]
            recall_str = '/'.join([f'{r:.2f}' for r in recall_per_class])
            
            # Clean epoch summary
            is_best = val_acc > best_balanced_acc
            best_marker = ' ⭐ Best!' if is_best else ''
            print(f"Epoch {epoch}/{total_epochs} [{epoch_time:.1f}s] | "
                  f"Train: Loss={train_loss:.3f} Acc={train_acc:.3f} | "
                  f"Val: Loss={val_loss:.3f} Acc={val_acc:.3f} F1={val_metrics['val/f1_macro']:.3f} | "
                  f"Recall:[{recall_str}] Worst:{worst_class}={worst_recall:.2f} | "
                  f"LR={current_lr:.2e}{best_marker}")
            
            # Log to W&B
            log_dict = {
                'epoch': epoch,
                'train/loss': train_loss,
                'train/balanced_accuracy': train_acc,
                'learning_rate': current_lr,
                **val_metrics
            }
            
            # If using dynamic loss, log class performance
            if hasattr(criterion, 'get_class_performance'):
                class_perf = criterion.get_class_performance()
                log_dict.update({
                    'dynamic/class_0_ema': class_perf[0],
                    'dynamic/class_1_ema': class_perf[1],
                    'dynamic/class_2_ema': class_perf[2],
                    'dynamic/class_3_ema': class_perf[3],
                })
            
            wandb.log(log_dict)
            
            # Save best model
            if is_best:
                best_balanced_acc = val_acc
                patience_counter = 0  # Reset patience counter
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'balanced_acc': best_balanced_acc,
                    'config': dict(config)
                }, f'best_model_{run.id}.pth')
                print(f"  → Saved checkpoint: best_model_{run.id}.pth (Acc: {best_balanced_acc:.4f})")
                print(f"✓ New best: {best_balanced_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⚠️  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                    print(f"   Best validation accuracy: {best_balanced_acc:.4f}")
                    break
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation balanced accuracy: {best_balanced_acc:.4f}")
        print(f"{'='*60}")


def parse_args():
    """Parse command line arguments for standalone training"""
    parser = argparse.ArgumentParser(description='GastroVision Training')
    
    # Model architecture
    parser.add_argument('--model_name', type=str, default='swin_base_patch4_window12_384')
    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--stochastic_depth', type=float, default=0.2)
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'warmrestarts'])
    
    # Loss function
    parser.add_argument('--loss_function', type=str, default='asymmetric')
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--lambda_worst', type=float, default=0.3)
    parser.add_argument('--poly_epsilon', type=float, default=1.0)
    parser.add_argument('--class_weight_boost', type=float, default=1.0)
    
    # Augmentations
    parser.add_argument('--color_jitter_brightness', type=float, default=0.2)
    parser.add_argument('--color_jitter_contrast', type=float, default=0.2)
    parser.add_argument('--color_jitter_saturation', type=float, default=0.2)
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--mixup_prob', type=float, default=0.5)
    parser.add_argument('--cutmix_alpha', type=float, default=1.0)
    
    return parser.parse_args()


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    
    # Convert argparse Namespace to dict for wandb
    config_dict = vars(args)
    
    # Run training with parsed config
    train(config=config_dict)

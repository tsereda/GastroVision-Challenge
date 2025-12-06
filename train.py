import torch
import wandb
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, confusion_matrix
from tqdm import tqdm

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
    """Load your dataset - IMPLEMENT THIS
    
    Returns:
        train_paths, train_labels, val_paths, val_labels
    """
    # TODO: Implement your data loading
    # Example:
    # train_df = pd.read_csv('train.csv')
    # train_paths = train_df['image_path'].tolist()
    # train_labels = train_df['label'].tolist()
    # ...
    raise NotImplementedError("Implement your data loading here!")


def train_epoch(model, loader, criterion, optimizer, device, config, epoch):
    """Single training epoch with mixup/cutmix"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    for images, targets in pbar:
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
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
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
        for images, targets in tqdm(loader, desc='Validation'):
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
        
        # Calculate class weights
        unique, counts = np.unique(train_labels, return_counts=True)
        class_weights = 1.0 / counts
        class_weights = class_weights / class_weights.sum() * len(unique)
        print(f"Class weights: {class_weights}")
        
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
        
        # Training loop
        best_balanced_acc = 0.0
        
        for epoch in range(1, total_epochs + 1):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, config, epoch
            )
            
            val_metrics, conf_matrix = validate(model, val_loader, criterion, device)
            
            if epoch <= warmup_epochs:
                warmup_scheduler.step()
            else:
                scheduler.step()
            
            # Log to W&B
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/balanced_accuracy': train_acc,
                'learning_rate': optimizer.param_groups[0]['lr'],
                **val_metrics
            })
            
            # Save best model
            if val_metrics['val/balanced_accuracy'] > best_balanced_acc:
                best_balanced_acc = val_metrics['val/balanced_accuracy']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'balanced_acc': best_balanced_acc,
                    'config': dict(config)
                }, f'best_model_{run.id}.pth')
                print(f"✓ New best: {best_balanced_acc:.4f}")


if __name__ == '__main__':
    train()

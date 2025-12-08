import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class FocalWeightedLoss(nn.Module):
    """Focal Loss + Class Weighting"""
    def __init__(self, class_weights, gamma=2.0):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, 
            reduction='none',
            weight=self.class_weights
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class DynamicWorstClassLoss(nn.Module):
    """Dynamic loss that emphasizes worst-performing class
    
    Implements Distributionally Robust Optimization (DRO) strategy:
    Loss = base_loss + lambda * worst_class_loss
    
    This addresses the "Whac-A-Mole" problem where improving one class
    degrades another by dynamically focusing on the worst performer.
    
    Args:
        class_weights: Static class weights (optional)
        gamma: Focal loss gamma parameter
        lambda_worst: Weight for worst-class penalty (0-1)
        num_classes: Number of classes
    """
    def __init__(self, class_weights=None, gamma=2.0, lambda_worst=0.3, num_classes=4):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.lambda_worst = lambda_worst
        self.num_classes = num_classes
        
        # Track per-class performance (exponential moving average)
        self.register_buffer('class_recall_ema', torch.ones(num_classes))
        self.ema_decay = 0.9
    
    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        
        # Base focal loss with static weights
        ce_loss = F.cross_entropy(
            inputs, targets,
            reduction='none',
            weight=self.class_weights
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        base_loss = focal_loss.mean()
        
        # Compute per-class performance in this batch
        with torch.no_grad():
            preds = inputs.argmax(dim=1)
            for c in range(self.num_classes):
                class_mask = (targets == c)
                if class_mask.sum() > 0:
                    class_acc = (preds[class_mask] == c).float().mean()
                    # Update EMA
                    self.class_recall_ema[c] = (
                        self.ema_decay * self.class_recall_ema[c] + 
                        (1 - self.ema_decay) * class_acc
                    )
        
        # Identify worst-performing class
        worst_class = self.class_recall_ema.argmin()
        
        # Additional penalty for worst class samples
        worst_class_mask = (targets == worst_class)
        if worst_class_mask.sum() > 0:
            worst_class_loss = focal_loss[worst_class_mask].mean()
        else:
            worst_class_loss = torch.tensor(0.0, device=inputs.device)
        
        # Combined loss
        total_loss = base_loss + self.lambda_worst * worst_class_loss
        
        return total_loss
    
    def get_class_performance(self):
        """Return current class performance estimates"""
        return self.class_recall_ema.cpu().numpy()


def get_loss_function(loss_name, class_weights=None, focal_gamma=2.0, label_smoothing=0.1, 
                      lambda_worst=0.3, device='cuda'):
    """Factory for loss functions
    
    Args:
        loss_name: One of ['focal', 'weighted_ce', 'focal_weighted']
        class_weights: Tensor of class weights
        focal_gamma: Gamma parameter for focal loss
        label_smoothing: Label smoothing factor
        device: Device to put tensors on
    """
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    if loss_name == 'focal':
        return FocalLoss(alpha=None, gamma=focal_gamma)
    elif loss_name == 'weighted_ce':
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    elif loss_name == 'focal_weighted':
        return FocalWeightedLoss(class_weights=class_weights, gamma=focal_gamma)
    elif loss_name == 'dynamic_worst_class':
        return DynamicWorstClassLoss(
            class_weights=class_weights, 
            gamma=focal_gamma, 
            lambda_worst=lambda_worst,
            num_classes=4
        )
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

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


def get_loss_function(loss_name, class_weights=None, focal_gamma=2.0, label_smoothing=0.1, device='cuda'):
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
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

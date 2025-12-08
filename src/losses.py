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


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Class Classification.
    Optimized for imbalanced datasets (like GastroVision).
    Supports Soft Targets (Mixup/Cutmix).
    
    Decouples focusing for positive/negative samples:
    - Aggressively silences easy negative samples (e.g., Normal Esophagus)
    - Focuses gradients on hard positive samples (e.g., Erythema)
    
    Args:
        gamma_neg: Focusing parameter for negative samples (default: 4)
        gamma_pos: Focusing parameter for positive samples (default: 1)
        clip: Clipping value for negative probabilities (default: 0.05)
        eps: Numerical stability constant
        disable_torch_grad_focal_loss: Disable grad for focal weight computation
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, 
                 disable_torch_grad_focal_loss=True):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """
        Args:
            x: input logits [batch_size, num_classes]
            y: targets (integers or soft targets from mixup)
        """
        # Convert hard labels to one-hot if needed
        if y.ndim == 1:
            num_classes = x.size(1)
            y = F.one_hot(y, num_classes).float()
        
        # Calculate probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic Cross Entropy Calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        
        # Asymmetric Focusing
        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(False)
        pt0 = xs_pos * y
        pt1 = xs_neg * (1 - y)
        pt = pt0 + pt1
        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(True)
            
        loss = -one_sided_w * (los_pos + los_neg)
        return loss.sum() / x.size(0)  # Average over batch


class Poly1CrossEntropyLoss(nn.Module):
    """
    PolyLoss: A Polynomial Expansion of Cross Entropy.
    Often performs better than Focal Loss on imbalanced medical data.
    Supports Soft Targets from Mixup/Cutmix.
    
    Loss = CE + epsilon * (1 - Pt)
    
    Where Pt is the probability of the target class. This provides a more
    stable alternative to Focal Loss for Transformer models.
    
    Args:
        num_classes: Number of classes
        epsilon: Polynomial coefficient (default: 1.0, try 2.0 for stronger effect)
        weight: Class weights tensor
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, num_classes, epsilon=1.0, weight=None, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.weight = weight
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, logits, targets):
        # Handle mixup (soft targets)
        if targets.ndim > 1:
            # If targets are one-hot/mixed, use soft-label approach
            probs = F.softmax(logits, dim=-1)
            # Standard CE part
            ce_loss = -torch.sum(targets * torch.log(probs + 1e-8), dim=-1)
            # Poly1 term: epsilon * (1 - Pt)
            # Pt is the probability of the target class
            pt = torch.sum(targets * probs, dim=-1)
            poly1 = self.epsilon * (1 - pt)
            loss = ce_loss + poly1
        else:
            # Standard hard targets
            ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
            pt = torch.exp(-ce_loss)
            loss = ce_loss + self.epsilon * (1 - pt)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


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


class HierarchicalLoss(nn.Module):
    """Hierarchical loss for multi-level classification
    
    GastroVision has a natural hierarchy:
    Level 1: Normal (0, 1) vs Abnormal (2, 3)
    Level 2: Within each group, specific classification
    
    This loss encourages the model to learn both coarse and fine distinctions.
    
    Hierarchy:
    - Normal: Normal Mucosa (0), Normal Esophagus (1)
    - Abnormal: Colon Polyps (2), Erythema (3)
    
    Loss = alpha * L1_loss + (1 - alpha) * L2_loss
    
    Args:
        alpha: Weight for coarse (L1) vs fine (L2) loss (default: 0.3)
        class_weights: Class weights for fine-grained loss
        gamma: Focal loss gamma
    """
    def __init__(self, alpha=0.3, class_weights=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
        # Level 1: Normal (0) vs Abnormal (1)
        self.coarse_criterion = nn.CrossEntropyLoss()
        
        # Level 2: Fine-grained within each group
        if class_weights is not None:
            self.fine_criterion = FocalWeightedLoss(class_weights, gamma)
        else:
            self.fine_criterion = FocalLoss(gamma=gamma)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model logits (B, 4)
            targets: Fine-grained labels 0-3 (B,)
        """
        batch_size = inputs.size(0)
        
        # ===== LEVEL 1: Coarse Classification (Normal vs Abnormal) =====
        # Create coarse labels: 0,1 -> 0 (Normal), 2,3 -> 1 (Abnormal)
        coarse_targets = (targets >= 2).long()
        
        # Coarse logits: sum probabilities for each group
        fine_probs = F.softmax(inputs, dim=1)
        
        # Normal group: class 0 + class 1
        normal_prob = fine_probs[:, 0] + fine_probs[:, 1]
        # Abnormal group: class 2 + class 3
        abnormal_prob = fine_probs[:, 2] + fine_probs[:, 3]
        
        coarse_logits = torch.stack([normal_prob, abnormal_prob], dim=1)
        coarse_logits = torch.log(coarse_logits + 1e-8)  # Convert to log-probs
        
        coarse_loss = self.coarse_criterion(coarse_logits, coarse_targets)
        
        # ===== LEVEL 2: Fine-grained Classification =====
        fine_loss = self.fine_criterion(inputs, targets)
        
        # ===== COMBINED LOSS =====
        total_loss = self.alpha * coarse_loss + (1 - self.alpha) * fine_loss
        
        return total_loss


class HierarchicalFocalLoss(nn.Module):
    """Enhanced hierarchical loss with focal weighting at both levels
    
    Combines hierarchical structure with focal loss benefits:
    - Coarse level: Focus on hard Normal vs Abnormal distinctions
    - Fine level: Focus on hard within-group distinctions (e.g., Erythema)
    
    Args:
        alpha: Weight for coarse vs fine loss (default: 0.3)
        gamma_coarse: Focal gamma for coarse classification (default: 2.0)
        gamma_fine: Focal gamma for fine classification (default: 2.0)
        class_weights: Class weights for fine-grained loss
    """
    def __init__(self, alpha=0.3, gamma_coarse=2.0, gamma_fine=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma_coarse = gamma_coarse
        self.gamma_fine = gamma_fine
        
        # Coarse-level focal loss
        self.coarse_criterion = FocalLoss(gamma=gamma_coarse)
        
        # Fine-level focal loss
        if class_weights is not None:
            self.fine_criterion = FocalWeightedLoss(class_weights, gamma=gamma_fine)
        else:
            self.fine_criterion = FocalLoss(gamma=gamma_fine)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model logits (B, 4)
            targets: Fine-grained labels 0-3 (B,)
        """
        # Coarse targets
        coarse_targets = (targets >= 2).long()
        
        # Coarse logits
        fine_probs = F.softmax(inputs, dim=1)
        normal_prob = fine_probs[:, 0] + fine_probs[:, 1]
        abnormal_prob = fine_probs[:, 2] + fine_probs[:, 3]
        coarse_logits = torch.stack([normal_prob, abnormal_prob], dim=1)
        coarse_logits = torch.log(coarse_logits + 1e-8)
        
        # Coarse focal loss
        coarse_loss = self.coarse_criterion(coarse_logits, coarse_targets)
        
        # Fine focal loss
        fine_loss = self.fine_criterion(inputs, targets)
        
        # Combined
        total_loss = self.alpha * coarse_loss + (1 - self.alpha) * fine_loss
        
        return total_loss


def get_loss_function(loss_name, class_weights=None, focal_gamma=2.0, label_smoothing=0.1, 
                      lambda_worst=0.3, poly_epsilon=1.0, hierarchical_alpha=0.3, device='cuda'):
    """Factory for loss functions
    
    Args:
        loss_name: One of ['focal', 'weighted_ce', 'focal_weighted', 'dynamic_worst_class',
                          'asymmetric', 'poly', 'hierarchical', 'hierarchical_focal']
        class_weights: Tensor of class weights
        focal_gamma: Gamma parameter for focal loss
        label_smoothing: Label smoothing factor
        lambda_worst: Lambda for dynamic worst class loss
        poly_epsilon: Epsilon for PolyLoss (1.0-3.0, higher = stronger)
        hierarchical_alpha: Weight for coarse vs fine loss in hierarchical losses
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
    elif loss_name == 'asymmetric':
        # Asymmetric Loss: Aggressive on easy negatives, focus on hard positives
        # Recommended for datasets where one class dominates (e.g., Normal Esophagus)
        return AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    elif loss_name == 'poly':
        # PolyLoss: Polynomial expansion of CE, often more stable than Focal
        # Good for Transformers on imbalanced medical data
        return Poly1CrossEntropyLoss(
            num_classes=4,
            epsilon=poly_epsilon,
            weight=class_weights
        )
    elif loss_name == 'hierarchical':
        # Hierarchical Loss: Normal vs Abnormal, then fine-grained
        # Good for learning class relationships
        return HierarchicalLoss(
            alpha=hierarchical_alpha,
            class_weights=class_weights,
            gamma=focal_gamma
        )
    elif loss_name == 'hierarchical_focal':
        # Hierarchical + Focal: Best of both worlds
        # Combines class hierarchy with hard example mining
        return HierarchicalFocalLoss(
            alpha=hierarchical_alpha,
            gamma_coarse=focal_gamma,
            gamma_fine=focal_gamma,
            class_weights=class_weights
        )
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

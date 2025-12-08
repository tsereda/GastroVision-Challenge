"""
Test-Time Augmentation (TTA) for improved inference
Applies multiple augmentations and averages predictions
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Callable


class TTAWrapper:
    """Test-Time Augmentation wrapper for models
    
    Applies multiple augmentations to test images and averages predictions
    to improve robustness and accuracy.
    
    Args:
        model: PyTorch model
        transforms: List of TTA transform functions
        device: Device to run on
    """
    
    def __init__(self, model, transforms: List[Callable] = None, device='cuda'):
        self.model = model
        self.device = device
        
        if transforms is None:
            # Default TTA transforms
            self.transforms = [
                self._no_transform,
                self._hflip,
                self._vflip,
                self._hflip_vflip,
                self._rotate_90,
                self._rotate_180,
                self._rotate_270,
                self._center_crop_95,
            ]
        else:
            self.transforms = transforms
        
        print(f"TTA initialized with {len(self.transforms)} augmentations")
    
    def _no_transform(self, x):
        """Identity transform"""
        return x
    
    def _hflip(self, x):
        """Horizontal flip"""
        return torch.flip(x, dims=[3])
    
    def _vflip(self, x):
        """Vertical flip"""
        return torch.flip(x, dims=[2])
    
    def _hflip_vflip(self, x):
        """Horizontal + Vertical flip"""
        return torch.flip(x, dims=[2, 3])
    
    def _rotate_90(self, x):
        """Rotate 90 degrees clockwise"""
        return torch.rot90(x, k=1, dims=[2, 3])
    
    def _rotate_180(self, x):
        """Rotate 180 degrees"""
        return torch.rot90(x, k=2, dims=[2, 3])
    
    def _rotate_270(self, x):
        """Rotate 270 degrees clockwise"""
        return torch.rot90(x, k=3, dims=[2, 3])
    
    def _center_crop_95(self, x):
        """Center crop to 95% of original size then resize back"""
        B, C, H, W = x.shape
        crop_h, crop_w = int(H * 0.95), int(W * 0.95)
        
        top = (H - crop_h) // 2
        left = (W - crop_w) // 2
        
        cropped = x[:, :, top:top+crop_h, left:left+crop_w]
        return F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)
    
    @torch.no_grad()
    def predict(self, images, return_probs=True):
        """Predict with TTA
        
        Args:
            images: Batch of images (B, C, H, W)
            return_probs: Return probability distributions
        
        Returns:
            predictions: Class predictions (B,)
            probabilities: Average probabilities across all TTA transforms (B, num_classes)
        """
        images = images.to(self.device)
        batch_size = images.size(0)
        num_classes = 4
        
        # Accumulate probabilities
        total_probs = torch.zeros(batch_size, num_classes).to(self.device)
        
        for transform in self.transforms:
            # Apply transform
            transformed = transform(images)
            
            # Get predictions
            logits = self.model(transformed)
            probs = torch.softmax(logits, dim=1)
            
            # Accumulate
            total_probs += probs
        
        # Average probabilities
        avg_probs = total_probs / len(self.transforms)
        
        # Get final predictions
        predictions = torch.argmax(avg_probs, dim=1)
        
        if return_probs:
            return predictions, avg_probs
        else:
            return predictions


class MultiScaleTTA:
    """Multi-scale Test-Time Augmentation
    
    Tests at different image scales to capture features at multiple resolutions.
    Particularly useful for medical images where scale variations matter.
    
    Args:
        model: PyTorch model
        scales: List of scale factors (e.g., [0.9, 1.0, 1.1])
        base_size: Base image size
        device: Device to run on
    """
    
    def __init__(self, model, scales: List[float] = None, base_size: int = 384, device='cuda'):
        self.model = model
        self.device = device
        self.base_size = base_size
        
        if scales is None:
            self.scales = [0.9, 1.0, 1.1]
        else:
            self.scales = scales
        
        print(f"Multi-Scale TTA initialized with scales: {self.scales}")
    
    @torch.no_grad()
    def predict(self, images, return_probs=True):
        """Predict with multi-scale TTA
        
        Args:
            images: Batch of images (B, C, H, W)
            return_probs: Return probability distributions
        
        Returns:
            predictions: Class predictions (B,)
            probabilities: Average probabilities across scales (B, num_classes)
        """
        images = images.to(self.device)
        batch_size = images.size(0)
        num_classes = 4
        
        total_probs = torch.zeros(batch_size, num_classes).to(self.device)
        
        for scale in self.scales:
            # Resize to scale
            scaled_size = int(self.base_size * scale)
            scaled_images = F.interpolate(
                images, 
                size=(scaled_size, scaled_size), 
                mode='bilinear', 
                align_corners=False
            )
            
            # Get predictions
            logits = self.model(scaled_images)
            probs = torch.softmax(logits, dim=1)
            
            # Accumulate
            total_probs += probs
        
        # Average
        avg_probs = total_probs / len(self.scales)
        predictions = torch.argmax(avg_probs, dim=1)
        
        if return_probs:
            return predictions, avg_probs
        else:
            return predictions


class CombinedTTA:
    """Combines standard TTA with multi-scale TTA
    
    Args:
        model: PyTorch model
        use_flips: Apply horizontal/vertical flips
        use_rotations: Apply 90-degree rotations
        use_scales: Apply multi-scale testing
        scales: Scale factors for multi-scale
        device: Device to run on
    """
    
    def __init__(self, 
                 model, 
                 use_flips=True, 
                 use_rotations=True, 
                 use_scales=True,
                 scales=None,
                 device='cuda'):
        self.model = model
        self.device = device
        self.use_scales = use_scales
        
        # Build transform list
        transforms = [lambda x: x]  # Identity
        
        if use_flips:
            transforms.extend([
                lambda x: torch.flip(x, dims=[3]),  # hflip
                lambda x: torch.flip(x, dims=[2]),  # vflip
            ])
        
        if use_rotations:
            transforms.extend([
                lambda x: torch.rot90(x, k=1, dims=[2, 3]),  # 90 degrees
                lambda x: torch.rot90(x, k=3, dims=[2, 3]),  # 270 degrees
            ])
        
        self.transforms = transforms
        self.scales = scales if scales else [0.95, 1.0, 1.05]
        
        total_augs = len(self.transforms) * (len(self.scales) if use_scales else 1)
        print(f"Combined TTA: {total_augs} total augmentations")
        print(f"  - {len(self.transforms)} geometric transforms")
        print(f"  - {len(self.scales) if use_scales else 1} scales")
    
    @torch.no_grad()
    def predict(self, images, return_probs=True):
        """Predict with combined TTA"""
        images = images.to(self.device)
        batch_size = images.size(0)
        num_classes = 4
        original_size = images.size(-1)
        
        total_probs = torch.zeros(batch_size, num_classes).to(self.device)
        num_augmentations = 0
        
        # Iterate over scales
        scales = self.scales if self.use_scales else [1.0]
        
        for scale in scales:
            # Resize if needed
            if scale != 1.0:
                scaled_size = int(original_size * scale)
                scaled_images = F.interpolate(
                    images,
                    size=(scaled_size, scaled_size),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                scaled_images = images
            
            # Apply each transform
            for transform in self.transforms:
                transformed = transform(scaled_images)
                
                # Get predictions
                logits = self.model(transformed)
                probs = torch.softmax(logits, dim=1)
                
                total_probs += probs
                num_augmentations += 1
        
        # Average
        avg_probs = total_probs / num_augmentations
        predictions = torch.argmax(avg_probs, dim=1)
        
        if return_probs:
            return predictions, avg_probs
        else:
            return predictions

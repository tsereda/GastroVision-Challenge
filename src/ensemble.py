"""
Ensemble prediction module for combining multiple models
Supports weighted averaging and soft voting
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import wandb


class ModelEnsemble:
    """Ensemble multiple trained models for improved predictions
    
    Args:
        model_configs: List of dicts with 'checkpoint_path', 'weight', 'config'
        device: Device to run inference on
        mode: 'soft' (average probabilities) or 'hard' (majority vote)
    """
    
    def __init__(self, model_configs: List[Dict], device='cuda', mode='soft'):
        self.device = device
        self.mode = mode
        self.models = []
        self.weights = []
        
        for config in model_configs:
            model = self._load_model(
                config['checkpoint_path'],
                config.get('model_config', {})
            )
            self.models.append(model)
            self.weights.append(config.get('weight', 1.0))
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        print(f"Ensemble initialized with {len(self.models)} models")
        print(f"Weights: {self.weights}")
    
    def _load_model(self, checkpoint_path, model_config):
        """Load a single model from checkpoint"""
        from src.model import SwinClassifier
        
        # Create model
        model = SwinClassifier(
            model_name=model_config.get('model_name', 'swin_base_patch4_window12_384'),
            num_classes=4,
            pretrained=False,
            dropout=model_config.get('dropout_rate', 0.3),
            stochastic_depth=model_config.get('stochastic_depth', 0.2)
        )
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    @torch.no_grad()
    def predict(self, images):
        """Predict using ensemble
        
        Args:
            images: Batch of images (B, C, H, W)
        
        Returns:
            predictions: Class predictions (B,)
            probabilities: Class probabilities (B, num_classes)
        """
        images = images.to(self.device)
        
        if self.mode == 'soft':
            # Weighted average of probabilities
            ensemble_probs = None
            
            for model, weight in zip(self.models, self.weights):
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                
                if ensemble_probs is None:
                    ensemble_probs = weight * probs
                else:
                    ensemble_probs += weight * probs
            
            predictions = torch.argmax(ensemble_probs, dim=1)
            return predictions, ensemble_probs
        
        else:  # hard voting
            # Majority vote
            all_preds = []
            
            for model in self.models:
                logits = model(images)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds)
            
            all_preds = torch.stack(all_preds)  # (num_models, B)
            
            # Majority vote for each sample
            predictions = torch.mode(all_preds, dim=0)[0]
            
            # Return dummy probabilities for hard voting
            batch_size = images.size(0)
            probabilities = torch.zeros(batch_size, 4).to(self.device)
            probabilities[torch.arange(batch_size), predictions] = 1.0
            
            return predictions, probabilities


def create_ensemble_from_wandb(run_ids: List[str], 
                               entity: str = 'timgsereda',
                               project: str = 'gastrovision-challenge',
                               weights: Optional[List[float]] = None,
                               device: str = 'cuda'):
    """Create ensemble by downloading checkpoints from W&B runs
    
    Args:
        run_ids: List of W&B run IDs to include in ensemble
        entity: W&B entity name
        project: W&B project name
        weights: Optional list of weights for each model
        device: Device to run on
    
    Returns:
        ModelEnsemble instance
    """
    api = wandb.Api()
    model_configs = []
    
    checkpoint_dir = Path('./checkpoints/ensemble')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for i, run_id in enumerate(run_ids):
        run_path = f"{entity}/{project}/{run_id}"
        print(f"\nDownloading model {i+1}/{len(run_ids)}: {run_id}")
        
        run = api.run(run_path)
        
        # Download checkpoint
        checkpoint_name = f'best_model_{run_id}.pth'
        checkpoint_path = checkpoint_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            for f in run.files():
                if f.name.endswith('.pth'):
                    f.download(root=str(checkpoint_dir), replace=False)
                    # Rename to standard format
                    downloaded_path = checkpoint_dir / f.name
                    if downloaded_path != checkpoint_path:
                        downloaded_path.rename(checkpoint_path)
                    break
        
        # Get model config from run
        config = run.config
        model_config = {
            'model_name': config.get('model_name', 'swin_base_patch4_window12_384'),
            'dropout_rate': config.get('dropout_rate', 0.3),
            'stochastic_depth': config.get('stochastic_depth', 0.2),
        }
        
        model_configs.append({
            'checkpoint_path': str(checkpoint_path),
            'model_config': model_config,
            'weight': weights[i] if weights else 1.0
        })
    
    return ModelEnsemble(model_configs, device=device)


def get_top_runs_from_sweep(sweep_id: str,
                            entity: str = 'timgsereda', 
                            project: str = 'gastrovision-challenge',
                            top_k: int = 5,
                            metric: str = 'val/balanced_accuracy') -> List[str]:
    """Get top K runs from a W&B sweep
    
    Args:
        sweep_id: W&B sweep ID
        entity: W&B entity
        project: W&B project
        top_k: Number of top runs to retrieve
        metric: Metric to sort by
    
    Returns:
        List of run IDs
    """
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    
    # Get all runs and sort by metric
    runs = []
    for run in sweep.runs:
        if run.state == 'finished' and metric in run.summary:
            runs.append((run.id, run.summary[metric]))
    
    # Sort by metric (descending)
    runs.sort(key=lambda x: x[1], reverse=True)
    
    # Get top K
    top_runs = [run_id for run_id, _ in runs[:top_k]]
    
    print(f"\nTop {top_k} runs from sweep {sweep_id}:")
    for i, (run_id, score) in enumerate(runs[:top_k], 1):
        print(f"  {i}. {run_id}: {metric}={score:.4f}")
    
    return top_runs

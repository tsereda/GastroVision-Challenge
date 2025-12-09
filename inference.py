"""
Inference script for GastroVision Challenge
Downloads model from W&B run and generates submission CSV
"""

import argparse
import torch
import wandb
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from src import SwinClassifier, get_val_transforms


class TestDataset(Dataset):
    """Dataset for test images (no labels)"""
    
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


def load_model_from_wandb(run_path, device='cuda'):
    """
    Download model checkpoint from W&B run
    
    Args:
        run_path: W&B run path (e.g., 'timgsereda/gastrovision-challenge/8npbpiz3')
        device: Device to load model on
    
    Returns:
        model, config
    """
    print(f"🔄 Downloading model from W&B run: {run_path}")
    
    # Initialize W&B API
    api = wandb.Api()
    run = api.run(run_path)
    
    # Get run config
    config = run.config
    print(f"\n📋 Run Config:")
    print(f"   Model: {config.get('model_name', 'swin_base_patch4_window12_384')}")
    print(f"   Image Size: {config.get('img_size', 384)}")
    print(f"   Best Val Acc: {run.summary.get('val/balanced_accuracy', 'N/A')}")
    
    # Download checkpoint
    checkpoint_file = None
    for file in run.files():
        if file.name == 'best_model.pth':
            checkpoint_file = file
            break
    
    if checkpoint_file is None:
        raise ValueError(f"No best_model.pth found in run {run_path}")
    
    checkpoint_path = checkpoint_file.download(replace=True, root='./wandb_downloads')
    print(f"✅ Downloaded checkpoint to: {checkpoint_path.name}")
    
    # Create model
    model = SwinClassifier(
        model_name=config.get('model_name', 'swin_base_patch4_window12_384'),
        num_classes=4,
        pretrained=False,
        dropout=config.get('dropout', 0.3),
        stochastic_depth=config.get('stochastic_depth', 0.2)
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path.name, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    
    return model, config


def run_inference(model, test_loader, device='cuda'):
    """
    Run inference on test dataset
    
    Returns:
        predictions: List of (image_id, predicted_class_name)
    """
    # Class mapping (inverse of training)
    class_names = [
        'Normal_mucosa_large_bowel',  # 0
        'Normal_esophagus',             # 1
        'colon polyp',                  # 2
        'Erythema'                      # 3
    ]
    
    predictions = []
    
    print(f"\n🔮 Running inference on {len(test_loader.dataset)} images...")
    
    with torch.no_grad():
        for images, image_ids in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            for img_id, pred in zip(image_ids, preds):
                predictions.append((img_id, class_names[pred]))
    
    return predictions


def save_submission(predictions, output_path='submission_timothysereda.csv'):
    """Save predictions to CSV in required format"""
    
    print(f"\n💾 Saving submission to: {output_path}")
    
    with open(output_path, 'w') as f:
        f.write('image_id,prediction\n')
        for img_id, pred_class in sorted(predictions):
            f.write(f'{img_id},{pred_class}\n')
    
    print(f"✅ Submission saved! ({len(predictions)} predictions)")
    
    # Print class distribution
    class_counts = {}
    for _, pred_class in predictions:
        class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
    
    print(f"\n📊 Prediction Distribution:")
    for class_name, count in sorted(class_counts.items()):
        print(f"   {class_name}: {count} ({100*count/len(predictions):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='GastroVision Challenge Inference')
    parser.add_argument(
        '--run', 
        type=str, 
        required=True,
        help='W&B run path (e.g., timgsereda/gastrovision-challenge/8npbpiz3 or just run_id)'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        default='Gastrovision Challenge dataset/Test dataset',
        help='Path to test dataset directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='submission_timothysereda.csv',
        help='Output CSV filename'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for inference'
    )
    
    args = parser.parse_args()
    
    # Format run path
    if '/' not in args.run:
        # Just run ID provided, add project path
        run_path = f"timgsereda/gastrovision-challenge/{args.run}"
    else:
        run_path = args.run
    
    print("=" * 60)
    print("🔬 GastroVision Challenge - Inference")
    print("=" * 60)
    
    # Load model from W&B
    model, config = load_model_from_wandb(run_path, device=args.device)
    
    # Get test images
    test_dir = Path(args.test_dir)
    test_images = sorted(list(test_dir.glob('*.jpg')))
    print(f"\n📁 Found {len(test_images)} test images in {test_dir}")
    
    # Create test dataset
    img_size = config.get('img_size', 384)
    transforms = get_val_transforms(img_size=img_size)
    test_dataset = TestDataset(test_images, transform=transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Run inference
    predictions = run_inference(model, test_loader, device=args.device)
    
    # Save submission
    save_submission(predictions, output_path=args.output)
    
    print("\n" + "=" * 60)
    print("✅ Inference completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Prediction script for GastroVision Challenge
Loads a trained model and generates predictions for test images
"""

import torch
import numpy as np
import pandas as pd
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
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Return image and filename
        filename = Path(img_path).name
        return image, filename


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    
    print(f"\nLoading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config from checkpoint
    config = checkpoint.get('config', {})
    
    # Model parameters
    model_name = config.get('model_name', 'swin_base_patch4_window7_224')
    dropout = config.get('dropout_rate', 0.3)
    stochastic_depth = config.get('stochastic_depth', 0.2)
    
    print(f"  Model: {model_name}")
    print(f"  Dropout: {dropout}")
    print(f"  Stochastic depth: {stochastic_depth}")
    
    # Create model
    model = SwinClassifier(
        model_name=model_name,
        num_classes=4,
        pretrained=False,  # We're loading weights
        dropout=dropout,
        stochastic_depth=stochastic_depth
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  ✓ Model loaded successfully")
    
    # Print metrics if available
    if 'best_balanced_acc' in checkpoint:
        print(f"  Validation balanced accuracy: {checkpoint['best_balanced_acc']:.4f}")
    if 'epoch' in checkpoint:
        print(f"  Trained for {checkpoint['epoch']} epochs")
    
    return model, config


def predict_test_set(checkpoint_path, test_dir, output_csv='predictions.csv', batch_size=32):
    """Generate predictions for test set
    
    Args:
        checkpoint_path: Path to model checkpoint
        test_dir: Directory containing test images
        output_csv: Output CSV file path
        batch_size: Batch size for inference
    
    Returns:
        DataFrame with predictions
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model
    model, config = load_model(checkpoint_path, device)
    
    # Get test images
    test_path = Path(test_dir)
    test_images = sorted(list(test_path.glob('*.jpg')))
    print(f"\nFound {len(test_images)} test images")
    
    if len(test_images) == 0:
        raise ValueError(f"No .jpg images found in {test_dir}")
    
    # Create dataset and loader
    img_size = config.get('image_size', 224)
    test_dataset = TestDataset(
        image_paths=[str(p) for p in test_images],
        transform=get_val_transforms(img_size)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Class mapping (integer -> string label)
    class_names = [
        'Normal mucosa and vascular pattern in the large bowel',  # 0
        'Normal esophagus',                                        # 1
        'Colon polyps',                                            # 2
        'Erythema'                                                 # 3
    ]
    
    # Run inference
    print(f"\nRunning inference...")
    all_predictions = []
    all_filenames = []
    
    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc='Predicting'):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_filenames.extend(filenames)
    
    # Convert predictions to class names
    prediction_labels = [class_names[pred] for pred in all_predictions]
    
    # Create submission DataFrame
    df = pd.DataFrame({
        'image_id': all_filenames,
        'prediction': prediction_labels
    })
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Predictions saved to: {output_csv}")
    
    # Show distribution
    print(f"\nPrediction distribution:")
    for class_name in class_names:
        count = (df['prediction'] == class_name).sum()
        pct = count / len(df) * 100
        print(f"  {class_name}: {count} ({pct:.1f}%)")
    
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate predictions for GastroVision test set')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test-dir', type=str, default='Gastrovision Challenge dataset/Test data',
                        help='Directory containing test images')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output CSV file path')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')
    
    args = parser.parse_args()
    
    predict_test_set(
        checkpoint_path=args.checkpoint,
        test_dir=args.test_dir,
        output_csv=args.output,
        batch_size=args.batch_size
    )

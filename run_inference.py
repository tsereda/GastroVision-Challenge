#!/usr/bin/env python3
"""
Complete Inference Pipeline for GastroVision Challenge
1. Downloads checkpoint from W&B
2. Runs inference on test set
3. Generates submission CSV with exact label format
"""

import os
import torch
import wandb
import subprocess
from pathlib import Path


def download_checkpoint():
    """Download the best checkpoint from W&B"""
    
    print("="*60)
    print("STEP 1: Downloading checkpoint from W&B")
    print("="*60)
    
    # Make sure W&B is logged in
    try:
        wandb.login()
    except:
        print("Please set your W&B API key:")
        print("export WANDB_API_KEY=<your-key>")
        raise
    
    # Download checkpoint from specific run
    api = wandb.Api()
    run_path = "timgsereda/gastrovision-challenge/1n74voi1"
    
    print(f"\nDownloading from run: {run_path}")
    run = api.run(run_path)
    
    # Create checkpoint directory
    checkpoint_dir = Path('./checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Download the checkpoint file
    checkpoint_name = 'best_model_1n74voi1.pth'
    print(f"Looking for: {checkpoint_name}")
    
    # List all files
    files = list(run.files())
    checkpoint_file = None
    
    for f in files:
        if checkpoint_name in f.name:
            checkpoint_file = f
            break
    
    if checkpoint_file is None:
        print("\nAvailable files in run:")
        for f in files:
            print(f"  - {f.name}")
        raise FileNotFoundError(f"Checkpoint {checkpoint_name} not found")
    
    # Download
    checkpoint_path = checkpoint_dir / checkpoint_name
    checkpoint_file.download(root=str(checkpoint_dir), replace=True)
    
    print(f"✓ Downloaded to: {checkpoint_path}")
    
    # Verify file
    if checkpoint_path.exists():
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"  File size: {size_mb:.1f} MB")
    else:
        raise FileNotFoundError(f"Download failed: {checkpoint_path}")
    
    return checkpoint_path


def verify_test_data(test_dir='Gastrovision Challenge dataset/Test data'):
    """Verify test data directory exists"""
    
    print("\n" + "="*60)
    print("STEP 2: Verifying test data")
    print("="*60)
    
    test_path = Path(test_dir)
    
    if not test_path.exists():
        print(f"Error: Test directory not found: {test_dir}")
        print("\nPlease ensure the test data is available at:")
        print(f"  {test_path.absolute()}")
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # Count test images
    test_images = list(test_path.glob('*.jpg'))
    print(f"✓ Found {len(test_images)} test images")
    
    if len(test_images) == 0:
        raise ValueError(f"No .jpg images found in {test_dir}")
    
    return test_path


def run_inference_pipeline(checkpoint_path, test_dir, output_csv='predictions.csv'):
    """Run the inference pipeline"""
    
    print("\n" + "="*60)
    print("STEP 3: Running inference")
    print("="*60)
    
    # Import the prediction function
    from predict import predict_test_set
    
    # Run predictions
    df = predict_test_set(
        checkpoint_path=str(checkpoint_path),
        test_dir=str(test_dir),
        output_csv=output_csv,
        batch_size=32
    )
    
    return df


def verify_submission_format(csv_path='predictions.csv'):
    """Verify the submission CSV has correct format"""
    
    print("\n" + "="*60)
    print("STEP 4: Verifying submission format")
    print("="*60)
    
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    # Check columns
    required_cols = ['image_id', 'prediction']
    if list(df.columns) != required_cols:
        print(f"Error: CSV must have columns: {required_cols}")
        print(f"Found: {list(df.columns)}")
        return False
    
    # Valid class labels (EXACT as specified)
    valid_labels = [
        'Normal mucosa and vascular pattern in the large bowel',
        'Normal esophagus',
        'Colon polyps',
        'Erythema'
    ]
    
    # Check all predictions are valid
    invalid_labels = set(df['prediction']) - set(valid_labels)
    if invalid_labels:
        print(f"Error: Invalid labels found: {invalid_labels}")
        return False
    
    # Check for duplicates
    if df['image_id'].duplicated().any():
        print("Error: Duplicate image IDs found")
        return False
    
    print("✓ Submission format is correct")
    print(f"  - {len(df)} predictions")
    print(f"  - Columns: {list(df.columns)}")
    print(f"  - All labels valid")
    print(f"  - No duplicates")
    
    # Show class distribution
    print(f"\nClass distribution:")
    for label in valid_labels:
        count = (df['prediction'] == label).sum()
        pct = count / len(df) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    return True


def main():
    """Complete pipeline"""
    
    print("\n" + "="*70)
    print(" GASTROVISION CHALLENGE - INFERENCE PIPELINE")
    print("="*70)
    
    try:
        # Step 1: Download checkpoint
        checkpoint_path = download_checkpoint()
        
        # Step 2: Verify test data
        test_dir = verify_test_data()
        
        # Step 3: Run inference
        output_csv = 'predictions.csv'
        df = run_inference_pipeline(checkpoint_path, test_dir, output_csv)
        
        # Step 4: Verify submission format
        if verify_submission_format(output_csv):
            print("\n" + "="*70)
            print(" SUCCESS! Submission file is ready")
            print("="*70)
            print(f"\nSubmission file: {output_csv}")
            print(f"Total predictions: {len(df)}")
            print("\nYou can now submit this file to the competition!")
        else:
            print("\n" + "="*70)
            print(" ERROR: Submission format verification failed")
            print("="*70)
            
    except Exception as e:
        print("\n" + "="*70)
        print(" ERROR")
        print("="*70)
        print(f"\n{str(e)}")
        print("\nPipeline failed. Please check the error message above.")
        raise


if __name__ == '__main__':
    main()
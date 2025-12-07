#!/usr/bin/env python3
"""
Comprehensive W&B Sweep Analysis Script
Analyzes hyperparameter sweep results and provides actionable insights
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path


def load_and_clean_data(csv_path):
    """Load CSV and clean data"""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Total runs: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    
    # Filter to runs with validation metrics
    df_valid = df[df['val/balanced_accuracy'].notna()].copy()
    print(f"Runs with validation metrics: {len(df_valid)}")
    
    return df, df_valid


def print_top_runs(df, n=10):
    """Display top N runs by balanced accuracy"""
    print("\n" + "=" * 100)
    print(f"TOP {n} RUNS BY BALANCED ACCURACY")
    print("=" * 100)
    
    df_sorted = df.sort_values('val/balanced_accuracy', ascending=False)
    
    for i, (idx, row) in enumerate(df_sorted.head(n).iterrows(), 1):
        print(f"\n{i}. {row['Name']} - Val Acc: {row['val/balanced_accuracy']:.4f}")
        print(f"   F1: {row['val/f1_macro']:.4f} | Worst class: {row['val/worst_class_recall']:.4f}")
        
        # Per-class recall
        recalls = [
            f"Mucosa={row['val/recall_normal_mucosa']:.3f}",
            f"Esoph={row['val/recall_normal_esophagus']:.3f}",
            f"Polyps={row['val/recall_polyps']:.3f}",
            f"Eryth={row['val/recall_erythema']:.3f}"
        ]
        print(f"   Recalls: {', '.join(recalls)}")
        
        # Hyperparameters
        print(f"   Config: LR={row['learning_rate']:.5f} WD={row['weight_decay']:.4f} "
              f"gamma={row['focal_gamma']:.2f} boost={row['class_weight_boost']:.1f} "
              f"warmup={int(row['warmup_epochs'])}")


def analyze_hyperparameters(df, top_n=10):
    """Analyze hyperparameter patterns in top runs"""
    print("\n" + "=" * 100)
    print(f"HYPERPARAMETER ANALYSIS (Top {top_n} runs)")
    print("=" * 100)
    
    top_runs = df.sort_values('val/balanced_accuracy', ascending=False).head(top_n)
    
    # Categorical parameters
    print("\n📊 Categorical Parameters:")
    for param in ['scheduler', 'loss_function']:
        if param in top_runs.columns:
            counts = top_runs[param].value_counts()
            print(f"\n{param}:")
            for val, count in counts.items():
                print(f"  {val}: {count}/{top_n}")
    
    # Continuous parameters
    print("\n📈 Continuous Parameters (mean ± std [min, max]):")
    numeric_params = [
        'learning_rate', 'weight_decay', 'focal_gamma', 'class_weight_boost',
        'warmup_epochs', 'label_smoothing', 'dropout_rate', 'stochastic_depth',
        'mixup_alpha', 'cutmix_alpha'
    ]
    
    for param in numeric_params:
        if param in top_runs.columns:
            values = top_runs[param]
            print(f"\n{param}:")
            print(f"  {values.mean():.5f} ± {values.std():.5f} [{values.min():.5f}, {values.max():.5f}]")
            
            # Show value distribution
            value_counts = values.value_counts().sort_index()
            if len(value_counts) <= 10:
                print(f"  Distribution: {dict(value_counts)}")


def analyze_class_imbalance(df):
    """Analyze per-class performance issues"""
    print("\n" + "=" * 100)
    print("PER-CLASS RECALL ANALYSIS")
    print("=" * 100)
    
    class_cols = [
        'val/recall_normal_mucosa',
        'val/recall_normal_esophagus', 
        'val/recall_polyps',
        'val/recall_erythema'
    ]
    
    class_names = ['Normal Mucosa', 'Normal Esophagus', 'Polyps', 'Erythema']
    
    print("\n📊 Average recall across all runs:")
    for name, col in zip(class_names, class_cols):
        avg = df[col].mean()
        std = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        print(f"  {name:20s}: {avg:.3f} ± {std:.3f} [{min_val:.3f}, {max_val:.3f}]")
    
    # Top 10 runs
    top_10 = df.sort_values('val/balanced_accuracy', ascending=False).head(10)
    print("\n📊 Average recall in top 10 runs:")
    for name, col in zip(class_names, class_cols):
        avg = top_10[col].mean()
        print(f"  {name:20s}: {avg:.3f}")
    
    # Identify problematic class
    avg_recalls = [df[col].mean() for col in class_cols]
    worst_idx = np.argmin(avg_recalls)
    print(f"\n⚠️  Weakest class: {class_names[worst_idx]} (avg recall: {avg_recalls[worst_idx]:.3f})")


def correlation_analysis(df, target='val/balanced_accuracy'):
    """Analyze which hyperparameters correlate with performance"""
    print("\n" + "=" * 100)
    print(f"CORRELATION WITH {target.upper()}")
    print("=" * 100)
    
    numeric_params = [
        'learning_rate', 'weight_decay', 'focal_gamma', 'class_weight_boost',
        'warmup_epochs', 'label_smoothing', 'dropout_rate', 'stochastic_depth',
        'mixup_alpha', 'cutmix_alpha'
    ]
    
    correlations = []
    for param in numeric_params:
        if param in df.columns:
            corr = df[param].corr(df[target])
            if not np.isnan(corr):
                correlations.append((param, corr))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\nCorrelations (sorted by absolute value):")
    for param, corr in correlations:
        direction = "↑" if corr > 0 else "↓"
        print(f"  {param:25s}: {corr:+.3f} {direction}")


def generate_recommendations(df):
    """Generate actionable recommendations"""
    print("\n" + "=" * 100)
    print("🎯 RECOMMENDATIONS")
    print("=" * 100)
    
    best_run = df.sort_values('val/balanced_accuracy', ascending=False).iloc[0]
    top_5 = df.sort_values('val/balanced_accuracy', ascending=False).head(5)
    
    print("\n1. BEST SINGLE CONFIGURATION:")
    print(f"   Run: {best_run['Name']}")
    print(f"   Val Acc: {best_run['val/balanced_accuracy']:.4f}")
    print(f"\n   Hyperparameters:")
    print(f"     learning_rate: {best_run['learning_rate']:.5f}")
    print(f"     weight_decay: {best_run['weight_decay']:.4f}")
    print(f"     focal_gamma: {best_run['focal_gamma']:.2f}")
    print(f"     class_weight_boost: {best_run['class_weight_boost']:.1f}")
    print(f"     warmup_epochs: {int(best_run['warmup_epochs'])}")
    print(f"     label_smoothing: {best_run['label_smoothing']:.3f}")
    print(f"     dropout: {best_run['dropout_rate']:.2f}")
    print(f"     stochastic_depth: {best_run['stochastic_depth']:.2f}")
    print(f"     mixup_alpha: {best_run['mixup_alpha']:.2f}")
    print(f"     cutmix_alpha: {best_run['cutmix_alpha']:.1f}")
    
    print("\n2. STABLE HYPERPARAMETER RANGES (from top 5):")
    for param in ['learning_rate', 'focal_gamma', 'class_weight_boost', 'warmup_epochs']:
        if param in top_5.columns:
            print(f"   {param}: [{top_5[param].min():.5f}, {top_5[param].max():.5f}]")
    
    print("\n3. ENSEMBLE CANDIDATES:")
    print("   Consider ensembling these top runs for maximum performance:")
    for i, (idx, row) in enumerate(df.sort_values('val/balanced_accuracy', ascending=False).head(5).iterrows(), 1):
        print(f"   {i}. {row['Name']} (Acc: {row['val/balanced_accuracy']:.4f})")
    
    # Class-specific recommendations
    class_cols = [
        'val/recall_normal_mucosa',
        'val/recall_normal_esophagus', 
        'val/recall_polyps',
        'val/recall_erythema'
    ]
    class_names = ['Normal Mucosa', 'Normal Esophagus', 'Polyps', 'Erythema']
    
    avg_recalls = [top_5[col].mean() for col in class_cols]
    worst_idx = np.argmin(avg_recalls)
    
    print(f"\n4. ADDRESS WEAK CLASS ({class_names[worst_idx]}):")
    print(f"   Current recall: {avg_recalls[worst_idx]:.3f}")
    print(f"   Suggestions:")
    print(f"   - Increase class_weight_boost further")
    print(f"   - Train class-specific classifier")
    print(f"   - Investigate data quality for this class")
    print(f"   - Add class-specific augmentations")


def export_best_config(df, output_path):
    """Export best configuration as YAML"""
    best = df.sort_values('val/balanced_accuracy', ascending=False).iloc[0]
    
    yaml_content = f"""# Best configuration from sweep
# Run: {best['Name']}
# Val Balanced Accuracy: {best['val/balanced_accuracy']:.4f}

model_name: {best['model_name']}
image_size: {int(best['image_size'])}

# Optimizer
learning_rate: {best['learning_rate']:.5f}
weight_decay: {best['weight_decay']:.5f}
warmup_epochs: {int(best['warmup_epochs'])}
scheduler: {best['scheduler']}

# Loss
loss_function: {best['loss_function']}
focal_gamma: {best['focal_gamma']:.2f}
label_smoothing: {best['label_smoothing']:.3f}
class_weight_boost: {best['class_weight_boost']:.1f}

# Regularization
dropout_rate: {best['dropout_rate']:.2f}
stochastic_depth: {best['stochastic_depth']:.2f}

# Augmentation
mixup_alpha: {best['mixup_alpha']:.2f}
cutmix_alpha: {best['cutmix_alpha']:.1f}
mixup_prob: {best['mixup_prob']:.2f}
color_jitter_brightness: {best['color_jitter_brightness']:.2f}
color_jitter_contrast: {best['color_jitter_contrast']:.2f}
color_jitter_saturation: {best['color_jitter_saturation']:.2f}

# Training
batch_size: {int(best['batch_size'])}
epochs: 60  # Increase from sweep value for final training
"""
    
    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Best configuration exported to: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_wandb_sweep.py <path_to_csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    # Load data
    df_full, df_valid = load_and_clean_data(csv_path)
    
    # Run analyses
    print_top_runs(df_valid, n=10)
    analyze_hyperparameters(df_valid, top_n=10)
    analyze_class_imbalance(df_valid)
    correlation_analysis(df_valid)
    generate_recommendations(df_valid)
    
    # Export best config
    output_path = 'best_config.yaml'  # Output to current directory
    export_best_config(df_valid, output_path)
    
    print("\n" + "=" * 100)
    print("Analysis complete!")
    print("=" * 100)


if __name__ == '__main__':
    main()
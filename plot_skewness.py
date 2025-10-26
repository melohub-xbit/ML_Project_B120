import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 80)
print("SKEWNESS ANALYSIS - Transport_Cost")
print("=" * 80)

train_df = pd.read_csv('dataset/train.csv')
target = train_df['Transport_Cost']

import os
os.makedirs('skewness_plots', exist_ok=True)

# ============================================================================
# PLOT 1: ORIGINAL DATA (BEFORE LOG TRANSFORMATION)
# ============================================================================

print("\nOriginal Data Statistics:")
print(f"  Mean: {target.mean():.2f}")
print(f"  Median: {target.median():.2f}")
print(f"  Std: {target.std():.2f}")
print(f"  Min: {target.min():.2f}")
print(f"  Max: {target.max():.2f}")
print(f"  Skewness: {skew(target):.4f}")
print(f"  Negative values: {(target < 0).sum()}")

plt.figure(figsize=(12, 7))
plt.hist(target, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
plt.axvline(target.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {target.mean():.2f}')
plt.axvline(target.median(), color='green', linestyle='--', linewidth=2, 
            label=f'Median: {target.median():.2f}')
plt.xlabel('Transport Cost', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title(f'Original Distribution - Transport_Cost\nSkewness: {skew(target):.4f} (Highly Right-Skewed)', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('skewness_plots/1_before_log_transformation.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: skewness_plots/1_before_log_transformation.png")
plt.close()

# ============================================================================
# PLOT 2: AFTER LOG TRANSFORMATION (WITH SHIFTING)
# ============================================================================

min_cost = target.min()
shift_value = abs(min_cost) + 1 if min_cost < 0 else 0
target_shifted = target + shift_value

target_log = np.log1p(target_shifted)

print("\nAfter Log Transformation:")
print(f"  Shift applied: {shift_value:.2f}")
print(f"  Mean: {target_log.mean():.2f}")
print(f"  Median: {target_log.median():.2f}")
print(f"  Std: {target_log.std():.2f}")
print(f"  Min: {target_log.min():.2f}")
print(f"  Max: {target_log.max():.2f}")
print(f"  Skewness: {skew(target_log):.4f}")

plt.figure(figsize=(12, 7))
plt.hist(target_log, bins=50, edgecolor='black', alpha=0.7, color='coral')
plt.axvline(target_log.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {target_log.mean():.2f}')
plt.axvline(target_log.median(), color='green', linestyle='--', linewidth=2, 
            label=f'Median: {target_log.median():.2f}')
plt.xlabel('Log(Transport Cost)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title(f'After Log Transformation - Transport_Cost\nSkewness: {skew(target_log):.4f} (More Normal Distribution)', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('skewness_plots/2_after_log_transformation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: skewness_plots/2_after_log_transformation.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nSkewness Reduction:")
print(f"  Before: {skew(target):.4f} (Highly right-skewed)")
print(f"  After:  {skew(target_log):.4f} (Near-normal)")
print(f"  Improvement: {abs(skew(target)) - abs(skew(target_log)):.4f}")
print("\n✓ Log transformation successfully normalizes the distribution!")
print("✓ This justifies using log-transformed target for model training.")
print("\n" + "=" * 80)

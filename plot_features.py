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
print("FEATURE ANALYSIS - Distributions & Correlations")
print("=" * 80)

train_df = pd.read_csv('dataset/train.csv')

import os
os.makedirs('feature_plots', exist_ok=True)

# ============================================================================
# CORRELATION WITH TARGET PLOT
# ============================================================================

print("\nCreating Correlation with Target Plot...")

numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

numerical_cols = [col for col in numerical_cols if col not in ['Hospital_Id', 'Transport_Cost']]

correlations = {}
for col in numerical_cols:
    valid_data = train_df[[col, 'Transport_Cost']].dropna()
    if len(valid_data) > 0:
        corr = valid_data[col].corr(valid_data['Transport_Cost'])
        correlations[col] = corr

corr_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Correlation'])
corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
corr_df = corr_df.sort_values('Correlation', ascending=True)

print("\nCorrelations with Transport_Cost:")
for _, row in corr_df.iterrows():
    print(f"  {row['Feature']}: {row['Correlation']:.4f}")

fig, ax = plt.subplots(figsize=(12, 8))

colors = []
for corr in corr_df['Correlation']:
    if abs(corr) > 0.3:
        colors.append('darkgreen')
    elif abs(corr) > 0.15:
        colors.append('orange')
    else:
        colors.append('lightcoral')

bars = ax.barh(corr_df['Feature'], corr_df['Correlation'], color=colors, edgecolor='black', alpha=0.8)

ax.axvline(0, color='black', linestyle='-', linewidth=1)

ax.axvline(0.3, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Strong (>0.3)')
ax.axvline(-0.3, color='green', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(0.15, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Moderate (>0.15)')
ax.axvline(-0.15, color='orange', linestyle='--', linewidth=1, alpha=0.5)

for i, (feature, corr) in enumerate(zip(corr_df['Feature'], corr_df['Correlation'])):
    ax.text(corr + (0.02 if corr > 0 else -0.02), i, f'{corr:.3f}', 
           va='center', ha='left' if corr > 0 else 'right', fontsize=9, fontweight='bold')

ax.set_xlabel('Correlation with Transport_Cost', fontsize=12, fontweight='bold')
ax.set_ylabel('Features', fontsize=12, fontweight='bold')
ax.set_title('Feature Correlations with Transport_Cost\n(Green: Strong predictors | Orange: Moderate | Red: Weak)', 
            fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('feature_plots/2_correlation_with_target.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: feature_plots/2_correlation_with_target.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY - CORRELATION ANALYSIS")
print("=" * 80)

print("\nTop 3 Strongest Positive Correlations:")
top_3 = corr_df.nlargest(3, 'Correlation')
for _, row in top_3.iterrows():
    print(f"  • {row['Feature']}: {row['Correlation']:.4f}")

print("\nTop 3 Weakest Correlations:")
bottom_3 = corr_df.nsmallest(3, 'Correlation')
for _, row in bottom_3.iterrows():
    print(f"  • {row['Feature']}: {row['Correlation']:.4f}")

print("\n" + "=" * 80)

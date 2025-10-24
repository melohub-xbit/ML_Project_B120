"""
Analyze and visualize ensemble weight performance.

This script analyzes your manual ensemble experiments and creates visualizations
to help identify the optimal weight combinations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

print("="*80)
print("ENSEMBLE WEIGHT ANALYSIS")
print("="*80)

# ============================================================================
# YOUR EXPERIMENTAL RESULTS
# ============================================================================

results = {
    'CB': [30, 25, 23, 20, 25, 20, 0],
    'XGB': [0, 5, 5, 5, 10, 10, 0],
    'RF': [70, 70, 72, 75, 65, 70, 100],
    'Score': [4034331886.602, 4045245673.109, 4054448945.049, 4069477868.271, 
              4076639179.598, 4095796103.876, 4219143131.733],
    'Label': ['30/0/70 ⭐BEST', '25/5/70', '23/5/72', '20/5/75', 
              '25/10/65', '20/10/70 (baseline)', '0/0/100 (pure RF)']
}

df = pd.DataFrame(results)
df = df.sort_values('Score')  # Sort by score (lower is better)

print("\n📊 ALL RESULTS (Sorted by Performance):")
print("="*80)
for idx, row in df.iterrows():
    rank = df.index.get_loc(idx) + 1
    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
    print(f"{medal} {row['Label']:25s} | Score: {row['Score']:,.0f} | CB:{row['CB']:3d}% XGB:{row['XGB']:3d}% RF:{row['RF']:3d}%")

# Calculate improvement over baseline
baseline_score = df[df['Label'].str.contains('baseline')]['Score'].values[0]
df['Improvement_vs_Baseline'] = ((baseline_score - df['Score']) / baseline_score * 100)
df['Improvement_vs_Best'] = ((df['Score'].min() - df['Score']) / df['Score'].min() * 100)

print(f"\n📈 IMPROVEMENT ANALYSIS:")
print("="*80)
best_score = df['Score'].min()
best_config = df[df['Score'] == best_score].iloc[0]
print(f"Best Configuration: {best_config['Label']}")
print(f"Best Score: {best_score:,.0f}")
print(f"Improvement over baseline: {best_config['Improvement_vs_Baseline']:.3f}%")
print(f"Improvement over pure RF: {((results['Score'][-1] - best_score) / results['Score'][-1] * 100):.3f}%")

# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================

fig = plt.figure(figsize=(18, 12))

# Plot 1: Bar chart of all configurations
ax1 = plt.subplot(2, 3, 1)
colors = ['#2ecc71' if i == 0 else '#3498db' if i < 3 else '#e74c3c' for i in range(len(df))]
bars = ax1.barh(range(len(df)), df['Score'], color=colors)
ax1.set_yticks(range(len(df)))
ax1.set_yticklabels(df['Label'])
ax1.set_xlabel('Score (Lower is Better)', fontsize=11, fontweight='bold')
ax1.set_title('Performance Comparison\n(Green=Best, Blue=Top3, Red=Others)', fontsize=12, fontweight='bold')
ax1.invert_yaxis()
for i, (score, improvement) in enumerate(zip(df['Score'], df['Improvement_vs_Baseline'])):
    ax1.text(score, i, f' {improvement:+.2f}%', va='center', fontsize=9)
ax1.axvline(baseline_score, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')
ax1.legend()

# Plot 2: Impact of each model weight
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(df['CB'], df['Score'], s=200, alpha=0.6, label='CatBoost %', color='#e67e22')
ax2.scatter(df['XGB'], df['Score'], s=200, alpha=0.6, label='XGBoost %', color='#9b59b6')
ax2.scatter(df['RF'], df['Score'], s=200, alpha=0.6, label='Random Forest %', color='#27ae60')
ax2.set_xlabel('Weight (%)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Score (Lower is Better)', fontsize=11, fontweight='bold')
ax2.set_title('Individual Weight Impact', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: RF% vs Score (key relationship)
ax3 = plt.subplot(2, 3, 3)
# Exclude pure RF for better trend visibility
df_no_pure_rf = df[df['RF'] < 100]
ax3.scatter(df_no_pure_rf['RF'], df_no_pure_rf['Score'], s=200, alpha=0.7, color='#27ae60')
for idx, row in df_no_pure_rf.iterrows():
    ax3.annotate(f"{int(row['CB'])}/{int(row['XGB'])}/{int(row['RF'])}", 
                xy=(row['RF'], row['Score']), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)
# Add trend line
z = np.polyfit(df_no_pure_rf['RF'], df_no_pure_rf['Score'], 2)
p = np.poly1d(z)
rf_range = np.linspace(df_no_pure_rf['RF'].min(), df_no_pure_rf['RF'].max(), 100)
ax3.plot(rf_range, p(rf_range), "r--", alpha=0.5, linewidth=2, label='Trend (quadratic)')
ax3.set_xlabel('Random Forest Weight (%)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Score (Lower is Better)', fontsize=11, fontweight='bold')
ax3.set_title('RF Weight vs Performance\n(Optimal around 70%)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: CB vs XGB contribution (when RF is fixed)
ax4 = plt.subplot(2, 3, 4)
df_rf70 = df[df['RF'] == 70]
if len(df_rf70) > 0:
    ax4.scatter(df_rf70['CB'], df_rf70['XGB'], s=df_rf70['Score']/1e7, alpha=0.6, c=df_rf70['Score'], cmap='RdYlGn_r')
    for idx, row in df_rf70.iterrows():
        ax4.annotate(f"{row['Label']}\n{row['Score']/1e9:.3f}B", 
                    xy=(row['CB'], row['XGB']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax4.set_xlabel('CatBoost Weight (%)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('XGBoost Weight (%)', fontsize=11, fontweight='bold')
    ax4.set_title('CB vs XGB Trade-off (RF=70%)\nBubble size = Score', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

# Plot 5: Improvement over baseline
ax5 = plt.subplot(2, 3, 5)
colors_improvement = ['#2ecc71' if x > 0 else '#e74c3c' for x in df['Improvement_vs_Baseline']]
bars = ax5.barh(range(len(df)), df['Improvement_vs_Baseline'], color=colors_improvement)
ax5.set_yticks(range(len(df)))
ax5.set_yticklabels(df['Label'])
ax5.set_xlabel('Improvement over Baseline (%)', fontsize=11, fontweight='bold')
ax5.set_title('Relative Performance Gain\n(Green=Better, Red=Worse)', fontsize=12, fontweight='bold')
ax5.axvline(0, color='black', linestyle='-', linewidth=1)
ax5.invert_yaxis()
for i, val in enumerate(df['Improvement_vs_Baseline']):
    ax5.text(val, i, f' {val:+.2f}%', va='center', fontsize=9)

# Plot 6: 3D scatter (CB, XGB, RF vs Score)
ax6 = plt.subplot(2, 3, 6, projection='3d')
df_no_pure = df[df['RF'] < 100]  # Exclude pure RF for better visualization
scatter = ax6.scatter(df_no_pure['CB'], df_no_pure['XGB'], df_no_pure['RF'], 
                     c=df_no_pure['Score'], s=200, cmap='RdYlGn_r', alpha=0.7)
ax6.set_xlabel('CatBoost %', fontsize=10, fontweight='bold')
ax6.set_ylabel('XGBoost %', fontsize=10, fontweight='bold')
ax6.set_zlabel('Random Forest %', fontsize=10, fontweight='bold')
ax6.set_title('3D Weight Space\n(Color=Score, Green=Better)', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax6, label='Score', pad=0.1)

plt.tight_layout()
plt.savefig('ensemble_weight_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: ensemble_weight_analysis.png")

# ============================================================================
# DETAILED ANALYSIS & RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("🔍 DETAILED ANALYSIS")
print("="*80)

print("\n1️⃣ KEY FINDING: CatBoost Contribution")
print("-"*80)
df_sorted_cb = df[df['RF'] == 70].sort_values('Score')
if len(df_sorted_cb) > 0:
    print(f"When RF=70%, performance improves as CB increases:")
    for _, row in df_sorted_cb.iterrows():
        print(f"  CB={row['CB']:2d}%, XGB={row['XGB']:2d}% → Score: {row['Score']:,.0f}")
    print(f"\n✓ Conclusion: Higher CB weight is better (30% > 25% > 20%)")

print("\n2️⃣ KEY FINDING: XGBoost Contribution")
print("-"*80)
print(f"Comparing XGB impact:")
print(f"  30% CB + 0% XGB + 70% RF = {df[(df['CB']==30) & (df['XGB']==0)]['Score'].values[0]:,.0f} ⭐BEST")
print(f"  25% CB + 5% XGB + 70% RF = {df[(df['CB']==25) & (df['XGB']==5) & (df['RF']==70)]['Score'].values[0]:,.0f}")
print(f"  20% CB + 10% XGB + 70% RF = {df[(df['CB']==20) & (df['XGB']==10)]['Score'].values[0]:,.0f}")
print(f"\n✓ Conclusion: XGBoost HURTS performance! Best without it (0% XGB)")

print("\n3️⃣ KEY FINDING: Random Forest Optimal Range")
print("-"*80)
rf_scores = df[df['RF'] <= 100].sort_values('RF')[['RF', 'Score']]
print(f"RF percentage vs Score:")
for _, row in rf_scores.iterrows():
    marker = " ⭐" if row['Score'] == df['Score'].min() else ""
    print(f"  RF={int(row['RF']):3d}% → Score: {row['Score']:,.0f}{marker}")
print(f"\n✓ Conclusion: Optimal RF is around 70% (not 65%, not 75%)")

print("\n4️⃣ ENSEMBLE VALUE")
print("-"*80)
pure_rf_score = df[df['RF'] == 100]['Score'].values[0]
best_ensemble = df[df['Score'] == df['Score'].min()].iloc[0]
improvement = ((pure_rf_score - best_ensemble['Score']) / pure_rf_score * 100)
print(f"Pure RF score:     {pure_rf_score:,.0f}")
print(f"Best ensemble:     {best_ensemble['Score']:,.0f}")
print(f"Improvement:       {improvement:.3f}%")
print(f"\n✓ Conclusion: Ensemble adds {improvement:.3f}% value over single model")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("🎯 RECOMMENDATIONS")
print("="*80)

print("\n📌 FINDING: Your optimal weights are 30% CB + 0% XGB + 70% RF")
print("\nWhy this works:")
print("  1. RF is your strongest model → needs high weight (70%)")
print("  2. CatBoost adds complementary value → needs moderate weight (30%)")
print("  3. XGBoost is redundant/harmful → should be excluded (0%)")

print("\n🚀 NEXT EXPERIMENTS TO TRY:")
print("-"*80)

# Find the pattern: increase CB, keep RF at 70%
next_experiments = [
    (32, 0, 68, "Slightly more CB, slightly less RF"),
    (35, 0, 65, "More CB shift from RF"),
    (28, 0, 72, "Slightly less CB, slightly more RF"),
    (30, 0, 70, "Re-test current best for confirmation"),
    (40, 0, 60, "Aggressive CB increase (test boundary)"),
]

print("Try these weight combinations (in order of priority):\n")
for i, (cb, xgb, rf, reason) in enumerate(next_experiments, 1):
    print(f"{i}. CB={cb}%, XGB={xgb}%, RF={rf}%")
    print(f"   Reason: {reason}\n")

print("="*80)
print("💡 STRATEGY:")
print("="*80)
print("Based on your results, the pattern is clear:")
print("  ✓ Keep XGB at 0% (it only hurts performance)")
print("  ✓ Find optimal CB/RF ratio (currently 30/70 is best)")
print("  ✓ Test slight variations: 32/68, 35/65, 28/72")
print("  ✓ Expected optimal range: 28-35% CB, 65-72% RF, 0% XGB")
print("="*80)

# Create a heatmap for CB vs RF (XGB=0)
print("\n📊 Creating CB vs RF heatmap...")
fig2, ax = plt.subplots(figsize=(10, 8))

# Create interpolated heatmap
from scipy.interpolate import griddata

# Filter data where XGB is small
df_low_xgb = df[df['XGB'] <= 10].copy()

# Create grid
cb_range = np.linspace(df_low_xgb['CB'].min(), df_low_xgb['CB'].max(), 50)
rf_range = np.linspace(df_low_xgb['RF'].min(), df_low_xgb['RF'].max(), 50)
cb_grid, rf_grid = np.meshgrid(cb_range, rf_range)

# Interpolate scores
points = df_low_xgb[['CB', 'RF']].values
values = df_low_xgb['Score'].values
score_grid = griddata(points, values, (cb_grid, rf_grid), method='cubic')

# Plot
contour = ax.contourf(cb_grid, rf_grid, score_grid, levels=20, cmap='RdYlGn_r')
plt.colorbar(contour, ax=ax, label='Score (Lower is Better)')

# Mark actual experiments
ax.scatter(df_low_xgb['CB'], df_low_xgb['RF'], s=200, c='blue', marker='o', 
          edgecolors='black', linewidths=2, zorder=5, label='Tested')

# Mark best
best = df_low_xgb[df_low_xgb['Score'] == df_low_xgb['Score'].min()].iloc[0]
ax.scatter(best['CB'], best['RF'], s=400, c='gold', marker='*', 
          edgecolors='black', linewidths=2, zorder=6, label='Best (30/0/70)')

# Annotate experiments
for _, row in df_low_xgb.iterrows():
    ax.annotate(f"{int(row['CB'])}/{int(row['XGB'])}/{int(row['RF'])}", 
               xy=(row['CB'], row['RF']), 
               xytext=(5, 5), textcoords='offset points', 
               fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax.set_xlabel('CatBoost Weight (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Random Forest Weight (%)', fontsize=12, fontweight='bold')
ax.set_title('Performance Heatmap: CB vs RF (XGB ≤ 10%)\nGreen=Better, Red=Worse', 
            fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('ensemble_heatmap_cb_rf.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: ensemble_heatmap_cb_rf.png")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. ensemble_weight_analysis.png - Comprehensive 6-plot analysis")
print("  2. ensemble_heatmap_cb_rf.png - CB vs RF performance landscape")
print("\nNext step: Test the recommended weight combinations above!")
print("="*80)

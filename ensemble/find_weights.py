"""
Find optimal ensemble weights using Ridge regression (stacking).

This script uses the OOF predictions from all three models to find the optimal
weights that minimize RMSE on the validation set.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import sys

print("="*80)
print("FINDING OPTIMAL ENSEMBLE WEIGHTS")
print("="*80)

# ============================================================================
# STEP 1: LOAD OOF PREDICTIONS
# ============================================================================

print("\n[1/3] Loading OOF predictions...")

try:
    oof_cb_preds = np.load('../exp_catboost_3_v2/oof_cb_preds.npy')
    oof_cb_true = np.load('../exp_catboost_3_v2/oof_cb_true.npy')
    oof_xgb_preds = np.load('oof_xgb_preds.npy')
    oof_rf_preds = np.load('oof_rf_preds.npy')
    
    print(f"  ✓ CatBoost OOF:      {oof_cb_preds.shape}")
    print(f"  ✓ XGBoost OOF:       {oof_xgb_preds.shape}")
    print(f"  ✓ Random Forest OOF: {oof_rf_preds.shape}")
    print(f"  ✓ True values:       {oof_cb_true.shape}")
    
except FileNotFoundError as e:
    print(f"  ✗ ERROR: {e}")
    print("\nPlease run get_oof_preds.py first!")
    sys.exit(1)

# Verify all shapes match
assert oof_cb_preds.shape == oof_xgb_preds.shape == oof_rf_preds.shape == oof_cb_true.shape
print("  ✓ All shapes match")

# ============================================================================
# STEP 2: CALCULATE BASELINE PERFORMANCE
# ============================================================================

print("\n[2/3] Calculating baseline performance...")

# Individual model performance
rmse_cb = np.sqrt(mean_squared_error(oof_cb_true, oof_cb_preds))
rmse_xgb = np.sqrt(mean_squared_error(oof_cb_true, oof_xgb_preds))
rmse_rf = np.sqrt(mean_squared_error(oof_cb_true, oof_rf_preds))

print(f"\nIndividual model RMSE (on validation set, log scale):")
print(f"  CatBoost:      {rmse_cb:.6f}")
print(f"  XGBoost:       {rmse_xgb:.6f}")
print(f"  Random Forest: {rmse_rf:.6f}")

# Simple average ensemble (baseline)
simple_avg = (oof_cb_preds + oof_xgb_preds + oof_rf_preds) / 3
rmse_simple = np.sqrt(mean_squared_error(oof_cb_true, simple_avg))
print(f"\n  Simple Average (1/3, 1/3, 1/3): {rmse_simple:.6f}")

# ============================================================================
# STEP 3: FIND OPTIMAL WEIGHTS USING RIDGE REGRESSION
# ============================================================================

print("\n[3/3] Finding optimal weights using Ridge regression...")

# Create meta-features matrix (each column is predictions from one model)
X_meta = np.column_stack([oof_cb_preds, oof_xgb_preds, oof_rf_preds])
y_true = oof_cb_true

print(f"\n  Meta-features shape: {X_meta.shape}")
print(f"  Target shape: {y_true.shape}")

# Try different alpha values to find the best regularization
best_alpha = None
best_weights = None
best_rmse = float('inf')

alphas_to_try = [0.001, 0.01, 0.1, 1.0, 10.0]

print("\n  Testing different regularization strengths (alpha):")
for alpha in alphas_to_try:
    # Fit Ridge regression with positive constraint (all weights must be >= 0)
    meta_model = Ridge(alpha=alpha, positive=True, fit_intercept=False)
    meta_model.fit(X_meta, y_true)
    
    # Get weights and normalize to sum to 1
    weights = meta_model.coef_
    weights = weights / weights.sum()
    
    # Calculate RMSE with these weights
    ensemble_preds = X_meta @ weights
    rmse = np.sqrt(mean_squared_error(y_true, ensemble_preds))
    
    print(f"    alpha={alpha:6.3f}: RMSE={rmse:.6f}, weights=[{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}]")
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_weights = weights
        best_alpha = alpha

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "="*80)
print("OPTIMAL WEIGHTS FOUND")
print("="*80)

print(f"\nBest alpha (regularization): {best_alpha}")
print(f"Best validation RMSE: {best_rmse:.6f}")

print(f"\nOptimal Ensemble Weights:")
print(f"  CatBoost:      {best_weights[0]:.4f}  ({best_weights[0]*100:.1f}%)")
print(f"  XGBoost:       {best_weights[1]:.4f}  ({best_weights[1]*100:.1f}%)")
print(f"  Random Forest: {best_weights[2]:.4f}  ({best_weights[2]*100:.1f}%)")
print(f"  Sum:           {best_weights.sum():.4f}")

# Improvement metrics
print(f"\n" + "-"*80)
print("PERFORMANCE COMPARISON")
print("-"*80)

print(f"\nBest Individual Model:")
best_single = min(rmse_cb, rmse_xgb, rmse_rf)
best_single_name = ["CatBoost", "XGBoost", "Random Forest"][np.argmin([rmse_cb, rmse_xgb, rmse_rf])]
print(f"  {best_single_name}: {best_single:.6f}")

print(f"\nSimple Average Ensemble:")
print(f"  RMSE: {rmse_simple:.6f}")
print(f"  Improvement over best single: {((best_single - rmse_simple) / best_single * 100):+.2f}%")

print(f"\nOptimized Weighted Ensemble:")
print(f"  RMSE: {best_rmse:.6f}")
print(f"  Improvement over best single: {((best_single - best_rmse) / best_single * 100):+.2f}%")
print(f"  Improvement over simple average: {((rmse_simple - best_rmse) / rmse_simple * 100):+.2f}%")

# Save the optimal weights
np.save('optimal_weights.npy', best_weights)
print(f"\n✓ Saved optimal weights to: optimal_weights.npy")

# Also save as text file for easy reference
with open('optimal_weights.txt', 'w') as f:
    f.write("OPTIMAL ENSEMBLE WEIGHTS\n")
    f.write("="*50 + "\n\n")
    f.write(f"Best Regularization (alpha): {best_alpha}\n")
    f.write(f"Validation RMSE: {best_rmse:.6f}\n\n")
    f.write("Weights:\n")
    f.write(f"  W_CB  = {best_weights[0]:.6f}  ({best_weights[0]*100:.2f}%)\n")
    f.write(f"  W_XGB = {best_weights[1]:.6f}  ({best_weights[1]*100:.2f}%)\n")
    f.write(f"  W_RF  = {best_weights[2]:.6f}  ({best_weights[2]*100:.2f}%)\n")
    f.write(f"\nSum: {best_weights.sum():.6f}\n")

print(f"✓ Saved optimal weights to: optimal_weights.txt")

print("\n" + "="*80)
print("WEIGHTS OPTIMIZATION COMPLETE!")
print("="*80)
print("\nNext step: Run ensemble.py to generate final predictions")
print("="*80)

"""
Generate ensemble predictions using MANUAL weights (not optimized).

This script allows you to manually specify the weights for each model
instead of using the optimal weights from Ridge regression.

CUSTOMIZE WEIGHTS HERE:
"""

import pandas as pd
import numpy as np
import sys
import os

print("="*80)
print("GENERATING ENSEMBLE PREDICTIONS (MANUAL WEIGHTS)")
print("="*80)

# ============================================================================
# MANUAL WEIGHT CONFIGURATION - CHANGE THESE VALUES!
# ============================================================================

print("\n[CONFIGURATION] Manual Weights")
print("-"*80)

# =============================================================================
# BASED ON YOUR FINDINGS: RF dominates! (20/10/70 is your best so far)
# Performance: 20/10/70 > 15/15/70 ~ 10/10/80 > 5/5/90
# 
# Strategy: Fine-tune around 65-75% RF with optimized CB/XGB ratios
# Key insight: CB contributes more than XGB when RF is high
# =============================================================================

# 🏆 PRIORITY 1: Increase CB, decrease XGB (keep RF at 70%)
# W_CB = 0.25   # 25% CatBoost (up from 20%)
# W_XGB = 0.05  # 5% XGBoost (down from 10%)
# W_RF = 0.70   # 70% Random Forest (your sweet spot)

# 🥈 PRIORITY 2: Try 75% RF with 20/5 split
# W_CB = 0.20
# W_XGB = 0.05
# W_RF = 0.75

# 🥉 PRIORITY 3: Try 65% RF with more CB
# W_CB = 0.25
# W_XGB = 0.10
# W_RF = 0.65

# Option 4: Test 72% RF (between your best and 75%)
# W_CB = 0.23
# W_XGB = 0.05
# W_RF = 0.72

# Option 5: Pure RF test (control experiment)
# W_CB = 0.00
# W_XGB = 0.00
# W_RF = 1.00



# Option 6: Remove XGB entirely (test if it helps)
# W_CB = 0.26
# W_XGB = 0.00
# W_RF = 0.74

# BASELINE: Your current best (for comparison)
# W_CB = 0.20
# W_XGB = 0.10
# W_RF = 0.70

####################################################################################
############################################################################
## This is best ensemble so far - option 65 cb, 35 rf
########################################
##################################################
# BEST ENSEMBLE: 65% CB, 0% XGB, 35% RF
# W_CB = 0.65
# W_XGB = 0.00
# W_RF = 0.35

### Trying equal for all 3
W_CB = 0.645
W_XGB = 0.01
W_RF = 0.345

print(f"  CatBoost:      {W_CB:.3f} ({W_CB*100:.1f}%)")
print(f"  XGBoost:       {W_XGB:.3f} ({W_XGB*100:.1f}%)")
print(f"  Random Forest: {W_RF:.3f} ({W_RF*100:.1f}%)")
print(f"  Sum:           {W_CB + W_XGB + W_RF:.3f}")
print("-"*80)

# Verify weights sum to 1.0
weight_sum = W_CB + W_XGB + W_RF
if not np.isclose(weight_sum, 1.0):
    print(f"\n⚠️  WARNING: Weights sum to {weight_sum:.6f}, not 1.0!")
    print("  Normalizing weights...")
    W_CB = W_CB / weight_sum
    W_XGB = W_XGB / weight_sum
    W_RF = W_RF / weight_sum
    print(f"\n  Normalized weights:")
    print(f"    CatBoost:      {W_CB:.6f} ({W_CB*100:.2f}%)")
    print(f"    XGBoost:       {W_XGB:.6f} ({W_XGB*100:.2f}%)")
    print(f"    Random Forest: {W_RF:.6f} ({W_RF*100:.2f}%)")

# ============================================================================
# STEP 1: LOAD LOG-SCALE PREDICTIONS FROM ALL MODELS
# ============================================================================

print("\n[1/3] Loading log-scale predictions from each model...")

try:
    log_preds_cb = np.load('../exp_catboost_3_v2/log_preds_cb.npy')
    print(f"  ✓ CatBoost:      {log_preds_cb.shape}")
except FileNotFoundError:
    print("  ✗ ERROR: log_preds_cb.npy not found!")
    print("  Please run: cd exp_catboost_3_v2 && python train_and_predict_cb.py")
    sys.exit(1)

try:
    log_preds_xgb = np.load('../tune_xgboost_v2/log_preds_xgb.npy')
    print(f"  ✓ XGBoost:       {log_preds_xgb.shape}")
except FileNotFoundError:
    print("  ✗ ERROR: log_preds_xgb.npy not found!")
    print("  Please run: cd tune_xgboost_v2 && python tune_xgboost.py")
    sys.exit(1)

try:
    log_preds_rf = np.load('../tune_random_forest_v2/log_preds_rf.npy')
    print(f"  ✓ Random Forest: {log_preds_rf.shape}")
except FileNotFoundError:
    print("  ✗ ERROR: log_preds_rf.npy not found!")
    print("  Please run: cd tune_random_forest_v2 && python tune_random_forest.py")
    sys.exit(1)

# Verify all shapes match
assert log_preds_cb.shape == log_preds_xgb.shape == log_preds_rf.shape
print("  ✓ All shapes match")

# ============================================================================
# STEP 2: LOAD METADATA (shift_value and test_ids)
# ============================================================================

print("\n[2/3] Loading metadata...")

try:
    test_processed = pd.read_csv('../exp_catboost_3_v2/processed_data/test_processed.csv')
    train_processed = pd.read_csv('../exp_catboost_3_v2/processed_data/train_processed.csv')
    
    test_ids = test_processed['Hospital_Id']
    shift_value = train_processed['Target_Shift_Value'].iloc[0]
    
    print(f"  ✓ Test IDs shape: {test_ids.shape}")
    print(f"  ✓ Shift value: {shift_value:.2f}")
    
except FileNotFoundError as e:
    print(f"  ✗ ERROR: {e}")
    print("  Please ensure processed_data folder exists with train/test files")
    sys.exit(1)

# ============================================================================
# STEP 3: CREATE ENSEMBLE AND GENERATE SUBMISSION
# ============================================================================

print("\n[3/3] Creating weighted ensemble...")

# CRITICAL: Calculate weighted average on LOG SCALE
ensemble_log_preds = (log_preds_cb * W_CB) + \
                     (log_preds_xgb * W_XGB) + \
                     (log_preds_rf * W_RF)

print(f"  ✓ Weighted average calculated (log scale)")
print(f"    Formula: ({W_CB:.4f} × CB) + ({W_XGB:.4f} × XGB) + ({W_RF:.4f} × RF)")

# Inverse transform from log scale to original scale
final_predictions = np.expm1(ensemble_log_preds) - shift_value

# NOTE: Do NOT clip negative values! CatBoost direct submission keeps them.
# final_predictions = np.maximum(0, final_predictions)  # REMOVED - causes mismatch!

print(f"  ✓ Inverse transformed to original scale")

# ============================================================================
# SAVE SUBMISSION
# ============================================================================

print("\n" + "="*80)
print("SAVING SUBMISSION FILE")
print("="*80)

submission = pd.DataFrame({
    'Hospital_Id': test_ids,
    'Transport_Cost': final_predictions
})

# Create filename with weights in it
filename = f'submission_ensemble_manual_{int(W_CB*100)}_{int(W_XGB*100)}_{int(W_RF*100)}.csv'
submission.to_csv(filename, index=False)

print(f"\n✓ Ensemble submission created: {filename}")
print(f"  Shape: {submission.shape}")

print(f"\nFirst 10 predictions:")
print(submission.head(10))

print(f"\nPrediction statistics:")
print(f"  Min:    ${submission['Transport_Cost'].min():,.2f}")
print(f"  Max:    ${submission['Transport_Cost'].max():,.2f}")
print(f"  Mean:   ${submission['Transport_Cost'].mean():,.2f}")
print(f"  Median: ${submission['Transport_Cost'].median():,.2f}")
print(f"  Std:    ${submission['Transport_Cost'].std():,.2f}")

# ============================================================================
# COMPARISON WITH INDIVIDUAL MODELS AND OPTIMAL ENSEMBLE
# ============================================================================

print("\n" + "="*80)
print("COMPARISON WITH INDIVIDUAL MODELS")
print("="*80)

# Load individual submissions for comparison
try:
    sub_cb = pd.read_csv('../exp_catboost_3_v2/submission_catboost.csv')
    sub_xgb = pd.read_csv('../tune_xgboost_v2/submission_xgboost_tuned.csv')
    sub_rf = pd.read_csv('../tune_random_forest_v2/submission_random_forest_tuned.csv')
    
    print("\nMean predictions:")
    print(f"  CatBoost:      ${sub_cb['Transport_Cost'].mean():,.2f}")
    print(f"  XGBoost:       ${sub_xgb['Transport_Cost'].mean():,.2f}")
    print(f"  Random Forest: ${sub_rf['Transport_Cost'].mean():,.2f}")
    print(f"  This Ensemble: ${submission['Transport_Cost'].mean():,.2f}")
    
    print("\nMedian predictions:")
    print(f"  CatBoost:      ${sub_cb['Transport_Cost'].median():,.2f}")
    print(f"  XGBoost:       ${sub_xgb['Transport_Cost'].median():,.2f}")
    print(f"  Random Forest: ${sub_rf['Transport_Cost'].median():,.2f}")
    print(f"  This Ensemble: ${submission['Transport_Cost'].median():,.2f}")
    
    # Compare with optimal ensemble if it exists
    try:
        sub_optimal = pd.read_csv('submission_ensemble_optimal.csv')
        print("\n" + "-"*80)
        print("Comparison with Optimal Ensemble:")
        print(f"  Optimal Mean:  ${sub_optimal['Transport_Cost'].mean():,.2f}")
        print(f"  Manual Mean:   ${submission['Transport_Cost'].mean():,.2f}")
        print(f"  Difference:    ${abs(sub_optimal['Transport_Cost'].mean() - submission['Transport_Cost'].mean()):,.2f}")
    except FileNotFoundError:
        pass
    
except FileNotFoundError:
    print("\n(Individual submission files not found for comparison)")

print("\n" + "="*80)
print("ENSEMBLE COMPLETE!")
print("="*80)
print(f"\nSubmit this file to Kaggle: {filename}")
print("\nEnsemble Details:")
print(f"  - Uses MANUAL weights (not optimized)")
print(f"  - Weighted averaging done on LOG SCALE (critical!)")
print(f"  - Based on {W_CB*100:.1f}% CatBoost + {W_XGB*100:.1f}% XGBoost + {W_RF*100:.1f}% RF")
print("\n💡 TIP: Edit the weights at the top of this script and re-run to try different combinations!")
print("="*80)

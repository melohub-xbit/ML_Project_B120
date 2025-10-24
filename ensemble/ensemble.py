"""
Generate final ensemble predictions using optimal weights.

This script combines the log-scale predictions from all three models
using the optimal weights found by find_weights.py.
"""

import pandas as pd
import numpy as np
import sys
import os

print("="*80)
print("GENERATING ENSEMBLE PREDICTIONS")
print("="*80)

# ============================================================================
# STEP 1: LOAD OPTIMAL WEIGHTS
# ============================================================================

print("\n[1/4] Loading optimal weights...")

try:
    optimal_weights = np.load('optimal_weights.npy')
    W_CB = optimal_weights[0]
    W_XGB = optimal_weights[1]
    W_RF = optimal_weights[2]
    
    print(f"  ✓ Loaded optimal weights:")
    print(f"    CatBoost:      {W_CB:.6f} ({W_CB*100:.2f}%)")
    print(f"    XGBoost:       {W_XGB:.6f} ({W_XGB*100:.2f}%)")
    print(f"    Random Forest: {W_RF:.6f} ({W_RF*100:.2f}%)")
    print(f"    Sum:           {W_CB + W_XGB + W_RF:.6f}")
    
except FileNotFoundError:
    print("  ✗ ERROR: optimal_weights.npy not found!")
    print("\nPlease run find_weights.py first!")
    sys.exit(1)

# Verify weights sum to 1.0
assert np.isclose(W_CB + W_XGB + W_RF, 1.0), "Weights don't sum to 1.0!"

# ============================================================================
# STEP 2: LOAD LOG-SCALE PREDICTIONS FROM ALL MODELS
# ============================================================================

print("\n[2/4] Loading log-scale predictions from each model...")

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
# STEP 3: LOAD METADATA (shift_value and test_ids)
# ============================================================================

print("\n[3/4] Loading metadata...")

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
# STEP 4: CREATE ENSEMBLE AND GENERATE SUBMISSION
# ============================================================================

print("\n[4/4] Creating weighted ensemble...")

# CRITICAL: Calculate weighted average on LOG SCALE
ensemble_log_preds = (log_preds_cb * W_CB) + \
                     (log_preds_xgb * W_XGB) + \
                     (log_preds_rf * W_RF)

print(f"  ✓ Weighted average calculated (log scale)")
print(f"    Formula: ({W_CB:.4f} × CB) + ({W_XGB:.4f} × XGB) + ({W_RF:.4f} × RF)")

# Inverse transform from log scale to original scale
final_predictions = np.expm1(ensemble_log_preds) - shift_value

# NOTE: Do NOT clip negative values! Individual models keep them.
# Clipping to 0 was causing mismatch with direct model submissions.

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
filename = f'submission_ensemble_optimal.csv'
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
# COMPARISON WITH INDIVIDUAL MODELS
# ============================================================================

print("\n" + "="*80)
print("COMPARISON WITH INDIVIDUAL MODEL PREDICTIONS")
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
    print(f"  Ensemble:      ${submission['Transport_Cost'].mean():,.2f}")
    
    print("\nMedian predictions:")
    print(f"  CatBoost:      ${sub_cb['Transport_Cost'].median():,.2f}")
    print(f"  XGBoost:       ${sub_xgb['Transport_Cost'].median():,.2f}")
    print(f"  Random Forest: ${sub_rf['Transport_Cost'].median():,.2f}")
    print(f"  Ensemble:      ${submission['Transport_Cost'].median():,.2f}")
    
except FileNotFoundError:
    print("\n(Individual submission files not found for comparison)")

print("\n" + "="*80)
print("ENSEMBLE COMPLETE!")
print("="*80)
print(f"\nSubmit this file to Kaggle: {filename}")
print("\nEnsemble Details:")
print(f"  - Uses optimal weights from Ridge regression")
print(f"  - Weighted averaging done on LOG SCALE (critical!)")
print(f"  - Based on {W_CB*100:.1f}% CatBoost + {W_XGB*100:.1f}% XGBoost + {W_RF*100:.1f}% RF")
print("="*80)

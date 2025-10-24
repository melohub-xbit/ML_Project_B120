"""
Debug script to compare CatBoost predictions between:
1. Direct CatBoost submission
2. Ensemble with 100% CatBoost weight
3. Ensemble with 26% CB / 74% RF

This will identify if there's a mismatch in the ensemble logic.
"""

import pandas as pd
import numpy as np

print("="*80)
print("DEBUGGING: WHY PURE CATBOOST BEATS ENSEMBLE")
print("="*80)

# Load shift value
train = pd.read_csv('../exp_catboost_3_v2/processed_data/train_processed.csv')
shift_value = train['Target_Shift_Value'].iloc[0]
print(f"\nShift value: {shift_value}")

# ============================================================================
# 1. LOAD CATBOOST DIRECT SUBMISSION (3997M score)
# ============================================================================
print("\n" + "="*80)
print("1. CATBOOST DIRECT SUBMISSION (from exp_catboost_3_v2)")
print("="*80)

sub_cb_direct = pd.read_csv('../exp_catboost_3_v2/submission_catboost.csv')
print(f"Shape: {sub_cb_direct.shape}")
print(f"Mean:   ${sub_cb_direct['Transport_Cost'].mean():,.2f}")
print(f"Median: ${sub_cb_direct['Transport_Cost'].median():,.2f}")
print(f"Min:    ${sub_cb_direct['Transport_Cost'].min():,.2f}")
print(f"Max:    ${sub_cb_direct['Transport_Cost'].max():,.2f}")

# ============================================================================
# 2. RECREATE CATBOOST PREDICTIONS FROM LOG SCALE
# ============================================================================
print("\n" + "="*80)
print("2. CATBOOST RECREATED FROM log_preds_cb.npy")
print("="*80)

log_preds_cb = np.load('../exp_catboost_3_v2/log_preds_cb.npy')
print(f"Log predictions shape: {log_preds_cb.shape}")
print(f"Log predictions - Min: {log_preds_cb.min():.4f}, Max: {log_preds_cb.max():.4f}, Mean: {log_preds_cb.mean():.4f}")

# Apply inverse transform EXACTLY as CatBoost script does
recreated_preds = np.expm1(log_preds_cb) - shift_value

print(f"\nRecreated predictions:")
print(f"Mean:   ${recreated_preds.mean():,.2f}")
print(f"Median: ${np.median(recreated_preds):,.2f}")
print(f"Min:    ${recreated_preds.min():,.2f}")
print(f"Max:    ${recreated_preds.max():,.2f}")

# Compare with direct submission
diff = np.abs(sub_cb_direct['Transport_Cost'].values - recreated_preds)
print(f"\nDifference from direct submission:")
print(f"Max difference: ${diff.max():,.2f}")
print(f"Mean difference: ${diff.mean():,.2f}")
print(f"Are they identical? {np.allclose(sub_cb_direct['Transport_Cost'].values, recreated_preds)}")

# ============================================================================
# 3. LOAD ENSEMBLE SUBMISSION (100% CB)
# ============================================================================
print("\n" + "="*80)
print("3. ENSEMBLE WITH 100% CATBOOST")
print("="*80)

try:
    sub_ensemble_100 = pd.read_csv('submission_ensemble_manual_100_0_0.csv')
    print(f"Shape: {sub_ensemble_100.shape}")
    print(f"Mean:   ${sub_ensemble_100['Transport_Cost'].mean():,.2f}")
    print(f"Median: ${sub_ensemble_100['Transport_Cost'].median():,.2f}")
    
    # Compare with direct CatBoost
    diff_100 = np.abs(sub_cb_direct['Transport_Cost'].values - sub_ensemble_100['Transport_Cost'].values)
    print(f"\nDifference from direct CatBoost:")
    print(f"Max difference: ${diff_100.max():,.2f}")
    print(f"Mean difference: ${diff_100.mean():,.2f}")
    print(f"Are they identical? {np.allclose(sub_cb_direct['Transport_Cost'].values, sub_ensemble_100['Transport_Cost'].values)}")
    
except FileNotFoundError:
    print("ERROR: submission_ensemble_manual_100_0_0.csv not found!")
    print("Please run ensemble_manual_weights.py with W_CB=1.0, W_XGB=0.0, W_RF=0.0 first")

# ============================================================================
# 4. LOAD BEST ENSEMBLE (26% CB, 74% RF)
# ============================================================================
print("\n" + "="*80)
print("4. BEST ENSEMBLE (26% CB, 0% XGB, 74% RF)")
print("="*80)

try:
    sub_ensemble_26_74 = pd.read_csv('submission_ensemble_manual_26_0_74.csv')
    print(f"Shape: {sub_ensemble_26_74.shape}")
    print(f"Mean:   ${sub_ensemble_26_74['Transport_Cost'].mean():,.2f}")
    print(f"Median: ${sub_ensemble_26_74['Transport_Cost'].median():,.2f}")
    
except FileNotFoundError:
    print("ERROR: submission_ensemble_manual_26_0_74.csv not found!")

# ============================================================================
# 5. CHECK RANDOM FOREST PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("5. RANDOM FOREST PREDICTIONS")
print("="*80)

try:
    log_preds_rf = np.load('../tune_random_forest_v2/log_preds_rf.npy')
    print(f"Log predictions shape: {log_preds_rf.shape}")
    print(f"Log predictions - Min: {log_preds_rf.min():.4f}, Max: {log_preds_rf.max():.4f}, Mean: {log_preds_rf.mean():.4f}")
    
    # Recreate RF predictions
    recreated_rf_preds = np.expm1(log_preds_rf) - shift_value
    print(f"\nRecreated RF predictions:")
    print(f"Mean:   ${recreated_rf_preds.mean():,.2f}")
    print(f"Median: ${np.median(recreated_rf_preds):,.2f}")
    
    # Load direct RF submission for comparison
    try:
        sub_rf_direct = pd.read_csv('../tune_random_forest_v2/submission_random_forest_tuned.csv')
        diff_rf = np.abs(sub_rf_direct['Transport_Cost'].values - recreated_rf_preds)
        print(f"\nDifference from direct RF submission:")
        print(f"Max difference: ${diff_rf.max():,.2f}")
        print(f"Mean difference: ${diff_rf.mean():,.2f}")
        print(f"Are they identical? {np.allclose(sub_rf_direct['Transport_Cost'].values, recreated_rf_preds)}")
    except FileNotFoundError:
        print("Direct RF submission not found for comparison")
    
except FileNotFoundError:
    print("ERROR: log_preds_rf.npy not found!")

# ============================================================================
# 6. MANUALLY RECREATE 26/74 ENSEMBLE
# ============================================================================
print("\n" + "="*80)
print("6. MANUALLY RECREATE 26% CB + 74% RF ENSEMBLE")
print("="*80)

try:
    # Load log predictions
    log_preds_cb = np.load('../exp_catboost_3_v2/log_preds_cb.npy')
    log_preds_rf = np.load('../tune_random_forest_v2/log_preds_rf.npy')
    
    # Ensemble on LOG scale
    W_CB = 0.26
    W_RF = 0.74
    ensemble_log = (log_preds_cb * W_CB) + (log_preds_rf * W_RF)
    
    print(f"Ensemble log predictions - Min: {ensemble_log.min():.4f}, Max: {ensemble_log.max():.4f}, Mean: {ensemble_log.mean():.4f}")
    
    # Inverse transform
    manual_ensemble = np.expm1(ensemble_log) - shift_value
    
    print(f"\nManual ensemble predictions:")
    print(f"Mean:   ${manual_ensemble.mean():,.2f}")
    print(f"Median: ${np.median(manual_ensemble):,.2f}")
    print(f"Min:    ${manual_ensemble.min():,.2f}")
    print(f"Max:    ${manual_ensemble.max():,.2f}")
    
    # Compare with script-generated ensemble
    try:
        sub_ensemble_26_74 = pd.read_csv('submission_ensemble_manual_26_0_74.csv')
        diff_manual = np.abs(sub_ensemble_26_74['Transport_Cost'].values - manual_ensemble)
        print(f"\nDifference from script-generated ensemble:")
        print(f"Max difference: ${diff_manual.max():,.2f}")
        print(f"Mean difference: ${diff_manual.mean():,.2f}")
        print(f"Are they identical? {np.allclose(sub_ensemble_26_74['Transport_Cost'].values, manual_ensemble)}")
    except:
        pass
    
except Exception as e:
    print(f"ERROR: {e}")

# ============================================================================
# 7. ANALYSIS: WHY IS CATBOOST BETTER?
# ============================================================================
print("\n" + "="*80)
print("7. ANALYSIS")
print("="*80)

print("\nKaggle Scores (from your screenshot):")
print("  Pure CatBoost:       3,997,911,549.826  ⭐ BEST")
print("  100% CB ensemble:    4,556,503,848.872  (worse by 558M!)")
print("  26% CB / 74% RF:     4,019,684,928.886  (worse by 22M)")

print("\n🚨 HYPOTHESIS 1: Shift value mismatch")
print("   If XGB/RF used different shift_value during training, ensemble is wrong")

print("\n🚨 HYPOTHESIS 2: Different preprocessing")
print("   If CB, XGB, RF used different processed data, they're incompatible")

print("\n🚨 HYPOTHESIS 3: Log predictions saved incorrectly")
print("   If log_preds_*.npy files don't match actual model outputs")

print("\n🔍 CHECK THESE:")
print("   1. Are all models using the SAME processed_data folder?")
print("   2. Do all models save log predictions BEFORE inverse transform?")
print("   3. Is the shift_value consistent across all models?")

print("\n" + "="*80)

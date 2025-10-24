"""
Generate Out-of-Fold (OOF) predictions for ensemble stacking.

This script generates validation predictions from XGBoost and Random Forest
using the SAME validation split that CatBoost used (80/20 with random_state=42).

This is critical for proper ensemble stacking.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import sys
import os

print("="*80)
print("GENERATING OOF PREDICTIONS FOR ENSEMBLE STACKING")
print("="*80)

# ============================================================================
# STEP 1: LOAD CATBOOST'S OOF PREDICTIONS (BASELINE)
# ============================================================================

print("\n[1/4] Loading CatBoost OOF predictions...")

try:
    oof_cb_preds = np.load('../exp_catboost_3_v2/oof_cb_preds.npy')
    oof_cb_true = np.load('../exp_catboost_3_v2/oof_cb_true.npy')
    print(f"  ✓ CatBoost OOF shape: {oof_cb_preds.shape}")
    print(f"  ✓ True values shape: {oof_cb_true.shape}")
except FileNotFoundError:
    print("  ✗ ERROR: CatBoost OOF files not found!")
    print("  Please run: cd exp_catboost_3_v2 && python train_and_predict_cb.py")
    sys.exit(1)

# ============================================================================
# STEP 2: LOAD PREPROCESSED DATA AND RECREATE THE SPLIT
# ============================================================================

print("\n[2/4] Loading preprocessed data and recreating CatBoost's split...")

# Load the preprocessed data (use CatBoost's version as reference)
train_processed = pd.read_csv('../exp_catboost_3_v2/processed_data/train_processed.csv')

y_train_full = train_processed['Transport_Cost_Log'].values
X_train_full = train_processed.drop(['Transport_Cost', 'Transport_Cost_Log', 'Target_Shift_Value'], axis=1)

print(f"  Full dataset shape: {X_train_full.shape}")

# Recreate the EXACT SAME split that CatBoost used (80/20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

print(f"  Train split: {X_train.shape[0]} samples")
print(f"  Val split: {X_val.shape[0]} samples")

# Sanity check: Verify we have the same validation set
assert len(y_val) == len(oof_cb_true), "Validation set size mismatch!"
print(f"  ✓ Validation set size matches CatBoost")

# ============================================================================
# STEP 3: GENERATE XGBOOST OOF PREDICTIONS
# ============================================================================

print("\n[3/4] Generating XGBoost OOF predictions...")

try:
    # Load the trained XGBoost model and encoder
    xgb_model = joblib.load('../tune_xgboost_v2/xgboost_model.pkl')
    xgb_encoder = joblib.load('../tune_xgboost_v2/xgboost_encoder.pkl')
    print("  ✓ Loaded XGBoost model and encoder")
    
    # Encode the validation set using the pre-trained encoder
    # IMPORTANT: We only transform (not fit_transform) since encoder was fit on training data
    X_val_xgb = xgb_encoder.transform(X_val)
    
    # Generate predictions on the validation set
    oof_xgb_preds = xgb_model.predict(X_val_xgb)
    
    # Save OOF predictions
    np.save('oof_xgb_preds.npy', oof_xgb_preds)
    print(f"  ✓ XGBoost OOF predictions saved: oof_xgb_preds.npy")
    print(f"  ✓ Shape: {oof_xgb_preds.shape}")
    
except FileNotFoundError as e:
    print(f"  ✗ ERROR: {e}")
    print("  Please run: cd tune_xgboost_v2 && python tune_xgboost.py")
    sys.exit(1)

# ============================================================================
# STEP 4: GENERATE RANDOM FOREST OOF PREDICTIONS
# ============================================================================

print("\n[4/4] Generating Random Forest OOF predictions...")

try:
    # Load the trained Random Forest model and encoder
    rf_model = joblib.load('../tune_random_forest_v2/random_forest_model.pkl')
    rf_encoder = joblib.load('../tune_random_forest_v2/random_forest_encoder.pkl')
    print("  ✓ Loaded Random Forest model and encoder")
    
    # Encode the validation set using the pre-trained encoder
    X_val_rf = rf_encoder.transform(X_val)
    
    # Generate predictions on the validation set
    oof_rf_preds = rf_model.predict(X_val_rf)
    
    # Save OOF predictions
    np.save('oof_rf_preds.npy', oof_rf_preds)
    print(f"  ✓ Random Forest OOF predictions saved: oof_rf_preds.npy")
    print(f"  ✓ Shape: {oof_rf_preds.shape}")
    
except FileNotFoundError as e:
    print(f"  ✗ ERROR: {e}")
    print("  Please run: cd tune_random_forest_v2 && python tune_random_forest.py")
    sys.exit(1)

# ============================================================================
# VALIDATION CHECK
# ============================================================================

print("\n" + "="*80)
print("VALIDATION CHECK")
print("="*80)

from sklearn.metrics import mean_squared_error

print("\nIndividual model RMSE on validation set (log scale):")
print(f"  CatBoost:      {np.sqrt(mean_squared_error(oof_cb_true, oof_cb_preds)):.6f}")
print(f"  XGBoost:       {np.sqrt(mean_squared_error(oof_cb_true, oof_xgb_preds)):.6f}")
print(f"  Random Forest: {np.sqrt(mean_squared_error(oof_cb_true, oof_rf_preds)):.6f}")

# Simple average ensemble
simple_avg = (oof_cb_preds + oof_xgb_preds + oof_rf_preds) / 3
print(f"\n  Simple Average Ensemble: {np.sqrt(mean_squared_error(oof_cb_true, simple_avg)):.6f}")

print("\n" + "="*80)
print("OOF PREDICTIONS GENERATED SUCCESSFULLY!")
print("="*80)
print("\nGenerated files:")
print("  - oof_xgb_preds.npy")
print("  - oof_rf_preds.npy")
print("\nNext step: Run find_weights.py to find optimal ensemble weights")
print("="*80)

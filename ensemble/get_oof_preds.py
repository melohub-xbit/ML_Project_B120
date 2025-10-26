import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sys
import os

try:
    oof_cb_preds = np.load('../exp_catboost_3_v2/oof_cb_preds.npy')
    oof_cb_true = np.load('../exp_catboost_3_v2/oof_cb_true.npy')
except FileNotFoundError:
    print("ERROR: CatBoost OOF files not found!")
    sys.exit(1)

train_processed = pd.read_csv('../exp_catboost_3_v2/processed_data/train_processed.csv')

y_train_full = train_processed['Transport_Cost_Log'].values
X_train_full = train_processed.drop(['Transport_Cost', 'Transport_Cost_Log', 'Target_Shift_Value'], axis=1)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

assert len(y_val) == len(oof_cb_true), "Validation set size mismatch!"

try:
    xgb_model = joblib.load('../tune_xgboost_v2/xgboost_model.pkl')
    xgb_encoder = joblib.load('../tune_xgboost_v2/xgboost_encoder.pkl')
    
    X_val_xgb = xgb_encoder.transform(X_val)
    oof_xgb_preds = xgb_model.predict(X_val_xgb)
    np.save('oof_xgb_preds.npy', oof_xgb_preds)
    
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

try:
    rf_model = joblib.load('../tune_random_forest_v2/random_forest_model.pkl')
    rf_encoder = joblib.load('../tune_random_forest_v2/random_forest_encoder.pkl')
    
    X_val_rf = rf_encoder.transform(X_val)
    oof_rf_preds = rf_model.predict(X_val_rf)
    np.save('oof_rf_preds.npy', oof_rf_preds)
    
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

print("\nIndividual model RMSE on validation set (log scale):")
print(f"CatBoost:      {np.sqrt(mean_squared_error(oof_cb_true, oof_cb_preds)):.6f}")
print(f"XGBoost:       {np.sqrt(mean_squared_error(oof_cb_true, oof_xgb_preds)):.6f}")
print(f"Random Forest: {np.sqrt(mean_squared_error(oof_cb_true, oof_rf_preds)):.6f}")

simple_avg = (oof_cb_preds + oof_xgb_preds + oof_rf_preds) / 3
print(f"Simple Average Ensemble: {np.sqrt(mean_squared_error(oof_cb_true, simple_avg)):.6f}")

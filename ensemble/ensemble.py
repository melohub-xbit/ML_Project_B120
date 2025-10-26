import pandas as pd
import numpy as np
import sys
import os

try:
    optimal_weights = np.load('optimal_weights.npy')
    W_CB = optimal_weights[0]
    W_XGB = optimal_weights[1]
    W_RF = optimal_weights[2]
except FileNotFoundError:
    print("ERROR: optimal_weights.npy not found!")
    sys.exit(1)

assert np.isclose(W_CB + W_XGB + W_RF, 1.0), "Weights don't sum to 1.0!"

try:
    log_preds_cb = np.load('../exp_catboost_3_v2/log_preds_cb.npy')
except FileNotFoundError:
    print("ERROR: log_preds_cb.npy not found!")
    sys.exit(1)

try:
    log_preds_xgb = np.load('../tune_xgboost_v2/log_preds_xgb.npy')
except FileNotFoundError:
    print("ERROR: log_preds_xgb.npy not found!")
    sys.exit(1)

try:
    log_preds_rf = np.load('../tune_random_forest_v2/log_preds_rf.npy')
except FileNotFoundError:
    print("ERROR: log_preds_rf.npy not found!")
    sys.exit(1)

assert log_preds_cb.shape == log_preds_xgb.shape == log_preds_rf.shape

try:
    test_processed = pd.read_csv('../exp_catboost_3_v2/processed_data/test_processed.csv')
    train_processed = pd.read_csv('../exp_catboost_3_v2/processed_data/train_processed.csv')
    
    test_ids = test_processed['Hospital_Id']
    shift_value = train_processed['Target_Shift_Value'].iloc[0]

except FileNotFoundError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

ensemble_log_preds = (log_preds_cb * W_CB) + \
                     (log_preds_xgb * W_XGB) + \
                     (log_preds_rf * W_RF)

final_predictions = np.expm1(ensemble_log_preds) - shift_value

submission = pd.DataFrame({
    'Hospital_Id': test_ids,
    'Transport_Cost': final_predictions
})

filename = f'submission_ensemble_optimal.csv'
submission.to_csv(filename, index=False)

print(f"\nEnsemble submission: {filename}")
print(f"Shape: {submission.shape}")

print(f"\nPrediction statistics:")
print(f"Min:    ${submission['Transport_Cost'].min():,.2f}")
print(f"Max:    ${submission['Transport_Cost'].max():,.2f}")
print(f"Mean:   ${submission['Transport_Cost'].mean():,.2f}")
print(f"Median: ${submission['Transport_Cost'].median():,.2f}")

try:
    sub_cb = pd.read_csv('../exp_catboost_3_v2/submission_catboost.csv')
    sub_xgb = pd.read_csv('../tune_xgboost_v2/submission_xgboost_tuned.csv')
    sub_rf = pd.read_csv('../tune_random_forest_v2/submission_random_forest_tuned.csv')
    
    print(f"\nMean predictions comparison:")
    print(f"CatBoost:      ${sub_cb['Transport_Cost'].mean():,.2f}")
    print(f"XGBoost:       ${sub_xgb['Transport_Cost'].mean():,.2f}")
    print(f"Random Forest: ${sub_rf['Transport_Cost'].mean():,.2f}")
    print(f"Ensemble:      ${submission['Transport_Cost'].mean():,.2f}")
    
except FileNotFoundError:
    pass

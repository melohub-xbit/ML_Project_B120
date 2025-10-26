import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import sys

try:
    oof_cb_preds = np.load('../exp_catboost_3_v2/oof_cb_preds.npy')
    oof_cb_true = np.load('../exp_catboost_3_v2/oof_cb_true.npy')
    oof_xgb_preds = np.load('oof_xgb_preds.npy')
    oof_rf_preds = np.load('oof_rf_preds.npy')
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

assert oof_cb_preds.shape == oof_xgb_preds.shape == oof_rf_preds.shape == oof_cb_true.shape

rmse_cb = np.sqrt(mean_squared_error(oof_cb_true, oof_cb_preds))
rmse_xgb = np.sqrt(mean_squared_error(oof_cb_true, oof_xgb_preds))
rmse_rf = np.sqrt(mean_squared_error(oof_cb_true, oof_rf_preds))

print(f"\nIndividual model RMSE (validation set, log scale):")
print(f"CatBoost:      {rmse_cb:.6f}")
print(f"XGBoost:       {rmse_xgb:.6f}")
print(f"Random Forest: {rmse_rf:.6f}")

simple_avg = (oof_cb_preds + oof_xgb_preds + oof_rf_preds) / 3
rmse_simple = np.sqrt(mean_squared_error(oof_cb_true, simple_avg))
print(f"Simple Average (1/3, 1/3, 1/3): {rmse_simple:.6f}")

X_meta = np.column_stack([oof_cb_preds, oof_xgb_preds, oof_rf_preds])
y_true = oof_cb_true

best_alpha = None
best_weights = None
best_rmse = float('inf')

alphas_to_try = [0.001, 0.01, 0.1, 1.0, 10.0]

print("\nTesting different regularization strengths:")
for alpha in alphas_to_try:
    meta_model = Ridge(alpha=alpha, positive=True, fit_intercept=False)
    meta_model.fit(X_meta, y_true)
    
    weights = meta_model.coef_
    weights = weights / weights.sum()
    
    ensemble_preds = X_meta @ weights
    rmse = np.sqrt(mean_squared_error(y_true, ensemble_preds))
    
    print(f"alpha={alpha:6.3f}: RMSE={rmse:.6f}, weights=[{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}]")
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_weights = weights
        best_alpha = alpha

print(f"\nBest alpha: {best_alpha}")
print(f"Best validation RMSE: {best_rmse:.6f}")

print(f"\nOptimal Ensemble Weights:")
print(f"CatBoost:      {best_weights[0]:.4f}  ({best_weights[0]*100:.1f}%)")
print(f"XGBoost:       {best_weights[1]:.4f}  ({best_weights[1]*100:.1f}%)")
print(f"Random Forest: {best_weights[2]:.4f}  ({best_weights[2]*100:.1f}%)")
print(f"Sum:           {best_weights.sum():.4f}")

best_single = min(rmse_cb, rmse_xgb, rmse_rf)
best_single_name = ["CatBoost", "XGBoost", "Random Forest"][np.argmin([rmse_cb, rmse_xgb, rmse_rf])]

print(f"\nBest Individual Model: {best_single_name} ({best_single:.6f})")
print(f"Simple Average: {rmse_simple:.6f} (improvement: {((best_single - rmse_simple) / best_single * 100):+.2f}%)")
print(f"Optimized Weighted: {best_rmse:.6f} (improvement: {((best_single - best_rmse) / best_single * 100):+.2f}%)")

np.save('optimal_weights.npy', best_weights)

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

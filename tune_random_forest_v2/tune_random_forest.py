import numpy as np
import pandas as pd
import sys
import os
import optuna
import json
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from category_encoders import TargetEncoder
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# Add preprocessing module to path
sys.path.append('../preprocessing')
from preprocess_data import preprocess_data

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data_for_tuning():
    """
    Loads data for tuning.
    Crucially, *DOES NOT* apply TargetEncoder yet.
    """
    print("Running preprocessing pipeline...")
    
    # Define paths
    train_path = '../dataset/train.csv'
    test_path = '../dataset/test.csv'
    output_dir = './processed_data'
    
    # Run preprocessing
    train_processed, test_processed = preprocess_data(train_path, test_path, output_dir)
    
    print(f"\nLoaded processed train data: {train_processed.shape}")
    print(f"Loaded processed test data: {test_processed.shape}")
    
    y_train_full = train_processed['Transport_Cost_Log'].values
    shift_value = train_processed['Target_Shift_Value'].iloc[0]
    X_train_full = train_processed.drop(['Transport_Cost', 'Transport_Cost_Log', 'Target_Shift_Value'], axis=1)
    
    X_test = test_processed.drop(['Hospital_Id'], axis=1)
    test_ids = test_processed['Hospital_Id'].copy()

    # Define categorical columns (as names)
    categorical_cols = [
        'Equipment_Type', 'Transport_Method', 'Hospital_Info', 
        'CrossBorder_Shipping', 'Urgent_Shipping', 'Installation_Service',
        'Fragile_Equipment', 'Rural_Hospital', 'Location_State', 'Location_Zip'
    ]
    categorical_cols = [col for col in categorical_cols if col in X_train_full.columns]

    print(f"Categorical features for TargetEncoding: {len(categorical_cols)}")
    
    return X_train_full, y_train_full, X_test, test_ids, categorical_cols, shift_value

# Load data once
X_train_full, y_train_full, X_test, test_ids, categorical_cols, shift_value = load_data_for_tuning()

# ============================================================================
# OPTUNA OBJECTIVE FUNCTION
# ============================================================================

def objective(trial):
    """Define the objective function for Optuna"""
    
    # 1. Define the hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 10, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.6, 1.0),
        'random_state': 42,
        'n_jobs': -1,
    }

    # 2. Use K-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    # Also tune the encoder's smoothing
    smoothing = trial.suggest_float('smoothing', 1.0, 5.0)

    for fold_num, (train_index, val_index) in enumerate(kf.split(X_train_full, y_train_full), 1):
        X_train, X_val = X_train_full.iloc[train_index], X_train_full.iloc[val_index]
        y_train, y_val = y_train_full[train_index], y_train_full[val_index]

        # --- !! CRITICAL: ENCODE INSIDE THE FOLD !! ---
        encoder = TargetEncoder(cols=categorical_cols, smoothing=smoothing)
        X_train_encoded = encoder.fit_transform(X_train, y_train)
        X_val_encoded = encoder.transform(X_val)
        # --- End of encoding ---
        
        # Initialize and fit the model
        model = RandomForestRegressor(**params)
        model.fit(X_train_encoded, y_train)
        
        # Get score
        preds = model.predict(X_val_encoded)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        fold_scores.append(rmse)

    # 3. Return the average score for this trial
    mean_rmse = np.mean(fold_scores)
    return mean_rmse

# ============================================================================
# RUN OPTUNA STUDY
# ============================================================================

print("\n" + "="*80)
print("STARTING OPTUNA HYPERPARAMETER TUNING FOR RANDOM FOREST")
print("="*80)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)  # RF is slower, so maybe fewer trials

print("\n" + "="*80)
print("TUNING COMPLETED")
print(f"  Best 5-Fold CV RMSE: {study.best_value:.6f}")
print("  Best Hyperparameters:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")

# ============================================================================
# SAVE BEST PARAMETERS TO FILE
# ============================================================================

# Create results directory if it doesn't exist
os.makedirs('./tuning_results', exist_ok=True)

# Prepare the results dictionary
results = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'model': 'Random Forest',
    'best_cv_rmse': float(study.best_value),
    'n_trials': len(study.trials),
    'best_params': study.best_params
}

# Save as JSON
json_path = './tuning_results/best_params_random_forest.json'
with open(json_path, 'w') as f:
    json.dump(results, f, indent=4)
print(f"\nBest parameters saved to: {json_path}")

# Also save as human-readable text
txt_path = './tuning_results/best_params_random_forest.txt'
with open(txt_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("RANDOM FOREST HYPERPARAMETER TUNING RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Tuning Date: {results['timestamp']}\n")
    f.write(f"Number of Trials: {results['n_trials']}\n")
    f.write(f"Best 5-Fold CV RMSE: {results['best_cv_rmse']:.6f}\n\n")
    f.write("Best Hyperparameters:\n")
    f.write("-" * 40 + "\n")
    for key, value in study.best_params.items():
        f.write(f"  {key}: {value}\n")
    f.write("\n" + "="*80 + "\n")
print(f"Best parameters saved to: {txt_path}")

# ============================================================================
# TRAIN FINAL MODEL AND PREDICT
# ============================================================================

print("\n" + "="*80)
print("TRAINING FINAL RANDOM FOREST MODEL WITH BEST PARAMETERS")
print("="*80)

# Get best params
best_params = study.best_params.copy()
best_smoothing = best_params.pop('smoothing')

# Update model params
best_params.update({
    'random_state': 42,
    'n_jobs': -1,
})

# --- FINAL ENCODING & TRAINING ---
# Unlike XGB, RF doesn't need early stopping, so we can train on 100% of data

# Encode the FULL training data
final_encoder_full = TargetEncoder(cols=categorical_cols, smoothing=best_smoothing)
X_train_full_encoded = final_encoder_full.fit_transform(X_train_full, y_train_full)

# Train final model on ALL data
final_model_full = RandomForestRegressor(**best_params)
print("Fitting final model on full dataset...")
final_model_full.fit(X_train_full_encoded, y_train_full)
print("Final model fitting complete.")

# Save the final model and encoder for ensemble
import joblib
joblib.dump(final_model_full, 'random_forest_model.pkl')
joblib.dump(final_encoder_full, 'random_forest_encoder.pkl')
print("\n[ENSEMBLE] Saved final model: random_forest_model.pkl")
print("[ENSEMBLE] Saved encoder: random_forest_encoder.pkl")

print("\n" + "="*80)
print("GENERATING FINAL PREDICTIONS")
print("="*80)

# Encode the TEST data using the encoder fit on FULL training data
X_test_encoded = final_encoder_full.transform(X_test)

# Predict on test set (log scale)
test_predictions_log = final_model_full.predict(X_test_encoded)

# Save log predictions for ensemble
np.save('log_preds_rf.npy', test_predictions_log)
print("[ENSEMBLE] Saved log predictions: log_preds_rf.npy")

# Inverse transform
test_predictions = np.expm1(test_predictions_log) - shift_value
test_predictions = np.maximum(0, test_predictions)

# Create submission file
submission = pd.DataFrame({
    'Hospital_Id': test_ids,
    'Transport_Cost': test_predictions
})

submission.to_csv('submission_random_forest_tuned.csv', index=False)
print(f"\nTuned submission file created: submission_random_forest_tuned.csv")
print(f"Submission shape: {submission.shape}")
print(f"\nFirst 10 predictions:")
print(submission.head(10))
print(f"\nPrediction statistics:")
print(f"  Min: ${submission['Transport_Cost'].min():.2f}")
print(f"  Max: ${submission['Transport_Cost'].max():.2f}")
print(f"  Mean: ${submission['Transport_Cost'].mean():.2f}")
print(f"  Median: ${submission['Transport_Cost'].median():.2f}")

import numpy as np
import pandas as pd
import sys
import os
import optuna
import json
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, Pool
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
    """Loads and preprocesses data for tuning"""
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
    
    # Convert categorical columns to 'category' dtype for CatBoost Pool
    for col in categorical_cols:
        X_train_full[col] = X_train_full[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    print(f"Categorical features identified: {len(categorical_cols)}")
    
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
        'iterations': 2000,  # High number, will use early stopping
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'verbose': 0,
        'random_seed': 42,
        'early_stopping_rounds': 100,
        
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.1, 1.0),
    }

    # 2. Use K-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    
    print(f"  Trial {trial.number + 1}: Testing params - LR={params['learning_rate']:.4f}, depth={params['depth']}")

    for fold_num, (train_index, val_index) in enumerate(kf.split(X_train_full, y_train_full), 1):
        X_train, X_val = X_train_full.iloc[train_index], X_train_full.iloc[val_index]
        y_train, y_val = y_train_full[train_index], y_train_full[val_index]

        # Create CatBoost pools
        train_pool = Pool(X_train, y_train, cat_features=categorical_cols)
        val_pool = Pool(X_val, y_val, cat_features=categorical_cols)

        # Initialize and fit the model
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        
        # Get score (RMSE on log-transformed target)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        fold_scores.append(rmse)

    # 3. Return the average score for this trial
    mean_rmse = np.mean(fold_scores)
    print(f"    → Completed! CV RMSE: {mean_rmse:.6f}")
    return mean_rmse

# ============================================================================
# RUN OPTUNA STUDY
# ============================================================================

print("\n" + "="*80)
print("STARTING OPTUNA HYPERPARAMETER TUNING FOR CATBOOST")
print("="*80)
print("Configuration: 50 trials × 5-fold CV = 250 model trainings")
print("Estimated time: 30-60 minutes")
print("="*80 + "\n")

# We want to MINIMIZE the RMSE
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)  # Run 50 different combinations

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
    'model': 'CatBoost',
    'best_cv_rmse': float(study.best_value),
    'n_trials': len(study.trials),
    'best_params': study.best_params
}

# Save as JSON
json_path = './tuning_results/best_params_catboost.json'
with open(json_path, 'w') as f:
    json.dump(results, f, indent=4)
print(f"\nBest parameters saved to: {json_path}")

# Also save as human-readable text
txt_path = './tuning_results/best_params_catboost.txt'
with open(txt_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("CATBOOST HYPERPARAMETER TUNING RESULTS\n")
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
print("TRAINING FINAL CATBOOST MODEL WITH BEST PARAMETERS")
print("="*80)

# Get best params and add back static ones
best_params = study.best_params.copy()
best_params.update({
    'iterations': 5000,  # Train for more iterations
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'verbose': 100,
})

final_model = CatBoostRegressor(**best_params)

# Create the full training pool
full_train_pool = Pool(X_train_full, y_train_full, cat_features=categorical_cols)

# Fit on ALL training data
final_model.fit(full_train_pool)

print("\n" + "="*80)
print("GENERATING FINAL PREDICTIONS")
print("="*80)

# Predict on test set (log scale)
test_predictions_log = final_model.predict(X_test)

# Inverse transform
test_predictions = np.expm1(test_predictions_log) - shift_value
test_predictions = np.maximum(0, test_predictions)  # Ensure no negative costs

# Create submission file
submission = pd.DataFrame({
    'Hospital_Id': test_ids,
    'Transport_Cost': test_predictions
})

submission.to_csv('submission_catboost_tuned.csv', index=False)
print(f"\nTuned submission file created: submission_catboost_tuned.csv")
print(f"Submission shape: {submission.shape}")
print(f"\nFirst 10 predictions:")
print(submission.head(10))
print(f"\nPrediction statistics:")
print(f"  Min: ${submission['Transport_Cost'].min():.2f}")
print(f"  Max: ${submission['Transport_Cost'].max():.2f}")
print(f"  Mean: ${submission['Transport_Cost'].mean():.2f}")
print(f"  Median: ${submission['Transport_Cost'].median():.2f}")

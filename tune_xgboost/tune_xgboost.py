import numpy as np
import pandas as pd
import sys
import os
import optuna
import json
from datetime import datetime
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from category_encoders import TargetEncoder
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

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
    
    train_path = '../dataset/train.csv'
    test_path = '../dataset/test.csv'
    output_dir = './processed_data'
    
    train_processed, test_processed = preprocess_data(train_path, test_path, output_dir)
    
    print(f"\nLoaded processed train data: {train_processed.shape}")
    print(f"Loaded processed test data: {test_processed.shape}")
    
    y_train_full = train_processed['Transport_Cost_Log'].values
    shift_value = train_processed['Target_Shift_Value'].iloc[0]
    X_train_full = train_processed.drop(['Transport_Cost', 'Transport_Cost_Log', 'Target_Shift_Value'], axis=1)
    
    X_test = test_processed.drop(['Hospital_Id'], axis=1)
    test_ids = test_processed['Hospital_Id'].copy()

    categorical_cols = [
        'Equipment_Type', 'Transport_Method', 'Hospital_Info', 
        'CrossBorder_Shipping', 'Urgent_Shipping', 'Installation_Service',
        'Fragile_Equipment', 'Rural_Hospital', 'Location_State', 'Location_Zip'
    ]
    categorical_cols = [col for col in categorical_cols if col in X_train_full.columns]

    print(f"Categorical features for TargetEncoding: {len(categorical_cols)}")
    
    return X_train_full, y_train_full, X_test, test_ids, categorical_cols, shift_value

X_train_full, y_train_full, X_test, test_ids, categorical_cols, shift_value = load_data_for_tuning()

# ============================================================================
# OPTUNA OBJECTIVE FUNCTION
# ============================================================================

def objective(trial):
    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 2000,  
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 100,  
        
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    smoothing = trial.suggest_float('smoothing', 1.0, 5.0)

    for fold_num, (train_index, val_index) in enumerate(kf.split(X_train_full, y_train_full), 1):
        X_train, X_val = X_train_full.iloc[train_index], X_train_full.iloc[val_index]
        y_train, y_val = y_train_full[train_index], y_train_full[val_index]

        encoder = TargetEncoder(cols=categorical_cols, smoothing=smoothing)
        
        X_train_encoded = encoder.fit_transform(X_train, y_train)
        
        X_val_encoded = encoder.transform(X_val)
        
        model = XGBRegressor(**params)
        model.fit(
            X_train_encoded, y_train,
            eval_set=[(X_val_encoded, y_val)],
            verbose=False
        )
        
        preds = model.predict(X_val_encoded)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        fold_scores.append(rmse)

    mean_rmse = np.mean(fold_scores)
    return mean_rmse

# ============================================================================
# RUN OPTUNA STUDY
# ============================================================================

print("\n" + "="*80)
print("STARTING OPTUNA HYPERPARAMETER TUNING FOR XGBOOST")
print("="*80)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)  

print("\n" + "="*80)
print("TUNING COMPLETED")
print(f"  Best 5-Fold CV RMSE: {study.best_value:.6f}")
print("  Best Hyperparameters:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")

# ============================================================================
# SAVE BEST PARAMETERS TO FILE
# ============================================================================

os.makedirs('./tuning_results', exist_ok=True)

results = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'model': 'XGBoost',
    'best_cv_rmse': float(study.best_value),
    'n_trials': len(study.trials),
    'best_params': study.best_params
}

json_path = './tuning_results/best_params_xgboost.json'
with open(json_path, 'w') as f:
    json.dump(results, f, indent=4)
print(f"\nBest parameters saved to: {json_path}")

txt_path = './tuning_results/best_params_xgboost.txt'
with open(txt_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("XGBOOST HYPERPARAMETER TUNING RESULTS\n")
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
print("TRAINING FINAL XGBOOST MODEL WITH BEST PARAMETERS")
print("="*80)

best_params = study.best_params.copy()
best_smoothing = best_params.pop('smoothing')

best_params.update({
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_estimators': 5000,  
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 100,
})

X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42
)

final_encoder = TargetEncoder(cols=categorical_cols, smoothing=best_smoothing)
X_train_final_encoded = final_encoder.fit_transform(X_train_final, y_train_final)
X_val_final_encoded = final_encoder.transform(X_val_final)

final_model = XGBRegressor(**best_params)
final_model.fit(
    X_train_final_encoded, y_train_final,
    eval_set=[(X_val_final_encoded, y_val_final)],
    verbose=100
)

best_iteration = final_model.best_iteration
print(f"\nBest iteration found: {best_iteration}")

print("Re-training on FULL dataset...")
final_model_full = XGBRegressor(**best_params)
final_model_full.set_params(n_estimators=best_iteration, early_stopping_rounds=None)

final_encoder_full = TargetEncoder(cols=categorical_cols, smoothing=best_smoothing)
X_train_full_encoded = final_encoder_full.fit_transform(X_train_full, y_train_full)

final_model_full.fit(X_train_full_encoded, y_train_full, verbose=False)

print("\n" + "="*80)
print("GENERATING FINAL PREDICTIONS")
print("="*80)

X_test_encoded = final_encoder_full.transform(X_test)

test_predictions_log = final_model_full.predict(X_test_encoded)

test_predictions = np.expm1(test_predictions_log) - shift_value
test_predictions = np.maximum(0, test_predictions)

submission = pd.DataFrame({
    'Hospital_Id': test_ids,
    'Transport_Cost': test_predictions
})

submission.to_csv('submission_xgboost_tuned.csv', index=False)
print(f"\nsubmission file: submission_xgboost_tuned.csv")
print(f"Submission shape: {submission.shape}")
print(f"\nFirst 10 predictions:")
print(submission.head(10))
print(f"\nPrediction statistics:")
print(f"  Min: ${submission['Transport_Cost'].min():.2f}")
print(f"  Max: ${submission['Transport_Cost'].max():.2f}")
print(f"  Mean: ${submission['Transport_Cost'].mean():.2f}")
print(f"  Median: ${submission['Transport_Cost'].median():.2f}")

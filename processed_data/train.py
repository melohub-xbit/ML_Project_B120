"""
Complete Machine Learning Pipeline: Train Multiple Models and Create Ensemble
=============================================================================
This script:
1. Loads preprocessed data
2. Trains CatBoost, LightGBM, and Random Forest models
3. Creates weighted ensemble predictions
4. Generates single submission file

No unnecessary files, plots, or visualizations - just training and prediction.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from category_encoders import TargetEncoder

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set to True to use pre-trained model predictions (faster, reproducible)
# Set to False to train new models from scratch
USE_PRETRAINED_MODELS = False

# Paths to pretrained predictions
PRETRAINED_PATHS = {
    'catboost': 'exp_catboost_3_v2/log_preds_cb.npy',
    'lightgbm': 'tune_lightgbm_v2/log_preds_lgbm.npy',
    'random_forest': 'tune_random_forest_v2/log_preds_rf.npy'
}

# Ensemble weights (must sum to 1.0)
ENSEMBLE_WEIGHTS = {
    'catboost': 0.55,
    'lightgbm': 0.20,
    'random_forest': 0.25
}

# Model hyperparameters (optimized from Optuna tuning)
CATBOOST_PARAMS = {
    'iterations': 500,
    'learning_rate': 0.1,
    'depth': 6,
    'l2_leaf_reg': 3,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'verbose': 25,
    'early_stopping_rounds': 50
}

# LightGBM hyperparameters (similar to XGBoost, will be tuned)
LIGHTGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 5000,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# Random Forest hyperparameters from Optuna tuning (CV RMSE: 0.1673)
RANDOM_FOREST_PARAMS = {
    'n_estimators': 485,
    'max_depth': 18,
    'min_samples_split': 18,
    'min_samples_leaf': 9,
    'max_features': 0.698531189666078,
    'random_state': 42,
    'n_jobs': -1
}

LIGHTGBM_TARGET_ENCODER_SMOOTHING = 2.0
RANDOM_FOREST_TARGET_ENCODER_SMOOTHING = 1.0128299345455185

# ============================================================================
# DATA LOADING
# ============================================================================

def load_preprocessed_data():
    """Load preprocessed data"""
    print("=" * 80)
    print("LOADING PREPROCESSED DATA")
    print("=" * 80)
    
    train_df = pd.read_csv('train_processed.csv')
    test_df = pd.read_csv('test_processed.csv')
    
    print(f"\nTrain shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Extract target and shift value
    y_train_log = train_df['Transport_Cost_Log'].values
    shift_value = train_df['Target_Shift_Value'].iloc[0]
    
    # Extract features
    X_train = train_df.drop(['Transport_Cost', 'Transport_Cost_Log', 'Target_Shift_Value'], axis=1)
    
    # Extract test data
    test_ids = test_df['Hospital_Id'].copy()
    X_test = test_df.drop(['Hospital_Id'], axis=1)
    
    # Define categorical columns
    categorical_cols = [
        'Equipment_Type', 'Transport_Method', 'Hospital_Info', 
        'CrossBorder_Shipping', 'Urgent_Shipping', 'Installation_Service',
        'Fragile_Equipment', 'Rural_Hospital', 'Location_State', 'Location_Zip'
    ]
    categorical_cols = [col for col in categorical_cols if col in X_train.columns]
    
    print(f"\nFeatures: {X_train.shape[1]}")
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Target shift value: {shift_value:.2f}")
    print(f"Target (log) - Min: {y_train_log.min():.4f}, Max: {y_train_log.max():.4f}")
    
    return X_train, y_train_log, X_test, test_ids, categorical_cols, shift_value


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def train_catboost(X_train, y_train, X_test, categorical_cols):
    """Train CatBoost model"""
    print("\n" + "=" * 80)
    print("TRAINING CATBOOST MODEL")
    print("=" * 80)
    
    # Split for early stopping validation (20% as in original)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"Train: {X_tr.shape[0]:,} samples | Validation: {X_val.shape[0]:,} samples")
    
    # Create pools
    train_pool = Pool(X_tr, y_tr, cat_features=categorical_cols)
    val_pool = Pool(X_val, y_val, cat_features=categorical_cols)
    
    # Train model
    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True, plot=False)
    
    # Validation performance
    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"\nValidation RMSE (log scale): {val_rmse:.6f}")
    print(f"Best iteration: {model.best_iteration_}")
    
    # Retrain on full data with best iteration count
    print("\nRetraining on full dataset...")
    final_params = CATBOOST_PARAMS.copy()
    final_params['iterations'] = model.best_iteration_
    final_params['verbose'] = False
    final_params.pop('early_stopping_rounds', None)
    
    full_pool = Pool(X_train, y_train, cat_features=categorical_cols)
    final_model = CatBoostRegressor(**final_params)
    final_model.fit(full_pool)
    
    # Predict on test (log scale)
    test_pred_log = final_model.predict(X_test)
    
    # Save log predictions for debugging/comparison
    np.save('log_preds_cb_new.npy', test_pred_log)
    
    print(f"[OK] CatBoost training complete")
    return test_pred_log, val_rmse


def train_lightgbm(X_train, y_train, X_test, categorical_cols):
    """Train LightGBM model with target encoding"""
    print("\n" + "=" * 80)
    print("TRAINING LIGHTGBM MODEL")
    print("=" * 80)
    
    # Split for validation (10% for consistency)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    
    print(f"Train: {X_tr.shape[0]:,} samples | Validation: {X_val.shape[0]:,} samples")
    
    # Target encode categorical features
    print("Applying target encoding...")
    encoder = TargetEncoder(cols=categorical_cols, smoothing=LIGHTGBM_TARGET_ENCODER_SMOOTHING)
    X_tr_encoded = encoder.fit_transform(X_tr, y_tr)
    X_val_encoded = encoder.transform(X_val)
    
    # Train model with early stopping
    model = LGBMRegressor(**LIGHTGBM_PARAMS)
    
    model.fit(
        X_tr_encoded, y_tr,
        eval_set=[(X_val_encoded, y_val)],
        eval_metric='rmse'
    )
    
    # Validation performance
    val_pred = model.predict(X_val_encoded)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"\nValidation RMSE (log scale): {val_rmse:.6f}")
    
    # Get best iteration (LightGBM uses best_iteration_ or n_estimators if no early stopping)
    best_iter = model.best_iteration_ if hasattr(model, 'best_iteration_') and model.best_iteration_ > 0 else model.n_estimators_
    print(f"Best iteration: {best_iter}")
    
    # Retrain on full data
    print("\nRetraining on full dataset...")
    final_params = LIGHTGBM_PARAMS.copy()
    final_params['n_estimators'] = best_iter
    final_model = LGBMRegressor(**final_params)
    
    final_encoder = TargetEncoder(cols=categorical_cols, smoothing=LIGHTGBM_TARGET_ENCODER_SMOOTHING)
    X_train_encoded = final_encoder.fit_transform(X_train, y_train)
    X_test_encoded = final_encoder.transform(X_test)
    
    final_model.fit(X_train_encoded, y_train)
    
    # Predict on test (log scale)
    test_pred_log = final_model.predict(X_test_encoded)
    
    # Save log predictions for debugging/comparison
    np.save('log_preds_lgbm_new.npy', test_pred_log)
    
    print(f"[OK] LightGBM training complete")
    return test_pred_log, val_rmse


def train_random_forest(X_train, y_train, X_test, categorical_cols):
    """Train Random Forest model with target encoding"""
    print("\n" + "=" * 80)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 80)
    
    # Note: RF doesn't use early stopping, so we train directly on full data
    # But first create a small validation set just to report performance
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )
    
    print(f"Train: {X_tr.shape[0]:,} samples | Validation: {X_val.shape[0]:,} samples (for metrics only)")
    
    # Target encode categorical features
    print("Applying target encoding...")
    encoder = TargetEncoder(cols=categorical_cols, smoothing=RANDOM_FOREST_TARGET_ENCODER_SMOOTHING)
    X_tr_encoded = encoder.fit_transform(X_tr, y_tr)
    X_val_encoded = encoder.transform(X_val)
    
    # Train model on split data (just for validation metrics)
    model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
    print("Fitting model...")
    model.fit(X_tr_encoded, y_tr)
    
    # Validation performance
    val_pred = model.predict(X_val_encoded)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"\nValidation RMSE (log scale): {val_rmse:.6f}")
    
    # Train on FULL data (no validation split for final model)
    print("\nTraining on FULL dataset...")
    final_encoder = TargetEncoder(cols=categorical_cols, smoothing=RANDOM_FOREST_TARGET_ENCODER_SMOOTHING)
    X_train_encoded = final_encoder.fit_transform(X_train, y_train)
    X_test_encoded = final_encoder.transform(X_test)
    
    final_model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
    final_model.fit(X_train_encoded, y_train)
    
    # Predict on test (log scale)
    test_pred_log = final_model.predict(X_test_encoded)
    
    # Save log predictions for debugging/comparison
    np.save('log_preds_rf_new.npy', test_pred_log)
    
    print(f"[OK] Random Forest training complete")
    return test_pred_log, val_rmse


# ============================================================================
# ENSEMBLE CREATION
# ============================================================================

def create_ensemble(predictions_dict, weights, shift_value):
    """Create weighted ensemble from log-scale predictions"""
    print("\n" + "=" * 80)
    print("CREATING WEIGHTED ENSEMBLE")
    print("=" * 80)
    
    # Normalize weights
    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0):
        print(f"[WARN] Normalizing weights (sum was {total_weight:.4f})")
        weights = {k: v/total_weight for k, v in weights.items()}
    
    print("\nEnsemble weights:")
    for model_name, weight in weights.items():
        print(f"  {model_name:15s}: {weight:.4f} ({weight*100:.1f}%)")
    
    # Calculate weighted average on log scale
    ensemble_log = np.zeros_like(predictions_dict['catboost'])
    for model_name, weight in weights.items():
        if weight > 0:
            ensemble_log += predictions_dict[model_name] * weight
    
    # Inverse transform to original scale
    final_predictions = np.expm1(ensemble_log) - shift_value
    
    print(f"\nEnsemble predictions (log scale):")
    print(f"  Min: {ensemble_log.min():.4f} | Max: {ensemble_log.max():.4f} | Mean: {ensemble_log.mean():.4f}")
    
    print(f"\nFinal predictions (original scale):")
    print(f"  Min: ${final_predictions.min():,.2f}")
    print(f"  Max: ${final_predictions.max():,.2f}")
    print(f"  Mean: ${final_predictions.mean():,.2f}")
    print(f"  Median: ${np.median(final_predictions):,.2f}")
    
    return final_predictions


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("COMPLETE ML PIPELINE: TRAIN & ENSEMBLE")
    print("=" * 80)
    
    if USE_PRETRAINED_MODELS:
        print("\n[MODE] Using pre-trained model predictions (fast & reproducible)")
        print("       Set USE_PRETRAINED_MODELS = False to train new models")
    else:
        print("\n[MODE] Training new models from scratch")
        print("       Set USE_PRETRAINED_MODELS = True to use existing predictions")
    
    print("\nThis will:")
    if USE_PRETRAINED_MODELS:
        print("  1. Load preprocessed data")
        print("  2. Load pre-trained model predictions")
        print("  3. Create weighted ensemble")
        print("  4. Generate submission file")
    else:
        print("  1. Load preprocessed data")
        print("  2. Train CatBoost, LightGBM, and Random Forest")
        print("  3. Create weighted ensemble")
        print("  4. Generate submission file")
    print("\n" + "=" * 80 + "\n")
    
    # Load data
    X_train, y_train, X_test, test_ids, categorical_cols, shift_value = load_preprocessed_data()
    
    # Train all models or load predictions
    predictions_log = {}
    validation_scores = {}
    
    if USE_PRETRAINED_MODELS:
        # Load pre-trained predictions
        print("\n" + "=" * 80)
        print("LOADING PRE-TRAINED MODEL PREDICTIONS")
        print("=" * 80)
        
        import os
        for model_name, path in PRETRAINED_PATHS.items():
            if ENSEMBLE_WEIGHTS[model_name] > 0:
                if os.path.exists(path):
                    predictions_log[model_name] = np.load(path)
                    print(f"[OK] Loaded {model_name}: {path}")
                    validation_scores[model_name] = np.nan  # Unknown
                else:
                    print(f"[ERROR] {path} not found!")
                    print(f"        Run the individual training script or set USE_PRETRAINED_MODELS = False")
                    return
            else:
                predictions_log[model_name] = np.zeros(len(test_ids))
                validation_scores[model_name] = np.nan
                print(f"[SKIP] Skipping {model_name} (weight = 0)")
    else:
        # Train models from scratch
        # Train CatBoost
        predictions_log['catboost'], validation_scores['catboost'] = train_catboost(
            X_train, y_train, X_test, categorical_cols
        )
        
        # Train LightGBM (only if weight > 0)
        if ENSEMBLE_WEIGHTS['lightgbm'] > 0:
            predictions_log['lightgbm'], validation_scores['lightgbm'] = train_lightgbm(
                X_train, y_train, X_test, categorical_cols
            )
        else:
            predictions_log['lightgbm'] = np.zeros_like(predictions_log['catboost'])
            validation_scores['lightgbm'] = np.nan
            print("\n[SKIP] Skipping LightGBM (weight = 0)")
        
        # Train Random Forest
        predictions_log['random_forest'], validation_scores['random_forest'] = train_random_forest(
            X_train, y_train, X_test, categorical_cols
        )
    
    # Print validation summary (if available)
    if not USE_PRETRAINED_MODELS:
        print("\n" + "=" * 80)
        print("VALIDATION PERFORMANCE SUMMARY")
        print("=" * 80)
        for model_name, score in validation_scores.items():
            if not np.isnan(score):
                print(f"{model_name:15s}: RMSE = {score:.6f} (log scale)")
    
    # Create ensemble
    final_predictions = create_ensemble(predictions_log, ENSEMBLE_WEIGHTS, shift_value)
    
    # Create submission file
    print("\n" + "=" * 80)
    print("CREATING SUBMISSION FILE")
    print("=" * 80)
    
    submission = pd.DataFrame({
        'Hospital_Id': test_ids,
        'Transport_Cost': final_predictions
    })
    
    # Generate descriptive filename
    weights_str = f"{int(ENSEMBLE_WEIGHTS['catboost']*100)}cb_{int(ENSEMBLE_WEIGHTS['lightgbm']*100)}lgbm_{int(ENSEMBLE_WEIGHTS['random_forest']*100)}rf"
    filename = f'submission_ensemble_{weights_str}.csv'
    
    submission.to_csv(filename, index=False)
    
    print(f"\n[OK] Submission file created: {filename}")
    print(f"     Shape: {submission.shape}")
    print(f"\nFirst 10 predictions:")
    print(submission.head(10).to_string(index=False))
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\n[SUBMIT] Submit this file: {filename}")
    print(f"\n[TIP] To change ensemble weights, edit ENSEMBLE_WEIGHTS at the top of this script")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

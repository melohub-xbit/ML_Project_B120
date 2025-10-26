"""

=============================================================================
This script:
1. Loads preprocessed data
2. Trains CatBoost, LightGBM, and Random Forest models
3. Creates weighted ensemble predictions


"""

import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from category_encoders import TargetEncoder
import optuna

warnings.filterwarnings('ignore')
np.random.seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

# Ensemble weights (will be optimized by Optuna if USE_OPTUNA_ENSEMBLE = True)
ENSEMBLE_WEIGHTS = {
    'catboost': 0.55,
    'lightgbm': 0.20,
    'random_forest': 0.25,
    'linear_regression': 0.00,  # Will be optimized by Optuna
    'ridge': 0.00,
    'lasso': 0.00,
    'elasticnet': 0.00
}

# Use Optuna to optimize ensemble weights
USE_OPTUNA_ENSEMBLE = True
OPTUNA_N_TRIALS = 3000  # Number of trials for ensemble optimization (increased for thorough search)
USE_KFOLD_CV = True  # Use K-Fold Cross-Validation for ensemble optimization
N_FOLDS = 5  # Number of folds for cross-validation

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


def train_linear_regression(X_train, y_train, X_test, categorical_cols):
    """Train Linear Regression model with target encoding"""
    print("\n" + "=" * 80)
    print("TRAINING LINEAR REGRESSION MODEL")
    print("=" * 80)
    
    # Create validation set
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )
    
    print(f"Train: {X_tr.shape[0]:,} samples | Validation: {X_val.shape[0]:,} samples")
    
    # Target encode categorical features
    print("Applying target encoding...")
    encoder = TargetEncoder(cols=categorical_cols, smoothing=1.0)
    X_tr_encoded = encoder.fit_transform(X_tr, y_tr)
    X_val_encoded = encoder.transform(X_val)
    
    # Train model
    model = LinearRegression()
    print("Fitting model...")
    model.fit(X_tr_encoded, y_tr)
    
    # Validation performance
    val_pred = model.predict(X_val_encoded)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"\nValidation RMSE (log scale): {val_rmse:.6f}")
    
    # Train on FULL data
    print("\nTraining on FULL dataset...")
    final_encoder = TargetEncoder(cols=categorical_cols, smoothing=1.0)
    X_train_encoded = final_encoder.fit_transform(X_train, y_train)
    X_test_encoded = final_encoder.transform(X_test)
    
    final_model = LinearRegression()
    final_model.fit(X_train_encoded, y_train)
    
    # Predict on test (log scale)
    test_pred_log = final_model.predict(X_test_encoded)
    
    # Save log predictions
    np.save('log_preds_lr_new.npy', test_pred_log)
    
    print(f"[OK] Linear Regression training complete")
    return test_pred_log, val_rmse


def train_ridge(X_train, y_train, X_test, categorical_cols):
    """Train Ridge Regression model with target encoding"""
    print("\n" + "=" * 80)
    print("TRAINING RIDGE REGRESSION MODEL")
    print("=" * 80)
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )
    
    print(f"Train: {X_tr.shape[0]:,} samples | Validation: {X_val.shape[0]:,} samples")
    
    # Target encode categorical features
    print("Applying target encoding...")
    encoder = TargetEncoder(cols=categorical_cols, smoothing=1.0)
    X_tr_encoded = encoder.fit_transform(X_tr, y_tr)
    X_val_encoded = encoder.transform(X_val)
    
    # Train with multiple alpha values and choose best
    best_alpha = None
    best_rmse = float('inf')
    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
    
    print("Tuning alpha parameter...")
    for alpha in alphas:
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X_tr_encoded, y_tr)
        val_pred = model.predict(X_val_encoded)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
    
    print(f"Best alpha: {best_alpha}")
    print(f"Validation RMSE (log scale): {best_rmse:.6f}")
    
    # Train on FULL data with best alpha
    print("\nTraining on FULL dataset...")
    final_encoder = TargetEncoder(cols=categorical_cols, smoothing=1.0)
    X_train_encoded = final_encoder.fit_transform(X_train, y_train)
    X_test_encoded = final_encoder.transform(X_test)
    
    final_model = Ridge(alpha=best_alpha, random_state=42)
    final_model.fit(X_train_encoded, y_train)
    
    test_pred_log = final_model.predict(X_test_encoded)
    np.save('log_preds_ridge_new.npy', test_pred_log)
    
    print(f"[OK] Ridge Regression training complete")
    return test_pred_log, best_rmse


def train_lasso(X_train, y_train, X_test, categorical_cols):
    """Train Lasso Regression model with target encoding"""
    print("\n" + "=" * 80)
    print("TRAINING LASSO REGRESSION MODEL")
    print("=" * 80)
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )
    
    print(f"Train: {X_tr.shape[0]:,} samples | Validation: {X_val.shape[0]:,} samples")
    
    # Target encode categorical features
    print("Applying target encoding...")
    encoder = TargetEncoder(cols=categorical_cols, smoothing=1.0)
    X_tr_encoded = encoder.fit_transform(X_tr, y_tr)
    X_val_encoded = encoder.transform(X_val)
    
    # Train with multiple alpha values
    best_alpha = None
    best_rmse = float('inf')
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
    
    print("Tuning alpha parameter...")
    for alpha in alphas:
        model = Lasso(alpha=alpha, random_state=42, max_iter=5000)
        model.fit(X_tr_encoded, y_tr)
        val_pred = model.predict(X_val_encoded)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
    
    print(f"Best alpha: {best_alpha}")
    print(f"Validation RMSE (log scale): {best_rmse:.6f}")
    
    # Train on FULL data with best alpha
    print("\nTraining on FULL dataset...")
    final_encoder = TargetEncoder(cols=categorical_cols, smoothing=1.0)
    X_train_encoded = final_encoder.fit_transform(X_train, y_train)
    X_test_encoded = final_encoder.transform(X_test)
    
    final_model = Lasso(alpha=best_alpha, random_state=42, max_iter=5000)
    final_model.fit(X_train_encoded, y_train)
    
    test_pred_log = final_model.predict(X_test_encoded)
    np.save('log_preds_lasso_new.npy', test_pred_log)
    
    print(f"[OK] Lasso Regression training complete")
    return test_pred_log, best_rmse


def train_elasticnet(X_train, y_train, X_test, categorical_cols):
    """Train ElasticNet Regression model with target encoding"""
    print("\n" + "=" * 80)
    print("TRAINING ELASTICNET REGRESSION MODEL")
    print("=" * 80)
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )
    
    print(f"Train: {X_tr.shape[0]:,} samples | Validation: {X_val.shape[0]:,} samples")
    
    # Target encode categorical features
    print("Applying target encoding...")
    encoder = TargetEncoder(cols=categorical_cols, smoothing=1.0)
    X_tr_encoded = encoder.fit_transform(X_tr, y_tr)
    X_val_encoded = encoder.transform(X_val)
    
    # Train with multiple alpha and l1_ratio values
    best_params = None
    best_rmse = float('inf')
    alphas = [0.001, 0.01, 0.1, 1.0]
    l1_ratios = [0.1, 0.5, 0.7, 0.9]
    
    print("Tuning alpha and l1_ratio parameters...")
    for alpha in alphas:
        for l1_ratio in l1_ratios:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=5000)
            model.fit(X_tr_encoded, y_tr)
            val_pred = model.predict(X_val_encoded)
            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {'alpha': alpha, 'l1_ratio': l1_ratio}
    
    print(f"Best params: alpha={best_params['alpha']}, l1_ratio={best_params['l1_ratio']}")
    print(f"Validation RMSE (log scale): {best_rmse:.6f}")
    
    # Train on FULL data with best params
    print("\nTraining on FULL dataset...")
    final_encoder = TargetEncoder(cols=categorical_cols, smoothing=1.0)
    X_train_encoded = final_encoder.fit_transform(X_train, y_train)
    X_test_encoded = final_encoder.transform(X_test)
    
    final_model = ElasticNet(alpha=best_params['alpha'], l1_ratio=best_params['l1_ratio'], 
                             random_state=42, max_iter=5000)
    final_model.fit(X_train_encoded, y_train)
    
    test_pred_log = final_model.predict(X_test_encoded)
    np.save('log_preds_elasticnet_new.npy', test_pred_log)
    
    print(f"[OK] ElasticNet Regression training complete")
    return test_pred_log, best_rmse


# ============================================================================
# ENSEMBLE OPTIMIZATION WITH OPTUNA
# ============================================================================

# ============================================================================
# ENSEMBLE OPTIMIZATION WITH OPTUNA (K-FOLD CV)
# ============================================================================

def optimize_ensemble_weights_optuna_kfold(X_train, y_train, categorical_cols, n_trials=3000, n_folds=5):
    """Use Optuna with K-Fold CV to find optimal ensemble weights"""
    print("\n" + "=" * 80)
    print("OPTIMIZING ENSEMBLE WEIGHTS WITH OPTUNA (K-FOLD CROSS-VALIDATION)")
    print("=" * 80)
    print(f"Using {n_folds}-Fold Cross-Validation")
    print(f"Optimizing with {n_trials} trials...")
    
    # Prepare K-Fold splits
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Store out-of-fold predictions for each model
    oof_predictions = {
        'catboost': np.zeros(len(X_train)),
        'lightgbm': np.zeros(len(X_train)),
        'random_forest': np.zeros(len(X_train)),
        'linear_regression': np.zeros(len(X_train)),
        'ridge': np.zeros(len(X_train)),
        'lasso': np.zeros(len(X_train)),
        'elasticnet': np.zeros(len(X_train))
    }
    
    # Generate out-of-fold predictions for all models
    print("\nGenerating out-of-fold predictions...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        print(f"\nFold {fold}/{n_folds}")
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # CatBoost
        print("  Training CatBoost...")
        train_pool = Pool(X_tr, y_tr, cat_features=categorical_cols)
        val_pool = Pool(X_val, y_val, cat_features=categorical_cols)
        cb_params = CATBOOST_PARAMS.copy()
        cb_params['verbose'] = False
        cb_model = CatBoostRegressor(**cb_params)
        cb_model.fit(train_pool, eval_set=val_pool)
        oof_predictions['catboost'][val_idx] = cb_model.predict(X_val)
        
        # LightGBM
        print("  Training LightGBM...")
        encoder_lgbm = TargetEncoder(cols=categorical_cols, smoothing=LIGHTGBM_TARGET_ENCODER_SMOOTHING)
        X_tr_lgbm = encoder_lgbm.fit_transform(X_tr, y_tr)
        X_val_lgbm = encoder_lgbm.transform(X_val)
        lgbm_model = LGBMRegressor(**LIGHTGBM_PARAMS)
        lgbm_model.fit(X_tr_lgbm, y_tr, eval_set=[(X_val_lgbm, y_val)], eval_metric='rmse')
        oof_predictions['lightgbm'][val_idx] = lgbm_model.predict(X_val_lgbm)
        
        # Random Forest
        print("  Training Random Forest...")
        encoder_rf = TargetEncoder(cols=categorical_cols, smoothing=RANDOM_FOREST_TARGET_ENCODER_SMOOTHING)
        X_tr_rf = encoder_rf.fit_transform(X_tr, y_tr)
        X_val_rf = encoder_rf.transform(X_val)
        rf_model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
        rf_model.fit(X_tr_rf, y_tr)
        oof_predictions['random_forest'][val_idx] = rf_model.predict(X_val_rf)
        
        # Linear Regression
        print("  Training Linear Regression...")
        encoder_lr = TargetEncoder(cols=categorical_cols, smoothing=1.0)
        X_tr_lr = encoder_lr.fit_transform(X_tr, y_tr)
        X_val_lr = encoder_lr.transform(X_val)
        lr_model = LinearRegression()
        lr_model.fit(X_tr_lr, y_tr)
        oof_predictions['linear_regression'][val_idx] = lr_model.predict(X_val_lr)
        
        # Ridge
        print("  Training Ridge...")
        ridge_model = Ridge(alpha=100.0, random_state=42)
        ridge_model.fit(X_tr_lr, y_tr)
        oof_predictions['ridge'][val_idx] = ridge_model.predict(X_val_lr)
        
        # Lasso
        print("  Training Lasso...")
        lasso_model = Lasso(alpha=10.0, random_state=42, max_iter=5000)
        lasso_model.fit(X_tr_lr, y_tr)
        oof_predictions['lasso'][val_idx] = lasso_model.predict(X_val_lr)
        
        # ElasticNet
        print("  Training ElasticNet...")
        elasticnet_model = ElasticNet(alpha=1.0, l1_ratio=0.9, random_state=42, max_iter=5000)
        elasticnet_model.fit(X_tr_lr, y_tr)
        oof_predictions['elasticnet'][val_idx] = elasticnet_model.predict(X_val_lr)
    
    print("\n" + "=" * 80)
    print("Out-of-fold predictions complete!")
    print("=" * 80)
    
    # Calculate individual model scores on full OOF predictions
    print("\nOut-of-Fold RMSE for individual models:")
    for model_name, preds in oof_predictions.items():
        rmse = np.sqrt(mean_squared_error(y_train, preds))
        print(f"  {model_name:20s}: {rmse:.6f}")
    
    # Define Optuna objective using OOF predictions
    def objective(trial):
        w_cb = trial.suggest_float('catboost', 0.0, 1.0)
        w_lgbm = trial.suggest_float('lightgbm', 0.0, 1.0)
        w_rf = trial.suggest_float('random_forest', 0.0, 1.0)
        w_lr = trial.suggest_float('linear_regression', 0.0, 1.0)
        w_ridge = trial.suggest_float('ridge', 0.0, 1.0)
        w_lasso = trial.suggest_float('lasso', 0.0, 1.0)
        w_elasticnet = trial.suggest_float('elasticnet', 0.0, 1.0)
        
        # Normalize weights
        total = w_cb + w_lgbm + w_rf + w_lr + w_ridge + w_lasso + w_elasticnet
        if total == 0:
            return float('inf')
        
        weights = [w_cb, w_lgbm, w_rf, w_lr, w_ridge, w_lasso, w_elasticnet]
        weights = [w/total for w in weights]
        
        # Calculate ensemble prediction using OOF predictions
        ensemble_pred = (weights[0] * oof_predictions['catboost'] + 
                        weights[1] * oof_predictions['lightgbm'] +
                        weights[2] * oof_predictions['random_forest'] +
                        weights[3] * oof_predictions['linear_regression'] +
                        weights[4] * oof_predictions['ridge'] +
                        weights[5] * oof_predictions['lasso'] +
                        weights[6] * oof_predictions['elasticnet'])
        
        # Calculate RMSE on full training set
        rmse = np.sqrt(mean_squared_error(y_train, ensemble_pred))
        return rmse
    
    # Run optimization
    print(f"\nRunning Optuna optimization with {n_trials} trials...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Get best weights
    best_params = study.best_params
    total = sum(best_params.values())
    optimized_weights = {k: v/total for k, v in best_params.items()}
    
    print(f"\n{'='*80}")
    print("OPTUNA K-FOLD CV OPTIMIZATION RESULTS")
    print(f"{'='*80}")
    print(f"\nBest Out-of-Fold RMSE: {study.best_value:.6f}")
    print(f"\nOptimized Ensemble Weights:")
    for model_name, weight in optimized_weights.items():
        print(f"  {model_name:20s}: {weight:.4f} ({weight*100:.2f}%)")
    
    return optimized_weights


def optimize_ensemble_weights_optuna(X_train, y_train, predictions_dict, categorical_cols, n_trials=200):
    """Use Optuna to find optimal ensemble weights on validation set"""
    print("\n" + "=" * 80)
    print("OPTIMIZING ENSEMBLE WEIGHTS WITH OPTUNA")
    print("=" * 80)
    
    # Create validation split (same as used for individual models)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )
    
    print(f"Train: {X_tr.shape[0]:,} samples | Validation: {X_val.shape[0]:,} samples")
    print(f"Optimizing with {n_trials} trials...")
    
    # Get validation predictions for all models
    val_predictions = {}
    
    # CatBoost validation predictions
    print("\nGenerating validation predictions for CatBoost...")
    train_pool = Pool(X_tr, y_tr, cat_features=categorical_cols)
    val_pool = Pool(X_val, y_val, cat_features=categorical_cols)
    cb_params = CATBOOST_PARAMS.copy()
    cb_params['verbose'] = False
    cb_model = CatBoostRegressor(**cb_params)
    cb_model.fit(train_pool, eval_set=val_pool)
    val_predictions['catboost'] = cb_model.predict(X_val)
    
    # LightGBM validation predictions
    print("Generating validation predictions for LightGBM...")
    encoder_lgbm = TargetEncoder(cols=categorical_cols, smoothing=LIGHTGBM_TARGET_ENCODER_SMOOTHING)
    X_tr_lgbm = encoder_lgbm.fit_transform(X_tr, y_tr)
    X_val_lgbm = encoder_lgbm.transform(X_val)
    lgbm_model = LGBMRegressor(**LIGHTGBM_PARAMS)
    lgbm_model.fit(X_tr_lgbm, y_tr, eval_set=[(X_val_lgbm, y_val)], eval_metric='rmse')
    val_predictions['lightgbm'] = lgbm_model.predict(X_val_lgbm)
    
    # Random Forest validation predictions
    print("Generating validation predictions for Random Forest...")
    encoder_rf = TargetEncoder(cols=categorical_cols, smoothing=RANDOM_FOREST_TARGET_ENCODER_SMOOTHING)
    X_tr_rf = encoder_rf.fit_transform(X_tr, y_tr)
    X_val_rf = encoder_rf.transform(X_val)
    rf_model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
    rf_model.fit(X_tr_rf, y_tr)
    val_predictions['random_forest'] = rf_model.predict(X_val_rf)
    
    # Linear Regression validation predictions
    print("Generating validation predictions for Linear Regression...")
    encoder_lr = TargetEncoder(cols=categorical_cols, smoothing=1.0)
    X_tr_lr = encoder_lr.fit_transform(X_tr, y_tr)
    X_val_lr = encoder_lr.transform(X_val)
    lr_model = LinearRegression()
    lr_model.fit(X_tr_lr, y_tr)
    val_predictions['linear_regression'] = lr_model.predict(X_val_lr)
    
    # Ridge Regression validation predictions
    print("Generating validation predictions for Ridge Regression...")
    ridge_model = Ridge(alpha=1.0, random_state=42)
    ridge_model.fit(X_tr_lr, y_tr)
    val_predictions['ridge'] = ridge_model.predict(X_val_lr)
    
    # Lasso Regression validation predictions
    print("Generating validation predictions for Lasso Regression...")
    lasso_model = Lasso(alpha=0.01, random_state=42, max_iter=5000)
    lasso_model.fit(X_tr_lr, y_tr)
    val_predictions['lasso'] = lasso_model.predict(X_val_lr)
    
    # ElasticNet Regression validation predictions
    print("Generating validation predictions for ElasticNet Regression...")
    elasticnet_model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=5000)
    elasticnet_model.fit(X_tr_lr, y_tr)
    val_predictions['elasticnet'] = elasticnet_model.predict(X_val_lr)
    
    print("\nValidation RMSE for individual models:")
    for model_name, preds in val_predictions.items():
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        print(f"  {model_name:20s}: {rmse:.6f}")
    
    # Define Optuna objective
    def objective(trial):
        w_cb = trial.suggest_float('catboost', 0.0, 1.0)
        w_lgbm = trial.suggest_float('lightgbm', 0.0, 1.0)
        w_rf = trial.suggest_float('random_forest', 0.0, 1.0)
        w_lr = trial.suggest_float('linear_regression', 0.0, 1.0)
        w_ridge = trial.suggest_float('ridge', 0.0, 1.0)
        w_lasso = trial.suggest_float('lasso', 0.0, 1.0)
        w_elasticnet = trial.suggest_float('elasticnet', 0.0, 1.0)
        
        # Normalize weights
        total = w_cb + w_lgbm + w_rf + w_lr + w_ridge + w_lasso + w_elasticnet
        if total == 0:
            return float('inf')
        
        weights = [w_cb, w_lgbm, w_rf, w_lr, w_ridge, w_lasso, w_elasticnet]
        weights = [w/total for w in weights]
        
        # Calculate ensemble prediction
        ensemble_pred = (weights[0] * val_predictions['catboost'] + 
                        weights[1] * val_predictions['lightgbm'] +
                        weights[2] * val_predictions['random_forest'] +
                        weights[3] * val_predictions['linear_regression'] +
                        weights[4] * val_predictions['ridge'] +
                        weights[5] * val_predictions['lasso'] +
                        weights[6] * val_predictions['elasticnet'])
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
        return rmse
    
    # Run optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Get best weights
    best_params = study.best_params
    total = sum(best_params.values())
    optimized_weights = {k: v/total for k, v in best_params.items()}
    
    print(f"\n{'='*80}")
    print("OPTUNA OPTIMIZATION RESULTS")
    print(f"{'='*80}")
    print(f"\nBest Validation RMSE: {study.best_value:.6f}")
    print(f"\nOptimized Ensemble Weights:")
    for model_name, weight in optimized_weights.items():
        print(f"  {model_name:20s}: {weight:.4f} ({weight*100:.2f}%)")
    
    return optimized_weights


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
            if model_name in ENSEMBLE_WEIGHTS and ENSEMBLE_WEIGHTS[model_name] > 0:
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
        
        # Train LightGBM
        predictions_log['lightgbm'], validation_scores['lightgbm'] = train_lightgbm(
            X_train, y_train, X_test, categorical_cols
        )
        
        # Train Random Forest
        predictions_log['random_forest'], validation_scores['random_forest'] = train_random_forest(
            X_train, y_train, X_test, categorical_cols
        )
        
        # Train Linear Regression
        predictions_log['linear_regression'], validation_scores['linear_regression'] = train_linear_regression(
            X_train, y_train, X_test, categorical_cols
        )
        
        # Train Ridge Regression
        predictions_log['ridge'], validation_scores['ridge'] = train_ridge(
            X_train, y_train, X_test, categorical_cols
        )
        
        # Train Lasso Regression
        predictions_log['lasso'], validation_scores['lasso'] = train_lasso(
            X_train, y_train, X_test, categorical_cols
        )
        
        # Train ElasticNet Regression
        predictions_log['elasticnet'], validation_scores['elasticnet'] = train_elasticnet(
            X_train, y_train, X_test, categorical_cols
        )
    
    # Print validation summary (if available)
    if not USE_PRETRAINED_MODELS:
        print("\n" + "=" * 80)
        print("VALIDATION PERFORMANCE SUMMARY")
        print("=" * 80)
        for model_name, score in validation_scores.items():
            if not np.isnan(score):
                print(f"{model_name:20s}: RMSE = {score:.6f} (log scale)")
    
    # Optimize ensemble weights with Optuna (if enabled)
    ensemble_weights = ENSEMBLE_WEIGHTS.copy()
    if USE_OPTUNA_ENSEMBLE and not USE_PRETRAINED_MODELS:
        if USE_KFOLD_CV:
            # Use K-Fold Cross-Validation for more robust optimization
            optimized_weights = optimize_ensemble_weights_optuna_kfold(
                X_train, y_train, categorical_cols, n_trials=OPTUNA_N_TRIALS, n_folds=N_FOLDS
            )
        else:
            # Use simple train/val split
            optimized_weights = optimize_ensemble_weights_optuna(
                X_train, y_train, predictions_log, categorical_cols, n_trials=OPTUNA_N_TRIALS
            )
        ensemble_weights = optimized_weights
    
    # Create ensemble
    final_predictions = create_ensemble(predictions_log, ensemble_weights, shift_value)
    
    # Create submission file
    print("\n" + "=" * 80)
    print("CREATING SUBMISSION FILE")
    print("=" * 80)
    
    submission = pd.DataFrame({
        'Hospital_Id': test_ids,
        'Transport_Cost': final_predictions
    })
    
    # Generate descriptive filename
    if USE_OPTUNA_ENSEMBLE and not USE_PRETRAINED_MODELS:
        # Include all models with non-zero weights
        weights_parts = []
        model_abbrev = {
            'catboost': 'cb',
            'lightgbm': 'lgbm', 
            'random_forest': 'rf',
            'linear_regression': 'lr',
            'ridge': 'ridge',
            'lasso': 'lasso',
            'elasticnet': 'en'
        }
        for model_name, abbrev in model_abbrev.items():
            if model_name in ensemble_weights:
                weight_pct = int(ensemble_weights[model_name] * 100)
                if weight_pct > 0:
                    weights_parts.append(f"{weight_pct}{abbrev}")
        weights_str = "_".join(weights_parts)
        cv_suffix = f"_kfold{N_FOLDS}" if USE_KFOLD_CV else ""
        filename = f'submission_ensemble_optuna{cv_suffix}_{weights_str}.csv'
    else:
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
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

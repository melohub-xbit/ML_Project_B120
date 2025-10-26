"""
Enhanced Machine Learning Pipeline with Optuna Optimization
===========================================================
This script:
1. Loads preprocessed data
2. Uses Optuna to tune LightGBM hyperparameters
3. Trains multiple models: CatBoost, LightGBM, RandomForest, LinearRegression, Ridge, Lasso, ElasticNet, XGBoost
4. Uses Optuna to optimize ensemble weights

Features:
- Optuna hyperparameter tuning for LightGBM
- Multiple regression models including Linear Regression
- Automatic ensemble weight optimization
- Cross-validation for robust evaluation
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
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
import optuna
from optuna.samplers import TPESampler

# Try to import XGBoost (optional)
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

warnings.filterwarnings('ignore')
np.random.seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Optuna configuration
N_OPTUNA_TRIALS_LGBM = 100  # Number of trials for LightGBM tuning
N_OPTUNA_TRIALS_ENSEMBLE = 200  # Number of trials for ensemble optimization
N_CV_FOLDS = 5  # Cross-validation folds

# Models to include (set to False to skip)
USE_MODELS = {
    'catboost': True,
    'lightgbm': True,
    'random_forest': True,
    'linear_regression': True,
    'ridge': True,
    'lasso': True,
    'elasticnet': True,
    'xgboost': XGBOOST_AVAILABLE
}

# Pre-tuned hyperparameters (from your previous Optuna runs)
CATBOOST_PARAMS = {
    'iterations': 500,
    'learning_rate': 0.1,
    'depth': 6,
    'l2_leaf_reg': 3,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'verbose': False,
    'early_stopping_rounds': 50
}

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
# OPTUNA: LIGHTGBM HYPERPARAMETER TUNING
# ============================================================================

def optimize_lightgbm(X_train, y_train, categorical_cols, n_trials=100):
    """Use Optuna to find best LightGBM hyperparameters"""
    print("\n" + "=" * 80)
    print(f"OPTUNA: TUNING LIGHTGBM ({n_trials} trials)")
    print("=" * 80)
    
    def objective(trial):
        # Suggest hyperparameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        smoothing = trial.suggest_float('target_encoder_smoothing', 0.5, 10.0)
        
        # K-Fold Cross-validation
        kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            # Target encoding
            encoder = TargetEncoder(cols=categorical_cols, smoothing=smoothing)
            X_tr_enc = encoder.fit_transform(X_tr, y_tr)
            X_val_enc = encoder.transform(X_val)
            
            # Train model
            model = LGBMRegressor(**params)
            model.fit(
                X_tr_enc, y_tr,
                eval_set=[(X_val_enc, y_val)],
                eval_metric='rmse',
                callbacks=[optuna.integration.LightGBMPruningCallback(trial, 'rmse')]
            )
            
            # Predict and evaluate
            val_pred = model.predict(X_val_enc)
            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            cv_scores.append(rmse)
        
        return np.mean(cv_scores)
    
    # Run optimization
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n[OPTUNA] Best RMSE: {study.best_value:.6f}")
    print(f"[OPTUNA] Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Extract best params
    best_params = study.best_params.copy()
    smoothing = best_params.pop('target_encoder_smoothing')
    best_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    })
    
    return best_params, smoothing


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def train_catboost(X_train, y_train, X_test, categorical_cols):
    """Train CatBoost model"""
    print("\n" + "=" * 80)
    print("TRAINING CATBOOST MODEL")
    print("=" * 80)
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    train_pool = Pool(X_tr, y_tr, cat_features=categorical_cols)
    val_pool = Pool(X_val, y_val, cat_features=categorical_cols)
    
    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True, plot=False)
    
    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"Validation RMSE: {val_rmse:.6f}")
    
    # Retrain on full data
    final_params = CATBOOST_PARAMS.copy()
    final_params['iterations'] = model.best_iteration_
    final_params.pop('early_stopping_rounds', None)
    
    full_pool = Pool(X_train, y_train, cat_features=categorical_cols)
    final_model = CatBoostRegressor(**final_params)
    final_model.fit(full_pool)
    
    test_pred_log = final_model.predict(X_test)
    
    return test_pred_log, val_rmse


def train_lightgbm(X_train, y_train, X_test, categorical_cols, params, smoothing):
    """Train LightGBM model with given parameters"""
    print("\n" + "=" * 80)
    print("TRAINING LIGHTGBM MODEL (OPTUNA-TUNED)")
    print("=" * 80)
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    
    # Target encoding
    encoder = TargetEncoder(cols=categorical_cols, smoothing=smoothing)
    X_tr_enc = encoder.fit_transform(X_tr, y_tr)
    X_val_enc = encoder.transform(X_val)
    
    model = LGBMRegressor(**params)
    model.fit(
        X_tr_enc, y_tr,
        eval_set=[(X_val_enc, y_val)],
        eval_metric='rmse'
    )
    
    val_pred = model.predict(X_val_enc)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"Validation RMSE: {val_rmse:.6f}")
    
    # Retrain on full data
    final_params = params.copy()
    best_iter = model.best_iteration_ if hasattr(model, 'best_iteration_') and model.best_iteration_ > 0 else model.n_estimators_
    final_params['n_estimators'] = best_iter
    
    final_encoder = TargetEncoder(cols=categorical_cols, smoothing=smoothing)
    X_train_enc = final_encoder.fit_transform(X_train, y_train)
    X_test_enc = final_encoder.transform(X_test)
    
    final_model = LGBMRegressor(**final_params)
    final_model.fit(X_train_enc, y_train)
    
    test_pred_log = final_model.predict(X_test_enc)
    
    return test_pred_log, val_rmse


def train_random_forest(X_train, y_train, X_test, categorical_cols):
    """Train Random Forest model"""
    print("\n" + "=" * 80)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 80)
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )
    
    encoder = TargetEncoder(cols=categorical_cols, smoothing=RANDOM_FOREST_TARGET_ENCODER_SMOOTHING)
    X_tr_enc = encoder.fit_transform(X_tr, y_tr)
    X_val_enc = encoder.transform(X_val)
    
    model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
    model.fit(X_tr_enc, y_tr)
    
    val_pred = model.predict(X_val_enc)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"Validation RMSE: {val_rmse:.6f}")
    
    # Train on full data
    final_encoder = TargetEncoder(cols=categorical_cols, smoothing=RANDOM_FOREST_TARGET_ENCODER_SMOOTHING)
    X_train_enc = final_encoder.fit_transform(X_train, y_train)
    X_test_enc = final_encoder.transform(X_test)
    
    final_model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
    final_model.fit(X_train_enc, y_train)
    
    test_pred_log = final_model.predict(X_test_enc)
    
    return test_pred_log, val_rmse


def train_linear_regression(X_train, y_train, X_test, categorical_cols):
    """Train Linear Regression model"""
    print("\n" + "=" * 80)
    print("TRAINING LINEAR REGRESSION MODEL")
    print("=" * 80)
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Target encoding
    encoder = TargetEncoder(cols=categorical_cols, smoothing=1.0)
    X_tr_enc = encoder.fit_transform(X_tr, y_tr)
    X_val_enc = encoder.transform(X_val)
    
    # Standardize features for linear models
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_enc)
    X_val_scaled = scaler.transform(X_val_enc)
    
    model = LinearRegression()
    model.fit(X_tr_scaled, y_tr)
    
    val_pred = model.predict(X_val_scaled)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"Validation RMSE: {val_rmse:.6f}")
    
    # Train on full data
    final_encoder = TargetEncoder(cols=categorical_cols, smoothing=1.0)
    X_train_enc = final_encoder.fit_transform(X_train, y_train)
    X_test_enc = final_encoder.transform(X_test)
    
    final_scaler = StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train_enc)
    X_test_scaled = final_scaler.transform(X_test_enc)
    
    final_model = LinearRegression()
    final_model.fit(X_train_scaled, y_train)
    
    test_pred_log = final_model.predict(X_test_scaled)
    
    return test_pred_log, val_rmse


def train_ridge(X_train, y_train, X_test, categorical_cols):
    """Train Ridge Regression model"""
    print("\n" + "=" * 80)
    print("TRAINING RIDGE REGRESSION MODEL")
    print("=" * 80)
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    encoder = TargetEncoder(cols=categorical_cols, smoothing=1.0)
    X_tr_enc = encoder.fit_transform(X_tr, y_tr)
    X_val_enc = encoder.transform(X_val)
    
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_enc)
    X_val_scaled = scaler.transform(X_val_enc)
    
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_tr_scaled, y_tr)
    
    val_pred = model.predict(X_val_scaled)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"Validation RMSE: {val_rmse:.6f}")
    
    # Train on full data
    final_encoder = TargetEncoder(cols=categorical_cols, smoothing=1.0)
    X_train_enc = final_encoder.fit_transform(X_train, y_train)
    X_test_enc = final_encoder.transform(X_test)
    
    final_scaler = StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train_enc)
    X_test_scaled = final_scaler.transform(X_test_enc)
    
    final_model = Ridge(alpha=1.0, random_state=42)
    final_model.fit(X_train_scaled, y_train)
    
    test_pred_log = final_model.predict(X_test_scaled)
    
    return test_pred_log, val_rmse


def train_lasso(X_train, y_train, X_test, categorical_cols):
    """Train Lasso Regression model"""
    print("\n" + "=" * 80)
    print("TRAINING LASSO REGRESSION MODEL")
    print("=" * 80)
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    encoder = TargetEncoder(cols=categorical_cols, smoothing=1.0)
    X_tr_enc = encoder.fit_transform(X_tr, y_tr)
    X_val_enc = encoder.transform(X_val)
    
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_enc)
    X_val_scaled = scaler.transform(X_val_enc)
    
    model = Lasso(alpha=0.01, random_state=42, max_iter=10000)
    model.fit(X_tr_scaled, y_tr)
    
    val_pred = model.predict(X_val_scaled)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"Validation RMSE: {val_rmse:.6f}")
    
    # Train on full data
    final_encoder = TargetEncoder(cols=categorical_cols, smoothing=1.0)
    X_train_enc = final_encoder.fit_transform(X_train, y_train)
    X_test_enc = final_encoder.transform(X_test)
    
    final_scaler = StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train_enc)
    X_test_scaled = final_scaler.transform(X_test_enc)
    
    final_model = Lasso(alpha=0.01, random_state=42, max_iter=10000)
    final_model.fit(X_train_scaled, y_train)
    
    test_pred_log = final_model.predict(X_test_scaled)
    
    return test_pred_log, val_rmse


def train_elasticnet(X_train, y_train, X_test, categorical_cols):
    """Train ElasticNet Regression model"""
    print("\n" + "=" * 80)
    print("TRAINING ELASTICNET REGRESSION MODEL")
    print("=" * 80)
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    encoder = TargetEncoder(cols=categorical_cols, smoothing=1.0)
    X_tr_enc = encoder.fit_transform(X_tr, y_tr)
    X_val_enc = encoder.transform(X_val)
    
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_enc)
    X_val_scaled = scaler.transform(X_val_enc)
    
    model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000)
    model.fit(X_tr_scaled, y_tr)
    
    val_pred = model.predict(X_val_scaled)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"Validation RMSE: {val_rmse:.6f}")
    
    # Train on full data
    final_encoder = TargetEncoder(cols=categorical_cols, smoothing=1.0)
    X_train_enc = final_encoder.fit_transform(X_train, y_train)
    X_test_enc = final_encoder.transform(X_test)
    
    final_scaler = StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train_enc)
    X_test_scaled = final_scaler.transform(X_test_enc)
    
    final_model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000)
    final_model.fit(X_train_scaled, y_train)
    
    test_pred_log = final_model.predict(X_test_scaled)
    
    return test_pred_log, val_rmse


def train_xgboost(X_train, y_train, X_test, categorical_cols):
    """Train XGBoost model"""
    print("\n" + "=" * 80)
    print("TRAINING XGBOOST MODEL")
    print("=" * 80)
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    
    encoder = TargetEncoder(cols=categorical_cols, smoothing=2.0)
    X_tr_enc = encoder.fit_transform(X_tr, y_tr)
    X_val_enc = encoder.transform(X_val)
    
    params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    }
    
    model = XGBRegressor(**params)
    model.fit(
        X_tr_enc, y_tr,
        eval_set=[(X_val_enc, y_val)],
        verbose=False
    )
    
    val_pred = model.predict(X_val_enc)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"Validation RMSE: {val_rmse:.6f}")
    
    # Train on full data
    final_params = params.copy()
    best_iter = model.best_iteration if hasattr(model, 'best_iteration') and model.best_iteration > 0 else model.n_estimators
    final_params['n_estimators'] = best_iter
    
    final_encoder = TargetEncoder(cols=categorical_cols, smoothing=2.0)
    X_train_enc = final_encoder.fit_transform(X_train, y_train)
    X_test_enc = final_encoder.transform(X_test)
    
    final_model = XGBRegressor(**final_params)
    final_model.fit(X_train_enc, y_train)
    
    test_pred_log = final_model.predict(X_test_enc)
    
    return test_pred_log, val_rmse


# ============================================================================
# OPTUNA: ENSEMBLE WEIGHT OPTIMIZATION
# ============================================================================

def optimize_ensemble_weights(predictions_dict, X_train, y_train, categorical_cols, n_trials=200):
    """Use Optuna to find optimal ensemble weights"""
    print("\n" + "=" * 80)
    print(f"OPTUNA: OPTIMIZING ENSEMBLE WEIGHTS ({n_trials} trials)")
    print("=" * 80)
    
    model_names = list(predictions_dict.keys())
    print(f"Models in ensemble: {model_names}")
    
    # Get out-of-fold predictions for validation
    print("\nGenerating out-of-fold predictions for validation...")
    oof_predictions = {name: np.zeros(len(y_train)) for name in model_names}
    
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"  Fold {fold_idx + 1}/{N_CV_FOLDS}...")
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # Train each model and get OOF predictions
        if 'catboost' in model_names:
            cb_pool = Pool(X_tr, y_tr, cat_features=categorical_cols)
            cb_model = CatBoostRegressor(**CATBOOST_PARAMS)
            cb_model.fit(cb_pool, verbose=False)
            oof_predictions['catboost'][val_idx] = cb_model.predict(X_val)
        
        if 'lightgbm' in model_names:
            encoder = TargetEncoder(cols=categorical_cols, smoothing=LIGHTGBM_TARGET_ENCODER_SMOOTHING)
            X_tr_enc = encoder.fit_transform(X_tr, y_tr)
            X_val_enc = encoder.transform(X_val)
            lgbm_model = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, verbose=-1)
            lgbm_model.fit(X_tr_enc, y_tr)
            oof_predictions['lightgbm'][val_idx] = lgbm_model.predict(X_val_enc)
        
        if 'random_forest' in model_names:
            encoder = TargetEncoder(cols=categorical_cols, smoothing=RANDOM_FOREST_TARGET_ENCODER_SMOOTHING)
            X_tr_enc = encoder.fit_transform(X_tr, y_tr)
            X_val_enc = encoder.transform(X_val)
            rf_model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
            rf_model.fit(X_tr_enc, y_tr)
            oof_predictions['random_forest'][val_idx] = rf_model.predict(X_val_enc)
        
        # For linear models, use simpler training
        for model_name in ['linear_regression', 'ridge', 'lasso', 'elasticnet']:
            if model_name in model_names:
                encoder = TargetEncoder(cols=categorical_cols, smoothing=1.0)
                X_tr_enc = encoder.fit_transform(X_tr, y_tr)
                X_val_enc = encoder.transform(X_val)
                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X_tr_enc)
                X_val_scaled = scaler.transform(X_val_enc)
                
                if model_name == 'linear_regression':
                    model = LinearRegression()
                elif model_name == 'ridge':
                    model = Ridge(alpha=1.0, random_state=42)
                elif model_name == 'lasso':
                    model = Lasso(alpha=0.01, random_state=42, max_iter=10000)
                else:  # elasticnet
                    model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000)
                
                model.fit(X_tr_scaled, y_tr)
                oof_predictions[model_name][val_idx] = model.predict(X_val_scaled)
        
        if 'xgboost' in model_names and XGBOOST_AVAILABLE:
            encoder = TargetEncoder(cols=categorical_cols, smoothing=2.0)
            X_tr_enc = encoder.fit_transform(X_tr, y_tr)
            X_val_enc = encoder.transform(X_val)
            xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
            xgb_model.fit(X_tr_enc, y_tr)
            oof_predictions['xgboost'][val_idx] = xgb_model.predict(X_val_enc)
    
    def objective(trial):
        # Suggest weights for each model
        weights = {}
        for name in model_names:
            weights[name] = trial.suggest_float(f'weight_{name}', 0.0, 1.0)
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            return float('inf')
        
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculate weighted ensemble prediction
        ensemble_pred = np.zeros(len(y_train))
        for name, weight in weights.items():
            ensemble_pred += oof_predictions[name] * weight
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_train, ensemble_pred))
        return rmse
    
    # Run optimization
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n[OPTUNA] Best ensemble RMSE: {study.best_value:.6f}")
    print(f"[OPTUNA] Optimal weights:")
    
    # Extract and normalize best weights
    best_weights = {}
    for name in model_names:
        best_weights[name] = study.best_params[f'weight_{name}']
    
    total_weight = sum(best_weights.values())
    best_weights = {k: v/total_weight for k, v in best_weights.items()}
    
    for name, weight in best_weights.items():
        print(f"  {name}: {weight:.4f} ({weight*100:.1f}%)")
    
    return best_weights


# ============================================================================
# ENSEMBLE CREATION
# ============================================================================

def create_ensemble(predictions_dict, weights, shift_value):
    """Create weighted ensemble from log-scale predictions"""
    print("\n" + "=" * 80)
    print("CREATING WEIGHTED ENSEMBLE")
    print("=" * 80)
    
    print("\nEnsemble weights:")
    for model_name, weight in weights.items():
        print(f"  {model_name:20s}: {weight:.4f} ({weight*100:.1f}%)")
    
    # Calculate weighted average on log scale
    ensemble_log = np.zeros_like(predictions_dict[list(predictions_dict.keys())[0]])
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
    print("ENHANCED ML PIPELINE WITH OPTUNA OPTIMIZATION")
    print("=" * 80)
    print("\nFeatures:")
    print("  • Optuna hyperparameter tuning for LightGBM")
    print("  • Multiple regression models (Linear, Ridge, Lasso, ElasticNet, XGBoost)")
    print("  • Optuna ensemble weight optimization")
    print("  • Cross-validation for robust evaluation")
    print("=" * 80 + "\n")
    
    # Load data
    X_train, y_train, X_test, test_ids, categorical_cols, shift_value = load_preprocessed_data()
    
    # Step 1: Optimize LightGBM hyperparameters
    lgbm_params, lgbm_smoothing = optimize_lightgbm(X_train, y_train, categorical_cols, N_OPTUNA_TRIALS_LGBM)
    
    # Step 2: Train all models
    predictions_log = {}
    validation_scores = {}
    
    if USE_MODELS['catboost']:
        predictions_log['catboost'], validation_scores['catboost'] = train_catboost(
            X_train, y_train, X_test, categorical_cols
        )
        np.save('log_preds_cb_optuna.npy', predictions_log['catboost'])
    
    if USE_MODELS['lightgbm']:
        predictions_log['lightgbm'], validation_scores['lightgbm'] = train_lightgbm(
            X_train, y_train, X_test, categorical_cols, lgbm_params, lgbm_smoothing
        )
        np.save('log_preds_lgbm_optuna.npy', predictions_log['lightgbm'])
    
    if USE_MODELS['random_forest']:
        predictions_log['random_forest'], validation_scores['random_forest'] = train_random_forest(
            X_train, y_train, X_test, categorical_cols
        )
        np.save('log_preds_rf_optuna.npy', predictions_log['random_forest'])
    
    if USE_MODELS['linear_regression']:
        predictions_log['linear_regression'], validation_scores['linear_regression'] = train_linear_regression(
            X_train, y_train, X_test, categorical_cols
        )
        np.save('log_preds_lr_optuna.npy', predictions_log['linear_regression'])
    
    if USE_MODELS['ridge']:
        predictions_log['ridge'], validation_scores['ridge'] = train_ridge(
            X_train, y_train, X_test, categorical_cols
        )
        np.save('log_preds_ridge_optuna.npy', predictions_log['ridge'])
    
    if USE_MODELS['lasso']:
        predictions_log['lasso'], validation_scores['lasso'] = train_lasso(
            X_train, y_train, X_test, categorical_cols
        )
        np.save('log_preds_lasso_optuna.npy', predictions_log['lasso'])
    
    if USE_MODELS['elasticnet']:
        predictions_log['elasticnet'], validation_scores['elasticnet'] = train_elasticnet(
            X_train, y_train, X_test, categorical_cols
        )
        np.save('log_preds_elasticnet_optuna.npy', predictions_log['elasticnet'])
    
    if USE_MODELS['xgboost'] and XGBOOST_AVAILABLE:
        predictions_log['xgboost'], validation_scores['xgboost'] = train_xgboost(
            X_train, y_train, X_test, categorical_cols
        )
        np.save('log_preds_xgb_optuna.npy', predictions_log['xgboost'])
    
    # Print validation summary
    print("\n" + "=" * 80)
    print("VALIDATION PERFORMANCE SUMMARY")
    print("=" * 80)
    sorted_scores = sorted(validation_scores.items(), key=lambda x: x[1])
    for model_name, score in sorted_scores:
        print(f"{model_name:20s}: RMSE = {score:.6f} (log scale)")
    
    # Step 3: Optimize ensemble weights
    optimal_weights = optimize_ensemble_weights(
        predictions_log, X_train, y_train, categorical_cols, N_OPTUNA_TRIALS_ENSEMBLE
    )
    
    # Step 4: Create ensemble
    final_predictions = create_ensemble(predictions_log, optimal_weights, shift_value)
    
    # Create submission file
    print("\n" + "=" * 80)
    
    print("=" * 80)
    
    submission = pd.DataFrame({
        'Hospital_Id': test_ids,
        'Transport_Cost': final_predictions
    })
    
    filename = 'submission_optuna_optimized_ensemble.csv'
    submission.to_csv(filename, index=False)
    
    print(f"\n[OK] Submission file created: {filename}")
    print(f"     Shape: {submission.shape}")
    print(f"\nFirst 10 predictions:")
    print(submission.head(10).to_string(index=False))
    
    # Save optimal weights to file
    weights_df = pd.DataFrame([optimal_weights])
    weights_df.to_csv('optimal_ensemble_weights.csv', index=False)
    print(f"\n[OK] Optimal weights saved to: optimal_ensemble_weights.csv")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    


if __name__ == "__main__":
    main()

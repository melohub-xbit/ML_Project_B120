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

USE_PRETRAINED_MODELS = False

PRETRAINED_PATHS = {
    'catboost': 'exp_catboost_3_v2/log_preds_cb.npy',
    'lightgbm': 'tune_lightgbm_v2/log_preds_lgbm.npy',
    'random_forest': 'tune_random_forest_v2/log_preds_rf.npy'
}

ENSEMBLE_WEIGHTS = {
    'catboost': 0.55,
    'lightgbm': 0.20,
    'random_forest': 0.25
}

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

def load_preprocessed_data():
    train_df = pd.read_csv('train_processed.csv')
    test_df = pd.read_csv('test_processed.csv')
    
    y_train_log = train_df['Transport_Cost_Log'].values
    shift_value = train_df['Target_Shift_Value'].iloc[0]
    
    X_train = train_df.drop(['Transport_Cost', 'Transport_Cost_Log', 'Target_Shift_Value'], axis=1)
    
    test_ids = test_df['Hospital_Id'].copy()
    X_test = test_df.drop(['Hospital_Id'], axis=1)
    
    categorical_cols = [
        'Equipment_Type', 'Transport_Method', 'Hospital_Info', 
        'CrossBorder_Shipping', 'Urgent_Shipping', 'Installation_Service',
        'Fragile_Equipment', 'Rural_Hospital', 'Location_State', 'Location_Zip'
    ]
    categorical_cols = [col for col in categorical_cols if col in X_train.columns]
    
    return X_train, y_train_log, X_test, test_ids, categorical_cols, shift_value

def train_catboost(X_train, y_train, X_test, categorical_cols):
    print("\nTraining CatBoost...")
    
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
    
    final_params = CATBOOST_PARAMS.copy()
    final_params['iterations'] = model.best_iteration_
    final_params['verbose'] = False
    final_params.pop('early_stopping_rounds', None)
    
    full_pool = Pool(X_train, y_train, cat_features=categorical_cols)
    final_model = CatBoostRegressor(**final_params)
    final_model.fit(full_pool)
    
    test_pred_log = final_model.predict(X_test)
    np.save('log_preds_cb_new.npy', test_pred_log)
    
    return test_pred_log, val_rmse


def train_lightgbm(X_train, y_train, X_test, categorical_cols):
    print("\nTraining LightGBM...")
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    
    encoder = TargetEncoder(cols=categorical_cols, smoothing=LIGHTGBM_TARGET_ENCODER_SMOOTHING)
    X_tr_encoded = encoder.fit_transform(X_tr, y_tr)
    X_val_encoded = encoder.transform(X_val)
    
    model = LGBMRegressor(**LIGHTGBM_PARAMS)
    model.fit(
        X_tr_encoded, y_tr,
        eval_set=[(X_val_encoded, y_val)],
        eval_metric='rmse'
    )
    
    val_pred = model.predict(X_val_encoded)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"Validation RMSE: {val_rmse:.6f}")
    
    best_iter = model.best_iteration_ if hasattr(model, 'best_iteration_') and model.best_iteration_ > 0 else model.n_estimators_
    
    final_params = LIGHTGBM_PARAMS.copy()
    final_params['n_estimators'] = best_iter
    final_model = LGBMRegressor(**final_params)
    
    final_encoder = TargetEncoder(cols=categorical_cols, smoothing=LIGHTGBM_TARGET_ENCODER_SMOOTHING)
    X_train_encoded = final_encoder.fit_transform(X_train, y_train)
    X_test_encoded = final_encoder.transform(X_test)
    
    final_model.fit(X_train_encoded, y_train)
    test_pred_log = final_model.predict(X_test_encoded)
    np.save('log_preds_lgbm_new.npy', test_pred_log)
    
    return test_pred_log, val_rmse


def train_random_forest(X_train, y_train, X_test, categorical_cols):
    print("\nTraining Random Forest...")
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )
    
    encoder = TargetEncoder(cols=categorical_cols, smoothing=RANDOM_FOREST_TARGET_ENCODER_SMOOTHING)
    X_tr_encoded = encoder.fit_transform(X_tr, y_tr)
    X_val_encoded = encoder.transform(X_val)
    
    model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
    model.fit(X_tr_encoded, y_tr)
    
    val_pred = model.predict(X_val_encoded)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"Validation RMSE: {val_rmse:.6f}")
    
    final_encoder = TargetEncoder(cols=categorical_cols, smoothing=RANDOM_FOREST_TARGET_ENCODER_SMOOTHING)
    X_train_encoded = final_encoder.fit_transform(X_train, y_train)
    X_test_encoded = final_encoder.transform(X_test)
    
    final_model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
    final_model.fit(X_train_encoded, y_train)
    
    test_pred_log = final_model.predict(X_test_encoded)
    np.save('log_preds_rf_new.npy', test_pred_log)
    
    return test_pred_log, val_rmse

def create_ensemble(predictions_dict, weights, shift_value):
    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0):
        weights = {k: v/total_weight for k, v in weights.items()}
    
    ensemble_log = np.zeros_like(predictions_dict['catboost'])
    for model_name, weight in weights.items():
        if weight > 0:
            ensemble_log += predictions_dict[model_name] * weight
    
    final_predictions = np.expm1(ensemble_log) - shift_value
    
    print(f"\nEnsemble predictions:")
    print(f"Min: ${final_predictions.min():,.2f} | Max: ${final_predictions.max():,.2f} | Mean: ${final_predictions.mean():,.2f}")
    
    return final_predictions

def main():
    X_train, y_train, X_test, test_ids, categorical_cols, shift_value = load_preprocessed_data()
    
    predictions_log = {}
    validation_scores = {}
    
    if USE_PRETRAINED_MODELS:
        print("\nLoading pre-trained predictions...")
        import os
        for model_name, path in PRETRAINED_PATHS.items():
            if ENSEMBLE_WEIGHTS[model_name] > 0:
                if os.path.exists(path):
                    predictions_log[model_name] = np.load(path)
                    validation_scores[model_name] = np.nan
                else:
                    print(f"ERROR: {path} not found!")
                    return
            else:
                predictions_log[model_name] = np.zeros(len(test_ids))
                validation_scores[model_name] = np.nan
    else:
        predictions_log['catboost'], validation_scores['catboost'] = train_catboost(
            X_train, y_train, X_test, categorical_cols
        )
        
        if ENSEMBLE_WEIGHTS['lightgbm'] > 0:
            predictions_log['lightgbm'], validation_scores['lightgbm'] = train_lightgbm(
                X_train, y_train, X_test, categorical_cols
            )
        else:
            predictions_log['lightgbm'] = np.zeros_like(predictions_log['catboost'])
            validation_scores['lightgbm'] = np.nan
        
        predictions_log['random_forest'], validation_scores['random_forest'] = train_random_forest(
            X_train, y_train, X_test, categorical_cols
        )
        
        print("\nValidation scores:")
        for model_name, score in validation_scores.items():
            if not np.isnan(score):
                print(f"{model_name}: {score:.6f}")
    
    final_predictions = create_ensemble(predictions_log, ENSEMBLE_WEIGHTS, shift_value)
    
    submission = pd.DataFrame({
        'Hospital_Id': test_ids,
        'Transport_Cost': final_predictions
    })
    
    weights_str = f"{int(ENSEMBLE_WEIGHTS['catboost']*100)}cb_{int(ENSEMBLE_WEIGHTS['lightgbm']*100)}lgbm_{int(ENSEMBLE_WEIGHTS['random_forest']*100)}rf"
    filename = f'submission_ensemble_{weights_str}.csv'
    
    submission.to_csv(filename, index=False)
    
    print(f"\nSubmission: {filename}")
    print(f"Shape: {submission.shape}")


if __name__ == "__main__":
    main()

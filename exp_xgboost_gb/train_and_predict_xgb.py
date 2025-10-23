import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from category_encoders import TargetEncoder
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Add preprocessing module to path
sys.path.append('../preprocessing')
from preprocess_data import preprocess_data

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_encode_data():
    """Run preprocessing and apply target encoding"""
    print("Running preprocessing pipeline...")
    
    # Define paths
    train_path = '../dataset/train.csv'
    test_path = '../dataset/test.csv'
    output_dir = './processed_data'
    
    # Run preprocessing
    train_processed, test_processed = preprocess_data(train_path, test_path, output_dir)
    
    print(f"\nLoading processed data from {output_dir}/")
    print(f"Train shape: {train_processed.shape}")
    print(f"Test shape: {test_processed.shape}")
    
    # Extract features and targets
    # Train data has: features + Transport_Cost + Transport_Cost_Log + Target_Shift_Value
    y_train_log = train_processed['Transport_Cost_Log'].values
    shift_value = train_processed['Target_Shift_Value'].iloc[0]
    
    # Drop target columns from features
    X_train = train_processed.drop(['Transport_Cost', 'Transport_Cost_Log', 'Target_Shift_Value'], axis=1)
    
    # Test data has: Hospital_Id + features
    test_ids = test_processed['Hospital_Id'].copy()
    X_test = test_processed.drop(['Hospital_Id'], axis=1)
    
    # Define categorical columns for target encoding
    categorical_cols = [
        'Equipment_Type', 
        'Transport_Method', 
        'Hospital_Info', 
        'CrossBorder_Shipping',
        'Urgent_Shipping',
        'Installation_Service',
        'Fragile_Equipment',
        'Rural_Hospital', 
        'Location_State', 
        'Location_Zip'
    ]
    
    # Verify categorical columns exist
    categorical_cols = [col for col in categorical_cols if col in X_train.columns]
    
    print(f"\nFeatures before encoding: {X_train.shape[1]}")
    print(f"Categorical features for target encoding: {len(categorical_cols)}")
    
    # Apply Target Encoding
    print("\nApplying Target Encoding...")
    target_encoder = TargetEncoder(cols=categorical_cols, smoothing=1.0)
    X_train_encoded = target_encoder.fit_transform(X_train, y_train_log)
    X_test_encoded = target_encoder.transform(X_test)
    
    print(f"Features after encoding: {X_train_encoded.shape[1]}")
    print(f"Target shift value: {shift_value}")
    
    return X_train_encoded, y_train_log, X_test_encoded, test_ids, shift_value


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_feature_importance(model, feature_names, model_name, save_path='plots/feature_importance.png'):
    """Plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    else:
        print(f"Model {model_name} does not have feature_importances_")
        return
    
    # Sort by importance
    indices = np.argsort(feature_importance)[::-1]
    
    # Plot top 20 features
    top_n = min(20, len(feature_importance))
    top_indices = indices[:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), feature_importance[top_indices])
    plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
    plt.xlabel('Importance Score')
    plt.title(f'Top 20 Feature Importance - {model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved feature importance plot to {save_path}")
    plt.close()


def plot_metrics(y_true, y_pred, split_name='Validation', model_name='Model', save_path='plots/metrics.png'):
    """Plot prediction metrics"""
    plt.figure(figsize=(15, 5))
    
    # Predictions vs Actual
    plt.subplot(1, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual (Log Scale)')
    plt.ylabel('Predicted (Log Scale)')
    plt.title(f'{model_name} - {split_name}: Predictions vs Actual')
    plt.grid(True, alpha=0.3)
    
    # Residuals
    plt.subplot(1, 3, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5, s=20)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted (Log Scale)')
    plt.ylabel('Residuals')
    plt.title(f'{model_name} - {split_name}: Residual Plot')
    plt.grid(True, alpha=0.3)
    
    # Distribution of residuals
    plt.subplot(1, 3, 3)
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title(f'{model_name} - {split_name}: Residual Distribution')
    plt.axvline(x=0, color='r', linestyle='--', lw=2)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved metrics plot to {save_path}")
    plt.close()


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model"""
    print("\n" + "=" * 80)
    print("TRAINING XGBOOST MODEL")
    print("=" * 80)
    
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=25
    )
    
    return model


def train_gradient_boosting(X_train, y_train):
    """Train Gradient Boosting model"""
    print("\n" + "=" * 80)
    print("TRAINING GRADIENT BOOSTING MODEL")
    print("=" * 80)
    
    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    return model


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("XGBOOST & GRADIENT BOOSTING WITH TARGET ENCODING")
    print("=" * 80)
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Load and encode data
    X_train_full, y_train_full, X_test, test_ids, shift_value = load_and_encode_data()
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Store feature names
    feature_names = X_train.columns.tolist()
    
    # ========================================================================
    # XGBOOST
    # ========================================================================
    
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    
    # Evaluate XGBoost on validation set
    print("\n" + "=" * 80)
    print("XGBOOST VALIDATION METRICS (LOG SCALE)")
    print("=" * 80)
    
    y_val_pred_xgb = xgb_model.predict(X_val)
    val_metrics_xgb = calculate_metrics(y_val, y_val_pred_xgb)
    
    for metric, value in val_metrics_xgb.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot XGBoost metrics and feature importance
    plot_metrics(y_val, y_val_pred_xgb, 'Validation', 'XGBoost', 'plots/xgboost_validation_metrics.png')
    plot_feature_importance(xgb_model, feature_names, 'XGBoost', 'plots/xgboost_feature_importance.png')
    
    # Train XGBoost on full dataset
    print("\n" + "=" * 80)
    print("TRAINING XGBOOST ON FULL DATASET")
    print("=" * 80)
    
    xgb_final = XGBRegressor(
        n_estimators=xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') else 500,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    xgb_final.fit(X_train_full, y_train_full)
    
    # Generate XGBoost predictions
    test_predictions_log_xgb = xgb_final.predict(X_test)
    test_predictions_xgb = np.expm1(test_predictions_log_xgb) - shift_value
    
    # Create XGBoost submission file
    submission_xgb = pd.DataFrame({
        'Hospital_Id': test_ids,
        'Transport_Cost': test_predictions_xgb
    })
    submission_xgb.to_csv('submission_xgboost.csv', index=False)
    print(f"\nXGBoost submission file created: submission_xgboost.csv")
    print(f"Shape: {submission_xgb.shape}")
    print(f"Prediction statistics:")
    print(submission_xgb['Transport_Cost'].describe())
    
    # ========================================================================
    # GRADIENT BOOSTING
    # ========================================================================
    
    gb_model = train_gradient_boosting(X_train, y_train)
    
    # Evaluate Gradient Boosting on validation set
    print("\n" + "=" * 80)
    print("GRADIENT BOOSTING VALIDATION METRICS (LOG SCALE)")
    print("=" * 80)
    
    y_val_pred_gb = gb_model.predict(X_val)
    val_metrics_gb = calculate_metrics(y_val, y_val_pred_gb)
    
    for metric, value in val_metrics_gb.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot Gradient Boosting metrics and feature importance
    plot_metrics(y_val, y_val_pred_gb, 'Validation', 'GradientBoosting', 'plots/gb_validation_metrics.png')
    plot_feature_importance(gb_model, feature_names, 'GradientBoosting', 'plots/gb_feature_importance.png')
    
    # Train Gradient Boosting on full dataset
    print("\n" + "=" * 80)
    print("TRAINING GRADIENT BOOSTING ON FULL DATASET")
    print("=" * 80)
    
    gb_final = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    gb_final.fit(X_train_full, y_train_full)
    
    # Generate Gradient Boosting predictions
    test_predictions_log_gb = gb_final.predict(X_test)
    test_predictions_gb = np.expm1(test_predictions_log_gb) - shift_value
    
    # Create Gradient Boosting submission file
    submission_gb = pd.DataFrame({
        'Hospital_Id': test_ids,
        'Transport_Cost': test_predictions_gb
    })
    submission_gb.to_csv('submission_gradient_boosting.csv', index=False)
    print(f"\nGradient Boosting submission file created: submission_gradient_boosting.csv")
    print(f"Shape: {submission_gb.shape}")
    print(f"Prediction statistics:")
    print(submission_gb['Transport_Cost'].describe())
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON (VALIDATION SET)")
    print("=" * 80)
    print(f"\nXGBoost:")
    for metric, value in val_metrics_xgb.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nGradient Boosting:")
    for metric, value in val_metrics_gb.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n" + "=" * 80)
    print("COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()

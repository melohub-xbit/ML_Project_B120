import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor, Pool
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_preprocessed_data():
    """Load preprocessed data and create local copies"""
    print("Loading preprocessed data...")
    
    # Create dataset directory
    os.makedirs('dataset_preprocessed', exist_ok=True)
    
    # Copy preprocessed files to local folder
    src_train = '../processed_data/train_processed.csv'
    src_test = '../processed_data/test_processed.csv'
    dst_train = 'dataset_preprocessed/train_processed.csv'
    dst_test = 'dataset_preprocessed/test_processed.csv'
    
    shutil.copy(src_train, dst_train)
    shutil.copy(src_test, dst_test)
    print(f"Copied preprocessed data to dataset_preprocessed/")
    
    # Load data
    train_df = pd.read_csv(dst_train)
    test_df = pd.read_csv(dst_test)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Extract features and targets
    # Train data has: features + Transport_Cost + Transport_Cost_Log + Target_Shift_Value
    y_train = train_df['Transport_Cost_Log'].values
    shift_value = train_df['Target_Shift_Value'].iloc[0]
    
    # Drop target columns from features
    X_train = train_df.drop(['Transport_Cost', 'Transport_Cost_Log', 'Target_Shift_Value'], axis=1)
    
    # Test data has: Hospital_Id + features
    test_ids = test_df['Hospital_Id'].copy()
    X_test = test_df.drop(['Hospital_Id'], axis=1)
    
    # Define categorical columns (these are now label-encoded but still categorical)
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
    
    print(f"\nFeatures: {X_train.shape[1]}")
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Target shift value: {shift_value}")
    
    return X_train, y_train, X_test, test_ids, categorical_cols, shift_value


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_training_history(model, save_path='plots/training_history.png'):
    """Plot training and validation losses"""
    train_rmse = model.evals_result_['learn']['RMSE']
    val_rmse = model.evals_result_['validation']['RMSE']
    
    plt.figure(figsize=(12, 5))
    
    # RMSE Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_rmse, label='Train RMSE', linewidth=2)
    plt.plot(val_rmse, label='Validation RMSE', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title('Training and Validation RMSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning curve
    plt.subplot(1, 2, 2)
    plt.plot(train_rmse, label='Train RMSE', linewidth=2)
    plt.plot(val_rmse, label='Validation RMSE', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title('Learning Curve (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training history plot to {save_path}")
    plt.close()


def plot_feature_importance(model, save_path='plots/feature_importance.png'):
    """Plot feature importance"""
    feature_importance = model.get_feature_importance()
    feature_names = model.feature_names_
    
    # Sort by importance
    indices = np.argsort(feature_importance)[::-1]
    
    # Plot top 20 features
    top_n = min(20, len(feature_importance))
    top_indices = indices[:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), feature_importance[top_indices])
    plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
    plt.xlabel('Importance Score')
    plt.title('Top 20 Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved feature importance plot to {save_path}")
    plt.close()


def plot_metrics(y_true, y_pred, split_name='Validation', save_path='plots/metrics.png'):
    """Plot prediction metrics"""
    plt.figure(figsize=(15, 5))
    
    # Predictions vs Actual
    plt.subplot(1, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Transport Cost')
    plt.ylabel('Predicted Transport Cost')
    plt.title(f'{split_name} Set: Predictions vs Actual')
    plt.grid(True, alpha=0.3)
    
    # Residuals
    plt.subplot(1, 3, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5, s=20)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Transport Cost')
    plt.ylabel('Residuals')
    plt.title(f'{split_name} Set: Residual Plot')
    plt.grid(True, alpha=0.3)
    
    # Distribution of residuals
    plt.subplot(1, 3, 3)
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title(f'{split_name} Set: Residual Distribution')
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
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("CATBOOST REGRESSION WITH PREPROCESSED DATA")
    print("=" * 80)
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Load preprocessed data
    X_train_full, y_train_full, X_test, test_ids, categorical_cols, shift_value = load_preprocessed_data()
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Create CatBoost pools with categorical features
    train_pool = Pool(X_train, y_train, cat_features=categorical_cols)
    val_pool = Pool(X_val, y_val, cat_features=categorical_cols)
    
    # Train model
    print("\n" + "=" * 80)
    print("TRAINING CATBOOST MODEL")
    print("=" * 80)
    
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=25,
        early_stopping_rounds=50
    )
    
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        plot=False
    )
    
    # Plot training history
    plot_training_history(model, 'plots/training_history.png')
    
    # Plot feature importance
    plot_feature_importance(model, 'plots/feature_importance.png')
    
    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("VALIDATION METRICS (LOG SCALE)")
    print("=" * 80)
    
    y_val_pred = model.predict(X_val)
    val_metrics = calculate_metrics(y_val, y_val_pred)
    
    for metric, value in val_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot validation metrics
    plot_metrics(y_val, y_val_pred, 'Validation', 'plots/validation_metrics.png')
    
    # Train on full dataset
    print("\n" + "=" * 80)
    print("TRAINING ON FULL DATASET")
    print("=" * 80)
    
    train_pool_full = Pool(X_train_full, y_train_full, cat_features=categorical_cols)
    final_model = CatBoostRegressor(
        iterations=model.best_iteration_,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3,
        loss_function='RMSE',
        random_seed=42,
        verbose=False
    )
    
    final_model.fit(train_pool_full)
    
    # Predict on test set
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)
    
    # Predictions are in log scale, need to reverse transform
    test_predictions_log = final_model.predict(X_test)
    test_predictions = np.expm1(test_predictions_log) - shift_value
    
    print(f"Shift value used for inverse transform: {shift_value}")
    
    # Create submission file
    submission = pd.DataFrame({
        'Hospital_Id': test_ids,
        'Transport_Cost': test_predictions
    })
    
    submission.to_csv('submission_catboost.csv', index=False)
    print(f"\nSubmission file created: submission_catboost.csv")
    print(f"Shape: {submission.shape}")
    print(f"\nFirst few predictions:")
    print(submission.head())
    print(f"\nPrediction statistics:")
    print(submission['Transport_Cost'].describe())
    
    print("\n" + "=" * 80)
    print("COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()

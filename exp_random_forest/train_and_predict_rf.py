import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from category_encoders import TargetEncoder
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

sys.path.append('../preprocessing')
from preprocess_data import preprocess_data

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_encode_data():
    print("Running preprocessing pipeline...")
    
    train_path = '../dataset/train.csv'
    test_path = '../dataset/test.csv'
    output_dir = './processed_data'
    
    train_processed, test_processed = preprocess_data(train_path, test_path, output_dir)
    
    print(f"\nLoading processed data from {output_dir}/")
    print(f"Train shape: {train_processed.shape}")
    print(f"Test shape: {test_processed.shape}")
    
    y_train_log = train_processed['Transport_Cost_Log'].values
    shift_value = train_processed['Target_Shift_Value'].iloc[0]
    
    X_train = train_processed.drop(['Transport_Cost', 'Transport_Cost_Log', 'Target_Shift_Value'], axis=1)
    
    test_ids = test_processed['Hospital_Id'].copy()
    X_test = test_processed.drop(['Hospital_Id'], axis=1)
    
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
    
    categorical_cols = [col for col in categorical_cols if col in X_train.columns]
    
    print(f"\nFeatures before encoding: {X_train.shape[1]}")
    print(f"Categorical features for target encoding: {len(categorical_cols)}")
    
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

def plot_feature_importance(model, feature_names, save_path='plots/feature_importance.png'):
    feature_importance = model.feature_importances_
    
    indices = np.argsort(feature_importance)[::-1]
    
    top_n = min(20, len(feature_importance))
    top_indices = indices[:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), feature_importance[top_indices])
    plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
    plt.xlabel('Importance Score')
    plt.title('Top 20 Feature Importance - Random Forest')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved feature importance plot to {save_path}")
    plt.close()


def plot_metrics(y_true, y_pred, split_name='Validation', save_path='plots/metrics.png'):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual (Log Scale)')
    plt.ylabel('Predicted (Log Scale)')
    plt.title(f'{split_name}: Predictions vs Actual')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5, s=20)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted (Log Scale)')
    plt.ylabel('Residuals')
    plt.title(f'{split_name}: Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title(f'{split_name}: Residual Distribution')
    plt.axvline(x=0, color='r', linestyle='--', lw=2)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved metrics plot to {save_path}")
    plt.close()


def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("RANDOM FOREST REGRESSION WITH TARGET ENCODING")
    print("=" * 80)
    
    os.makedirs('plots', exist_ok=True)
    
    X_train_full, y_train_full, X_test, test_ids, shift_value = load_and_encode_data()
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    feature_names = X_train.columns.tolist()
    
    print("\n" + "=" * 80)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 80)
    
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    plot_feature_importance(model, feature_names, 'plots/feature_importance.png')
    
    print("\n" + "=" * 80)
    print("VALIDATION METRICS (LOG SCALE)")
    print("=" * 80)
    
    y_val_pred = model.predict(X_val)
    val_metrics = calculate_metrics(y_val, y_val_pred)
    
    for metric, value in val_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    plot_metrics(y_val, y_val_pred, 'Validation', 'plots/validation_metrics.png')
    
    print("\n" + "=" * 80)
    print("TRAINING ON FULL DATASET")
    print("=" * 80)
    
    final_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    final_model.fit(X_train_full, y_train_full)
    
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)
    
    test_predictions_log = final_model.predict(X_test)
    test_predictions = np.expm1(test_predictions_log) - shift_value
    
    print(f"Shift value used for inverse transform: {shift_value}")
    
    submission = pd.DataFrame({
        'Hospital_Id': test_ids,
        'Transport_Cost': test_predictions
    })
    
    submission.to_csv('submission_random_forest.csv', index=False)
    print(f"\nSubmission file created: submission_random_forest.csv")
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

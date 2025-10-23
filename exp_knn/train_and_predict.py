import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
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
    """Run preprocessing, apply one-hot encoding, scaling, and PCA"""
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
    
    # Define categorical columns for one-hot encoding
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
    
    # Identify numerical columns
    numerical_cols = [col for col in X_train.columns if col not in categorical_cols]
    
    print(f"\nFeatures before encoding: {X_train.shape[1]}")
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Numerical features: {len(numerical_cols)}")
    
    # Apply One-Hot Encoding
    print("\nApplying One-Hot Encoding...")
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    X_train_cat = ohe.fit_transform(X_train[categorical_cols])
    X_test_cat = ohe.transform(X_test[categorical_cols])
    
    # Get feature names from one-hot encoding
    ohe_feature_names = ohe.get_feature_names_out(categorical_cols)
    
    # Create DataFrames for one-hot encoded features
    X_train_cat_df = pd.DataFrame(X_train_cat, columns=ohe_feature_names, index=X_train.index)
    X_test_cat_df = pd.DataFrame(X_test_cat, columns=ohe_feature_names, index=X_test.index)
    
    # Combine numerical and one-hot encoded features
    X_train_combined = pd.concat([X_train[numerical_cols].reset_index(drop=True), 
                                   X_train_cat_df.reset_index(drop=True)], axis=1)
    X_test_combined = pd.concat([X_test[numerical_cols].reset_index(drop=True), 
                                  X_test_cat_df.reset_index(drop=True)], axis=1)
    
    print(f"Features after one-hot encoding: {X_train_combined.shape[1]}")
    
    # Apply StandardScaler to ALL features
    print("\nApplying StandardScaler to all features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_combined)
    X_test_scaled = scaler.transform(X_test_combined)
    
    print(f"Features after scaling: {X_train_scaled.shape[1]}")
    
    # Apply PCA to reduce dimensionality (keep 95% variance)
    print("\nApplying PCA (n_components=0.95)...")
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"Features after PCA: {X_train_pca.shape[1]}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"Number of components: {pca.n_components_}")
    print(f"Target shift value: {shift_value}")
    
    return X_train_pca, y_train_log, X_test_pca, test_ids, shift_value, pca


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_pca_variance(pca, save_path='plots/pca_variance.png'):
    """Plot PCA explained variance"""
    plt.figure(figsize=(12, 5))
    
    # Individual explained variance
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Component')
    plt.grid(True, alpha=0.3)
    
    # Cumulative explained variance
    plt.subplot(1, 2, 2)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumsum) + 1), cumsum, linewidth=2)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved PCA variance plot to {save_path}")
    plt.close()


def plot_metrics(y_true, y_pred, split_name='Validation', save_path='plots/metrics.png'):
    """Plot prediction metrics"""
    plt.figure(figsize=(15, 5))
    
    # Predictions vs Actual
    plt.subplot(1, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual (Log Scale)')
    plt.ylabel('Predicted (Log Scale)')
    plt.title(f'{split_name}: Predictions vs Actual')
    plt.grid(True, alpha=0.3)
    
    # Residuals
    plt.subplot(1, 3, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5, s=20)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted (Log Scale)')
    plt.ylabel('Residuals')
    plt.title(f'{split_name}: Residual Plot')
    plt.grid(True, alpha=0.3)
    
    # Distribution of residuals
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
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("KNN REGRESSION WITH ONE-HOT ENCODING, SCALING & PCA")
    print("=" * 80)
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Load and encode data
    X_train_full, y_train_full, X_test, test_ids, shift_value, pca = load_and_encode_data()
    
    # Plot PCA variance
    plot_pca_variance(pca, 'plots/pca_variance.png')
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Validation set: {X_val.shape[0]} samples, {X_val.shape[1]} features")
    
    # Train model
    print("\n" + "=" * 80)
    print("TRAINING KNN MODEL")
    print("=" * 80)
    
    model = KNeighborsRegressor(
        n_neighbors=10,
        weights='distance',  # Weight by inverse distance
        algorithm='auto',
        n_jobs=-1
    )
    
    print("Fitting KNN model...")
    model.fit(X_train, y_train)
    print("Training completed!")
    
    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("VALIDATION METRICS (LOG SCALE)")
    print("=" * 80)
    
    print("Predicting on validation set...")
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
    
    final_model = KNeighborsRegressor(
        n_neighbors=10,
        weights='distance',
        algorithm='auto',
        n_jobs=-1
    )
    
    print("Fitting final KNN model...")
    final_model.fit(X_train_full, y_train_full)
    print("Training completed!")
    
    # Predict on test set
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)
    
    print("Predicting on test set...")
    # Predictions are in log scale, need to reverse transform
    test_predictions_log = final_model.predict(X_test)
    test_predictions = np.expm1(test_predictions_log) - shift_value
    
    print(f"Shift value used for inverse transform: {shift_value}")
    
    # Create submission file
    submission = pd.DataFrame({
        'Hospital_Id': test_ids,
        'Transport_Cost': test_predictions
    })
    
    submission.to_csv('submission_knn.csv', index=False)
    print(f"\nSubmission file created: submission_knn.csv")
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

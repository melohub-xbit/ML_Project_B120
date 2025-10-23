import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def preprocess_data(train_df, test_df):
    """Preprocess training and test data"""
    print("Preprocessing data...")
    
    # Separate target and features
    X_train = train_df.drop(['Transport_Cost'], axis=1)
    y_train = train_df['Transport_Cost'].values
    X_test = test_df.copy()
    
    # Store customer IDs
    train_ids = X_train['Hospital_Id'].copy()
    test_ids = X_test['Hospital_Id'].copy()
    
    # Drop ID column
    X_train = X_train.drop(['Hospital_Id'], axis=1)
    X_test = X_test.drop(['Hospital_Id'], axis=1)
    
    # Handle dates
    for df in [X_train, X_test]:
        df['Order_Placed_Date'] = pd.to_datetime(df['Order_Placed_Date'], format='%m/%d/%y', errors='coerce')
        df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'], format='%m/%d/%y', errors='coerce')
        
        df['Order_Year'] = df['Order_Placed_Date'].dt.year
        df['Order_Month'] = df['Order_Placed_Date'].dt.month
        df['Order_Day'] = df['Order_Placed_Date'].dt.day
        df['Delivery_Year'] = df['Delivery_Date'].dt.year
        df['Delivery_Month'] = df['Delivery_Date'].dt.month
        df['Delivery_Day'] = df['Delivery_Date'].dt.day
        df['Delivery_Time'] = (df['Delivery_Date'] - df['Order_Placed_Date']).dt.days
        
        df.drop(['Order_Placed_Date', 'Delivery_Date'], axis=1, inplace=True)
    
    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    # Fill missing values
    for col in numerical_cols:
        X_train[col].fillna(X_train[col].median(), inplace=True)
        X_test[col].fillna(X_train[col].median(), inplace=True)
    
    for col in categorical_cols:
        X_train[col].fillna('Unknown', inplace=True)
        X_test[col].fillna('Unknown', inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = X_test[col].apply(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1)
        label_encoders[col] = le
    
    print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    return X_train, y_train, X_test, test_ids


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_training_history(results, save_path='plots/training_history.png'):
    """Plot training and validation losses"""
    train_rmse = results['train']['rmse']
    val_rmse = results['validation']['rmse']
    
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


def plot_feature_importance(model, feature_names, save_path='plots/feature_importance.png'):
    """Plot feature importance"""
    importance = model.get_score(importance_type='weight')
    
    # Convert to sorted list
    importance_list = [(k, v) for k, v in importance.items()]
    importance_list.sort(key=lambda x: x[1], reverse=True)
    
    # Plot top 20 features
    top_n = min(20, len(importance_list))
    features = [x[0] for x in importance_list[:top_n]]
    scores = [x[1] for x in importance_list[:top_n]]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(features)), scores)
    plt.yticks(range(len(features)), features)
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
    print("XGBOOST REGRESSION")
    print("=" * 80)
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    sample_submission = pd.read_csv('dataset/sample_submission.csv')
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Preprocess data
    X_train_full, y_train_full, X_test, test_ids = preprocess_data(train_df, test_df)
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Train model
    print("\n" + "=" * 80)
    print("TRAINING XGBOOST MODEL")
    print("=" * 80)
    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'seed': 42
    }
    
    evals = [(dtrain, 'train'), (dval, 'validation')]
    evals_result = {}
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=20,
        verbose_eval=10
    )
    
    # Plot training history
    plot_training_history(evals_result, 'plots/training_history.png')
    
    # Plot feature importance
    plot_feature_importance(model, X_train.columns.tolist(), 'plots/feature_importance.png')
    
    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("VALIDATION METRICS")
    print("=" * 80)
    
    y_val_pred = model.predict(dval)
    val_metrics = calculate_metrics(y_val, y_val_pred)
    
    for metric, value in val_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot validation metrics
    plot_metrics(y_val, y_val_pred, 'Validation', 'plots/validation_metrics.png')
    
    # Train on full dataset
    print("\n" + "=" * 80)
    print("TRAINING ON FULL DATASET")
    print("=" * 80)
    
    dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full)
    final_model = xgb.train(
        params,
        dtrain_full,
        num_boost_round=model.best_iteration,
        verbose_eval=False
    )
    
    # Predict on test set
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)
    
    dtest = xgb.DMatrix(X_test)
    test_predictions = final_model.predict(dtest)
    
    # Create submission file
    submission = pd.DataFrame({
        'Hospital_Id': test_ids,
        'Transport_Cost': test_predictions
    })
    
    submission.to_csv('submission_xgboost.csv', index=False)
    print(f"\nSubmission file created: submission_xgboost.csv")
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

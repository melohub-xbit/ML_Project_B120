import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# RIDGE REGRESSION WITH GRADIENT DESCENT FROM SCRATCH
# ============================================================================

class RidgeRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000, alpha=1.0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.alpha = alpha  
        self.weights = None
        self.bias = None
        self.train_losses = []
        self.val_losses = []
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        n_samples, n_features = X_train.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iterations):
            y_pred = np.dot(X_train, self.weights) + self.bias
            
            mse_loss = np.mean((y_train - y_pred) ** 2)
            l2_penalty = self.alpha * np.sum(self.weights ** 2)
            total_loss = mse_loss + l2_penalty
            
            dw = (-2 / n_samples) * np.dot(X_train.T, (y_train - y_pred)) + 2 * self.alpha * self.weights
            db = (-2 / n_samples) * np.sum(y_train - y_pred)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            train_rmse = np.sqrt(mse_loss)
            self.train_losses.append(train_rmse)
            
            if X_val is not None and y_val is not None:
                y_val_pred = self.predict(X_val)
                val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                self.val_losses.append(val_rmse)
            
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.n_iterations}, Train RMSE: {train_rmse:.4f}", end="")
                if X_val is not None:
                    print(f", Val RMSE: {val_rmse:.4f}")
                else:
                    print()
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def preprocess_data(train_df, test_df):
    print("Preprocessing data...")
    
    X_train = train_df.drop(['Transport_Cost'], axis=1)
    y_train = train_df['Transport_Cost'].values
    X_test = test_df.copy()
    
    train_ids = X_train['Hospital_Id'].copy()
    test_ids = X_test['Hospital_Id'].copy()
    
    X_train = X_train.drop(['Hospital_Id'], axis=1)
    X_test = X_test.drop(['Hospital_Id'], axis=1)
    
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
    
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numerical_cols:
        X_train[col].fillna(X_train[col].median(), inplace=True)
        X_test[col].fillna(X_train[col].median(), inplace=True)
    
    for col in categorical_cols:
        X_train[col].fillna('Unknown', inplace=True)
        X_test[col].fillna('Unknown', inplace=True)
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = X_test[col].apply(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1)
        label_encoders[col] = le
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training samples: {X_train_scaled.shape[0]}, Features: {X_train_scaled.shape[1]}")
    print(f"Test samples: {X_test_scaled.shape[0]}")
    
    return X_train_scaled, y_train, X_test_scaled, test_ids


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_training_history(train_losses, val_losses, save_path='plots/training_history.png'):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train RMSE', linewidth=2)
    plt.plot(val_losses, label='Validation RMSE', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title('Training and Validation RMSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train RMSE', linewidth=2)
    plt.plot(val_losses, label='Validation RMSE', linewidth=2)
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


def plot_metrics(y_true, y_pred, split_name='Validation', save_path='plots/metrics.png'):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Transport Cost')
    plt.ylabel('Predicted Transport Cost')
    plt.title(f'{split_name} Set: Predictions vs Actual')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5, s=20)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Transport Cost')
    plt.ylabel('Residuals')
    plt.title(f'{split_name} Set: Residual Plot')
    plt.grid(True, alpha=0.3)
    
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
    print("RIDGE REGRESSION WITH GRADIENT DESCENT - FROM SCRATCH")
    print("=" * 80)
    
    import os
    os.makedirs('plots', exist_ok=True)
    
    print("\nLoading data...")
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    sample_submission = pd.read_csv('dataset/sample_submission.csv')
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    X_train_full, y_train_full, X_test, test_ids = preprocess_data(train_df, test_df)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    print("\n" + "=" * 80)
    print("TRAINING RIDGE REGRESSION MODEL")
    print("=" * 80)
    
    model = RidgeRegressionGD(
        learning_rate=0.01,
        n_iterations=1000,
        alpha=10.0
    )
    
    model.fit(X_train, y_train, X_val, y_val)
    
    plot_training_history(model.train_losses, model.val_losses, 'plots/training_history.png')
    
    print("\n" + "=" * 80)
    print("VALIDATION METRICS")
    print("=" * 80)
    
    y_val_pred = model.predict(X_val)
    val_metrics = calculate_metrics(y_val, y_val_pred)
    
    for metric, value in val_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    plot_metrics(y_val, y_val_pred, 'Validation', 'plots/validation_metrics.png')
    
    print("\n" + "=" * 80)
    print("TRAINING ON FULL DATASET")
    print("=" * 80)
    
    final_model = RidgeRegressionGD(
        learning_rate=0.01,
        n_iterations=1000,
        alpha=10.0
    )
    
    final_model.fit(X_train_full, y_train_full)
    
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)
    
    test_predictions = final_model.predict(X_test)
    
    submission = pd.DataFrame({
        'Hospital_Id': test_ids,
        'Transport_Cost': test_predictions
    })
    
    submission.to_csv('submission_ridge_regression.csv', index=False)
    print(f"\nSubmission file created: submission_ridge_regression.csv")
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

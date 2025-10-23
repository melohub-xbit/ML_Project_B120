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
# GRADIENT BOOSTING REGRESSION FROM SCRATCH
# ============================================================================

class DecisionTreeRegressor:
    """Simple decision tree for regression (used as weak learner in GB)"""
    def __init__(self, max_depth=3, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        n_samples = len(y)
        
        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return np.mean(y)
        
        # Find best split
        best_split = self._find_best_split(X, y)
        
        if best_split is None:
            return np.mean(y)
        
        feature_idx, threshold = best_split
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return np.mean(y)
        
        # Recursively build subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature_idx': feature_idx,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def _find_best_split(self, X, y):
        best_mse = float('inf')
        best_split = None
        n_features = X.shape[1]
        
        # Try random subset of features for efficiency
        feature_indices = np.random.choice(n_features, size=min(n_features, 10), replace=False)
        
        for feature_idx in feature_indices:
            thresholds = np.percentile(X[:, feature_idx], [25, 50, 75])
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                left_mse = np.var(y[left_mask]) * np.sum(left_mask)
                right_mse = np.var(y[right_mask]) * np.sum(right_mask)
                total_mse = left_mse + right_mse
                
                if total_mse < best_mse:
                    best_mse = total_mse
                    best_split = (feature_idx, threshold)
        
        return best_split
    
    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])
    
    def _predict_single(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        
        if x[tree['feature_idx']] <= tree['threshold']:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])


class GradientBoostingRegressor:
    """Gradient Boosting Regressor from scratch"""
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.initial_prediction = None
        self.train_losses = []
        self.val_losses = []
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Initial prediction (mean of target)
        self.initial_prediction = np.mean(y_train)
        predictions_train = np.full(len(y_train), self.initial_prediction)
        
        for i in range(self.n_estimators):
            # Compute residuals (negative gradient of MSE loss)
            residuals = y_train - predictions_train
            
            # Fit a tree to the residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth, 
                                        min_samples_split=self.min_samples_split)
            tree.fit(X_train, residuals)
            
            # Update predictions
            update = tree.predict(X_train)
            predictions_train += self.learning_rate * update
            
            # Store tree
            self.trees.append(tree)
            
            # Track losses
            train_loss = np.sqrt(mean_squared_error(y_train, predictions_train))
            self.train_losses.append(train_loss)
            
            if X_val is not None and y_val is not None:
                predictions_val = self.predict(X_val)
                val_loss = np.sqrt(mean_squared_error(y_val, predictions_val))
                self.val_losses.append(val_loss)
            
            if (i + 1) % 10 == 0:
                print(f"Iteration {i+1}/{self.n_estimators}, Train RMSE: {train_loss:.4f}", end="")
                if X_val is not None:
                    print(f", Val RMSE: {val_loss:.4f}")
                else:
                    print()
    
    def predict(self, X):
        predictions = np.full(len(X), self.initial_prediction)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions


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
    
    # Scale features
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
    """Plot training and validation losses"""
    plt.figure(figsize=(12, 5))
    
    # RMSE Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train RMSE', linewidth=2)
    plt.plot(val_losses, label='Validation RMSE', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title('Training and Validation RMSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning curve
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
    print("GRADIENT BOOSTING REGRESSION - FROM SCRATCH")
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
    
    # Train model
    print("\n" + "=" * 80)
    print("TRAINING GRADIENT BOOSTING MODEL")
    print("=" * 80)
    
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        min_samples_split=10
    )
    
    model.fit(X_train, y_train, X_val, y_val)
    
    # Plot training history
    plot_training_history(model.train_losses, model.val_losses, 'plots/training_history.png')
    
    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("VALIDATION METRICS")
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
    
    final_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        min_samples_split=10
    )
    
    final_model.fit(X_train_full, y_train_full)
    
    # Predict on test set
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)
    
    test_predictions = final_model.predict(X_test)
    
    # Create submission file
    submission = pd.DataFrame({
        'Hospital_Id': test_ids,
        'Transport_Cost': test_predictions
    })
    
    submission.to_csv('submission_gradient_boosting.csv', index=False)
    print(f"\nSubmission file created: submission_gradient_boosting.csv")
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

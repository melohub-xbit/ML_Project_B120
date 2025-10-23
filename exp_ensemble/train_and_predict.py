import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
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
        
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return np.mean(y)
        
        best_split = self._find_best_split(X, y)
        
        if best_split is None:
            return np.mean(y)
        
        feature_idx, threshold = best_split
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return np.mean(y)
        
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
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.initial_prediction = np.mean(y_train)
        predictions_train = np.full(len(y_train), self.initial_prediction)
        
        for i in range(self.n_estimators):
            residuals = y_train - predictions_train
            
            tree = DecisionTreeRegressor(max_depth=self.max_depth, 
                                        min_samples_split=self.min_samples_split)
            tree.fit(X_train, residuals)
            
            update = tree.predict(X_train)
            predictions_train += self.learning_rate * update
            
            self.trees.append(tree)
            
            if (i + 1) % 20 == 0:
                print(f"  GB Iteration {i+1}/{self.n_estimators}")
    
    def predict(self, X):
        predictions = np.full(len(X), self.initial_prediction)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions


# ============================================================================
# RIDGE REGRESSION WITH GRADIENT DESCENT
# ============================================================================

class RidgeRegressionGD:
    """Ridge Regression with Gradient Descent (L2 Regularization)"""
    def __init__(self, learning_rate=0.01, n_iterations=1000, alpha=1.0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.weights = None
        self.bias = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        n_samples, n_features = X_train.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iterations):
            y_pred = np.dot(X_train, self.weights) + self.bias
            
            dw = (-2 / n_samples) * np.dot(X_train.T, (y_train - y_pred)) + 2 * self.alpha * self.weights
            db = (-2 / n_samples) * np.sum(y_train - y_pred)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if (i + 1) % 200 == 0:
                print(f"  Ridge Iteration {i+1}/{self.n_iterations}")
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# ============================================================================
# SGD WITH POLYNOMIAL FEATURES
# ============================================================================

class SGDPolynomialRegression:
    """Stochastic Gradient Descent Regression with Polynomial Features"""
    def __init__(self, learning_rate=0.01, n_epochs=100, batch_size=32, alpha=1.0, degree=2):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.degree = degree
        self.weights = None
        self.bias = None
        self.poly = None
    
    def _create_polynomial_features(self, X):
        if self.poly is None:
            self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
            return self.poly.fit_transform(X)
        else:
            return self.poly.transform(X)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train_poly = self._create_polynomial_features(X_train)
        
        n_samples, n_features = X_train_poly.shape
        
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
        
        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train_poly[indices]
            y_shuffled = y_train[indices]
            
            for i in range(0, n_samples, self.batch_size):
                batch_end = min(i + self.batch_size, n_samples)
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                batch_size_actual = batch_end - i
                
                y_pred = np.dot(X_batch, self.weights) + self.bias
                
                error = y_batch - y_pred
                dw = (-2 / batch_size_actual) * np.dot(X_batch.T, error) + 2 * self.alpha * self.weights
                db = (-2 / batch_size_actual) * np.sum(error)
                
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
            if (epoch + 1) % 20 == 0:
                print(f"  SGD Epoch {epoch+1}/{self.n_epochs}")
    
    def predict(self, X):
        X_poly = self._create_polynomial_features(X)
        return np.dot(X_poly, self.weights) + self.bias


# ============================================================================
# ENSEMBLE MODEL
# ============================================================================

class EnsembleRegressor:
    """Ensemble of Gradient Boosting, Ridge, and SGD Polynomial models"""
    def __init__(self, gb_params, ridge_params, sgd_params):
        self.gb_model = GradientBoostingRegressor(**gb_params)
        self.ridge_model = RidgeRegressionGD(**ridge_params)
        self.sgd_model = SGDPolynomialRegression(**sgd_params)
        self.weights = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        print("Training Gradient Boosting Model...")
        self.gb_model.fit(X_train, y_train, X_val, y_val)
        
        print("\nTraining Ridge Regression Model...")
        self.ridge_model.fit(X_train, y_train, X_val, y_val)
        
        print("\nTraining SGD Polynomial Model...")
        self.sgd_model.fit(X_train, y_train, X_val, y_val)
        
        # Optimize ensemble weights on validation set
        if X_val is not None and y_val is not None:
            print("\nOptimizing ensemble weights...")
            gb_pred = self.gb_model.predict(X_val)
            ridge_pred = self.ridge_model.predict(X_val)
            sgd_pred = self.sgd_model.predict(X_val)
            
            # Simple grid search for optimal weights
            best_rmse = float('inf')
            best_weights = [1/3, 1/3, 1/3]
            
            for w1 in np.linspace(0.1, 0.6, 6):
                for w2 in np.linspace(0.1, 0.6, 6):
                    w3 = 1.0 - w1 - w2
                    if w3 < 0.1 or w3 > 0.6:
                        continue
                    
                    ensemble_pred = w1 * gb_pred + w2 * ridge_pred + w3 * sgd_pred
                    rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
                    
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_weights = [w1, w2, w3]
            
            self.weights = best_weights
            print(f"Optimal weights: GB={self.weights[0]:.3f}, Ridge={self.weights[1]:.3f}, SGD={self.weights[2]:.3f}")
            print(f"Ensemble Validation RMSE: {best_rmse:.4f}")
        else:
            self.weights = [1/3, 1/3, 1/3]
    
    def predict(self, X):
        gb_pred = self.gb_model.predict(X)
        ridge_pred = self.ridge_model.predict(X)
        sgd_pred = self.sgd_model.predict(X)
        
        return self.weights[0] * gb_pred + self.weights[1] * ridge_pred + self.weights[2] * sgd_pred


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def preprocess_data(train_df, test_df):
    """Preprocess training and test data"""
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

def plot_model_comparison(y_val, gb_pred, ridge_pred, sgd_pred, ensemble_pred, save_path='plots/model_comparison.png'):
    """Compare predictions from all models"""
    models = ['Gradient Boosting', 'Ridge Regression', 'SGD Polynomial', 'Ensemble']
    predictions = [gb_pred, ridge_pred, sgd_pred, ensemble_pred]
    
    plt.figure(figsize=(16, 10))
    
    for idx, (model_name, y_pred) in enumerate(zip(models, predictions)):
        # Predictions vs Actual
        plt.subplot(3, 4, idx + 1)
        plt.scatter(y_val, y_pred, alpha=0.5, s=20)
        plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{model_name}\nPredictions vs Actual')
        plt.grid(True, alpha=0.3)
        
        # Residuals
        plt.subplot(3, 4, idx + 5)
        residuals = y_val - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5, s=20)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title(f'{model_name}\nResidual Plot')
        plt.grid(True, alpha=0.3)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        plt.subplot(3, 4, idx + 9)
        plt.axis('off')
        metrics_text = f'{model_name}\n\n'
        metrics_text += f'RMSE: {rmse:.2f}\n'
        metrics_text += f'MAE: {mae:.2f}\n'
        metrics_text += f'R²: {r2:.4f}'
        plt.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved model comparison plot to {save_path}")
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
    print("ENSEMBLE REGRESSION - COMBINING 3 MODELS")
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
    print("TRAINING ENSEMBLE MODEL")
    print("=" * 80)
    
    gb_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 4, 'min_samples_split': 10}
    ridge_params = {'learning_rate': 0.01, 'n_iterations': 1000, 'alpha': 10.0}
    sgd_params = {'learning_rate': 0.01, 'n_epochs': 100, 'batch_size': 64, 'alpha': 1.0, 'degree': 2}
    
    ensemble = EnsembleRegressor(gb_params, ridge_params, sgd_params)
    ensemble.fit(X_train, y_train, X_val, y_val)
    
    # Get individual predictions for comparison
    gb_val_pred = ensemble.gb_model.predict(X_val)
    ridge_val_pred = ensemble.ridge_model.predict(X_val)
    sgd_val_pred = ensemble.sgd_model.predict(X_val)
    ensemble_val_pred = ensemble.predict(X_val)
    
    # Plot comparison
    plot_model_comparison(y_val, gb_val_pred, ridge_val_pred, sgd_val_pred, ensemble_val_pred, 
                         'plots/model_comparison.png')
    
    print("\n" + "=" * 80)
    print("VALIDATION METRICS - ALL MODELS")
    print("=" * 80)
    
    for model_name, pred in [('Gradient Boosting', gb_val_pred), 
                              ('Ridge Regression', ridge_val_pred),
                              ('SGD Polynomial', sgd_val_pred),
                              ('Ensemble', ensemble_val_pred)]:
        print(f"\n{model_name}:")
        metrics = calculate_metrics(y_val, pred)
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Train on full dataset
    print("\n" + "=" * 80)
    print("TRAINING ON FULL DATASET")
    print("=" * 80)
    
    final_ensemble = EnsembleRegressor(gb_params, ridge_params, sgd_params)
    final_ensemble.fit(X_train_full, y_train_full)
    final_ensemble.weights = ensemble.weights  # Use optimized weights
    
    # Generate predictions
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)
    
    test_predictions = final_ensemble.predict(X_test)
    
    submission = pd.DataFrame({
        'Hospital_Id': test_ids,
        'Transport_Cost': test_predictions
    })
    
    submission.to_csv('submission_ensemble.csv', index=False)
    print(f"\nSubmission file created: submission_ensemble.csv")
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

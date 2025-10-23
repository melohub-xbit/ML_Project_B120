"""
Custom Neural Network Regression Model implemented from scratch
Uses only numpy for core operations
"""

import numpy as np
import pickle
from datetime import datetime


class NeuralNetworkRegression:
    """
    Multi-layer neural network for regression implemented from scratch
    """
    
    def __init__(self, input_dim, hidden_layers=[128, 64, 32], learning_rate=0.001, random_seed=42):
        """
        Initialize the neural network
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            learning_rate: Learning rate for optimization
            random_seed: Random seed for reproducibility
        """
        np.random.seed(random_seed)
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        
        # Build network architecture
        self.layer_dims = [input_dim] + hidden_layers + [1]
        self.num_layers = len(self.layer_dims) - 1
        
        # Initialize weights and biases
        self.weights = {}
        self.biases = {}
        self._initialize_parameters()
        
        # Adam optimizer parameters
        self.m_w = {}  # First moment for weights
        self.v_w = {}  # Second moment for weights
        self.m_b = {}  # First moment for biases
        self.v_b = {}  # Second moment for biases
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0  # Time step
        
        self._initialize_adam()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'train_rmse': [],
            'val_rmse': [],
            'train_r2': [],
            'val_r2': []
        }
    
    def _initialize_parameters(self):
        """Initialize weights using He initialization and biases to zero"""
        for l in range(1, self.num_layers + 1):
            # He initialization for weights
            self.weights[l] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2.0 / self.layer_dims[l-1])
            self.biases[l] = np.zeros((self.layer_dims[l], 1))
    
    def _initialize_adam(self):
        """Initialize Adam optimizer moments"""
        for l in range(1, self.num_layers + 1):
            self.m_w[l] = np.zeros_like(self.weights[l])
            self.v_w[l] = np.zeros_like(self.weights[l])
            self.m_b[l] = np.zeros_like(self.biases[l])
            self.v_b[l] = np.zeros_like(self.biases[l])
    
    def relu(self, Z):
        """ReLU activation function"""
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        """Derivative of ReLU"""
        return (Z > 0).astype(float)
    
    def forward_propagation(self, X):
        """
        Forward pass through the network
        
        Args:
            X: Input data (features x samples)
            
        Returns:
            predictions and cache for backpropagation
        """
        cache = {'A0': X}
        A = X
        
        for l in range(1, self.num_layers + 1):
            Z = np.dot(self.weights[l], A) + self.biases[l]
            
            if l < self.num_layers:
                # Hidden layers use ReLU
                A = self.relu(Z)
            else:
                # Output layer is linear for regression
                A = Z
            
            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A
        
        return A, cache
    
    def compute_loss(self, y_true, y_pred):
        """Compute Mean Squared Error loss"""
        m = y_true.shape[1]
        loss = np.sum((y_pred - y_true) ** 2) / (2 * m)
        return loss
    
    def backward_propagation(self, y_true, cache):
        """
        Backward pass to compute gradients
        
        Args:
            y_true: True target values
            cache: Cache from forward propagation
            
        Returns:
            gradients dictionary
        """
        m = y_true.shape[1]
        gradients = {}
        
        # Output layer gradient
        dA = cache[f'A{self.num_layers}'] - y_true
        
        # Backpropagate through layers
        for l in reversed(range(1, self.num_layers + 1)):
            dZ = dA
            if l < self.num_layers:
                # Apply ReLU derivative for hidden layers
                dZ = dA * self.relu_derivative(cache[f'Z{l}'])
            
            # Compute gradients
            gradients[f'dW{l}'] = np.dot(dZ, cache[f'A{l-1}'].T) / m
            gradients[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True) / m
            
            if l > 1:
                # Gradient for next layer
                dA = np.dot(self.weights[l].T, dZ)
        
        return gradients
    
    def update_parameters(self, gradients):
        """Update parameters using Adam optimizer"""
        self.t += 1
        
        for l in range(1, self.num_layers + 1):
            # Update weights
            self.m_w[l] = self.beta1 * self.m_w[l] + (1 - self.beta1) * gradients[f'dW{l}']
            self.v_w[l] = self.beta2 * self.v_w[l] + (1 - self.beta2) * (gradients[f'dW{l}'] ** 2)
            
            m_w_corrected = self.m_w[l] / (1 - self.beta1 ** self.t)
            v_w_corrected = self.v_w[l] / (1 - self.beta2 ** self.t)
            
            self.weights[l] -= self.learning_rate * m_w_corrected / (np.sqrt(v_w_corrected) + self.epsilon)
            
            # Update biases
            self.m_b[l] = self.beta1 * self.m_b[l] + (1 - self.beta1) * gradients[f'db{l}']
            self.v_b[l] = self.beta2 * self.v_b[l] + (1 - self.beta2) * (gradients[f'db{l}'] ** 2)
            
            m_b_corrected = self.m_b[l] / (1 - self.beta1 ** self.t)
            v_b_corrected = self.v_b[l] / (1 - self.beta2 ** self.t)
            
            self.biases[l] -= self.learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon)
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input features (samples x features)
            
        Returns:
            predictions (samples,)
        """
        # Transpose for forward prop
        X_T = X.T
        predictions, _ = self.forward_propagation(X_T)
        return predictions.flatten()
    
    def compute_metrics(self, y_true, y_pred):
        """
        Compute regression metrics
        
        Returns:
            Dictionary of metrics
        """
        # MSE
        mse = np.mean((y_true - y_pred) ** 2)
        
        # RMSE
        rmse = np.sqrt(mse)
        
        # MAE
        mae = np.mean(np.abs(y_true - y_pred))
        
        # R² Score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def train_epoch(self, X_train, y_train, batch_size):
        """Train for one epoch using mini-batch gradient descent"""
        m = X_train.shape[0]
        indices = np.random.permutation(m)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size].T
            y_batch = y_shuffled[i:i+batch_size].reshape(1, -1)
            
            # Forward propagation
            y_pred, cache = self.forward_propagation(X_batch)
            
            # Compute loss
            loss = self.compute_loss(y_batch, y_pred)
            epoch_loss += loss
            num_batches += 1
            
            # Backward propagation
            gradients = self.backward_propagation(y_batch, cache)
            
            # Update parameters
            self.update_parameters(gradients)
        
        return epoch_loss / num_batches
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, 
            early_stopping_patience=20, min_delta=0.0001, verbose=True):
        """
        Train the neural network
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch gradient descent
            early_stopping_patience: Patience for early stopping
            min_delta: Minimum change for early stopping
            verbose: Print training progress
        """
        print(f"\nStarting training...")
        print(f"Model architecture: {self.layer_dims}")
        print(f"Training samples: {X_train.shape[0]}")
        if X_val is not None:
            print(f"Validation samples: {X_val.shape[0]}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print("="*70)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train for one epoch
            train_loss = self.train_epoch(X_train, y_train, batch_size)
            
            # Make predictions for metrics
            train_pred = self.predict(X_train)
            train_metrics = self.compute_metrics(y_train, train_pred)
            
            # Store training metrics
            self.history['train_loss'].append(train_loss)
            self.history['train_mae'].append(train_metrics['mae'])
            self.history['train_rmse'].append(train_metrics['rmse'])
            self.history['train_r2'].append(train_metrics['r2'])
            
            # Validation metrics
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_metrics = self.compute_metrics(y_val, val_pred)
                val_loss = val_metrics['mse'] / 2
                
                self.history['val_loss'].append(val_loss)
                self.history['val_mae'].append(val_metrics['mae'])
                self.history['val_rmse'].append(val_metrics['rmse'])
                self.history['val_r2'].append(val_metrics['r2'])
                
                # Early stopping check
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train - Loss: {train_loss:.4f}, MAE: {train_metrics['mae']:.4f}, "
                      f"RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}")
                if X_val is not None:
                    print(f"  Val   - Loss: {val_loss:.4f}, MAE: {val_metrics['mae']:.4f}, "
                          f"RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}")
                print()
        
        print("="*70)
        print("Training completed!")
        if X_val is not None:
            print(f"Best validation loss: {best_val_loss:.4f}")
    
    def save(self, filepath):
        """Save model weights and configuration"""
        model_state = {
            'weights': self.weights,
            'biases': self.biases,
            'layer_dims': self.layer_dims,
            'history': self.history,
            'learning_rate': self.learning_rate
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model weights and configuration"""
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)
        
        self.weights = model_state['weights']
        self.biases = model_state['biases']
        self.layer_dims = model_state['layer_dims']
        self.history = model_state['history']
        self.learning_rate = model_state['learning_rate']
        self.num_layers = len(self.layer_dims) - 1
        
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Test the model with dummy data
    print("Testing Neural Network Regression Model...")
    
    # Create dummy data
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.sum(X, axis=1) + np.random.randn(100) * 0.1
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Create and train model
    model = NeuralNetworkRegression(input_dim=10, hidden_layers=[32, 16], learning_rate=0.01)
    model.fit(X_train, y_train, X_val, y_val, epochs=50, batch_size=16)
    
    # Test predictions
    predictions = model.predict(X_val)
    metrics = model.compute_metrics(y_val, predictions)
    
    print("\nTest Metrics:")
    for key, value in metrics.items():
        print(f"  {key.upper()}: {value:.4f}")

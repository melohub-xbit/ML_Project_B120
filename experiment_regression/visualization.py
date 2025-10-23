"""
Visualization utilities for training metrics
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import config


def plot_training_history(history, save_dir='plots'):
    """
    Plot training and validation metrics
    
    Args:
        history: Dictionary containing training history
        save_dir: Directory to save plots
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    has_val = len(history['val_loss']) > 0
    
    # Plot 1: Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    if has_val:
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss (MSE/2)', fontsize=12)
    axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: MAE
    axes[0, 1].plot(epochs, history['train_mae'], 'b-', label='Training MAE', linewidth=2)
    if has_val:
        axes[0, 1].plot(epochs, history['val_mae'], 'r-', label='Validation MAE', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Mean Absolute Error', fontsize=12)
    axes[0, 1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: RMSE
    axes[1, 0].plot(epochs, history['train_rmse'], 'b-', label='Training RMSE', linewidth=2)
    if has_val:
        axes[1, 0].plot(epochs, history['val_rmse'], 'r-', label='Validation RMSE', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Root Mean Squared Error', fontsize=12)
    axes[1, 0].set_title('Root Mean Squared Error', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: R² Score
    axes[1, 1].plot(epochs, history['train_r2'], 'b-', label='Training R²', linewidth=2)
    if has_val:
        axes[1, 1].plot(epochs, history['val_r2'], 'r-', label='Validation R²', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('R² Score', fontsize=12)
    axes[1, 1].set_title('R² Score', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Perfect Score')
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    
    plt.close()


def plot_predictions_vs_actual(y_true, y_pred, dataset_name='Validation', save_dir='plots'):
    """
    Plot predicted vs actual values
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        dataset_name: Name of the dataset
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5, s=50, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Transport Cost', fontsize=12)
    plt.ylabel('Predicted Transport Cost', fontsize=12)
    plt.title(f'{dataset_name} Set: Predicted vs Actual', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add metrics as text
    from regression_model import NeuralNetworkRegression
    model = NeuralNetworkRegression(input_dim=1)  # Dummy instance for metrics
    metrics = model.compute_metrics(y_true, y_pred)
    
    textstr = f"R² = {metrics['r2']:.4f}\nRMSE = {metrics['rmse']:.2f}\nMAE = {metrics['mae']:.2f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{dataset_name.lower()}_predictions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"{dataset_name} predictions plot saved to {save_path}")
    
    plt.close()


def plot_residuals(y_true, y_pred, dataset_name='Validation', save_dir='plots'):
    """
    Plot residuals (errors) distribution
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        dataset_name: Name of the dataset
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'{dataset_name} Set: Residual Analysis', fontsize=14, fontweight='bold')
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=50, edgecolors='k', linewidth=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Transport Cost', fontsize=12)
    axes[0].set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    axes[0].set_title('Residual Plot', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Residuals Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    textstr = f'Mean: {mean_residual:.2f}\nStd: {std_residual:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    axes[1].text(0.05, 0.95, textstr, transform=axes[1].transAxes, fontsize=11,
                 verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{dataset_name.lower()}_residuals.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"{dataset_name} residuals plot saved to {save_path}")
    
    plt.close()


def plot_all_visualizations(history, y_train, train_pred, y_val, val_pred, save_dir='plots'):
    """
    Generate all visualization plots
    
    Args:
        history: Training history dictionary
        y_train: True training targets
        train_pred: Training predictions
        y_val: True validation targets
        val_pred: Validation predictions
        save_dir: Directory to save plots
    """
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)
    
    # Training history
    plot_training_history(history, save_dir)
    
    # Predictions vs Actual
    plot_predictions_vs_actual(y_train, train_pred, 'Training', save_dir)
    plot_predictions_vs_actual(y_val, val_pred, 'Validation', save_dir)
    
    # Residuals
    plot_residuals(y_train, train_pred, 'Training', save_dir)
    plot_residuals(y_val, val_pred, 'Validation', save_dir)
    
    print("\nAll visualizations generated successfully!")
    print("="*50)


if __name__ == "__main__":
    # Test visualization with dummy data
    print("Testing visualization functions...")
    
    # Create dummy history
    history = {
        'train_loss': [10.0 - i*0.1 for i in range(50)],
        'val_loss': [11.0 - i*0.1 for i in range(50)],
        'train_mae': [5.0 - i*0.05 for i in range(50)],
        'val_mae': [5.5 - i*0.05 for i in range(50)],
        'train_rmse': [7.0 - i*0.07 for i in range(50)],
        'val_rmse': [7.5 - i*0.07 for i in range(50)],
        'train_r2': [0.5 + i*0.01 for i in range(50)],
        'val_r2': [0.45 + i*0.01 for i in range(50)]
    }
    
    # Create dummy predictions
    np.random.seed(42)
    y_true = np.random.randn(100) * 100 + 500
    y_pred = y_true + np.random.randn(100) * 20
    
    plot_training_history(history)
    plot_predictions_vs_actual(y_true, y_pred)
    plot_residuals(y_true, y_pred)
    
    print("Visualization test completed!")

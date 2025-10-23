"""
Training script for the regression model
"""

import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

# Import local modules
import config
from data_preprocessing import DataPreprocessor
from regression_model import NeuralNetworkRegression
from visualization import plot_all_visualizations


def main():
    """Main training function"""
    print("\n" + "="*70)
    print(" " * 15 + "MEDICAL EQUIPMENT TRANSPORT COST PREDICTION")
    print(" " * 20 + "NEURAL NETWORK REGRESSION MODEL")
    print("="*70)
    
    # Set random seed for reproducibility
    np.random.seed(config.RANDOM_SEED)
    
    # Step 1: Load and preprocess data
    print("\n[STEP 1/5] DATA PREPROCESSING")
    print("-"*70)
    
    preprocessor = DataPreprocessor()
    X, y, indices = preprocessor.preprocess_train_data(config.TRAIN_DATA_PATH)
    
    # Step 2: Split data into train and validation
    print("\n[STEP 2/5] SPLITTING DATA")
    print("-"*70)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=config.VALIDATION_SPLIT,
        random_state=config.RANDOM_SEED
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Feature dimensions: {X_train.shape[1]}")
    
    # Step 3: Initialize and train model
    print("\n[STEP 3/5] MODEL TRAINING")
    print("-"*70)
    
    model = NeuralNetworkRegression(
        input_dim=X_train.shape[1],
        hidden_layers=config.HIDDEN_LAYERS,
        learning_rate=config.LEARNING_RATE,
        random_seed=config.RANDOM_SEED
    )
    
    model.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.MIN_DELTA,
        verbose=True
    )
    
    # Step 4: Evaluate model
    print("\n[STEP 4/5] MODEL EVALUATION")
    print("-"*70)
    
    # Training set evaluation
    train_pred = model.predict(X_train)
    train_metrics = model.compute_metrics(y_train, train_pred)
    
    print("\nTraining Set Metrics:")
    print(f"  MSE:  {train_metrics['mse']:.4f}")
    print(f"  RMSE: {train_metrics['rmse']:.4f}")
    print(f"  MAE:  {train_metrics['mae']:.4f}")
    print(f"  R²:   {train_metrics['r2']:.4f}")
    
    # Validation set evaluation
    val_pred = model.predict(X_val)
    val_metrics = model.compute_metrics(y_val, val_pred)
    
    print("\nValidation Set Metrics:")
    print(f"  MSE:  {val_metrics['mse']:.4f}")
    print(f"  RMSE: {val_metrics['rmse']:.4f}")
    print(f"  MAE:  {val_metrics['mae']:.4f}")
    print(f"  R²:   {val_metrics['r2']:.4f}")
    
    # Step 5: Save model and preprocessor
    print("\n[STEP 5/5] SAVING MODEL AND PREPROCESSOR")
    print("-"*70)
    
    model.save(config.MODEL_SAVE_PATH)
    preprocessor.save('preprocessor.pkl')
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_all_visualizations(
        model.history,
        y_train, train_pred,
        y_val, val_pred,
        save_dir=config.PLOTS_DIR
    )
    
    print("\n" + "="*70)
    print(" " * 25 + "TRAINING COMPLETED!")
    print("="*70)
    print(f"\nModel saved to: {config.MODEL_SAVE_PATH}")
    print(f"Preprocessor saved to: preprocessor.pkl")
    print(f"Plots saved to: {config.PLOTS_DIR}/")
    print("\nYou can now run predict.py to generate predictions on test data.")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: {str(e)}")
        print(f"{'='*70}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

"""
Prediction script for generating submission file
"""

import numpy as np
import pandas as pd
import os
import sys

# Import local modules
import config
from data_preprocessing import DataPreprocessor
from regression_model import NeuralNetworkRegression


def main():
    """Main prediction function"""
    print("\n" + "="*70)
    print(" " * 15 + "GENERATING PREDICTIONS FOR TEST DATA")
    print("="*70)
    
    # Check if model and preprocessor exist
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"\nERROR: Model file not found: {config.MODEL_SAVE_PATH}")
        print("Please run train.py first to train the model.")
        sys.exit(1)
    
    if not os.path.exists('preprocessor.pkl'):
        print(f"\nERROR: Preprocessor file not found: preprocessor.pkl")
        print("Please run train.py first to train the model.")
        sys.exit(1)
    
    # Step 1: Load preprocessor and model
    print("\n[STEP 1/4] LOADING MODEL AND PREPROCESSOR")
    print("-"*70)
    
    preprocessor = DataPreprocessor()
    preprocessor.load('preprocessor.pkl')
    
    model = NeuralNetworkRegression(input_dim=1)  # Will be overwritten by load
    model.load(config.MODEL_SAVE_PATH)
    
    print("Model and preprocessor loaded successfully!")
    
    # Step 2: Load and preprocess test data
    print("\n[STEP 2/4] PREPROCESSING TEST DATA")
    print("-"*70)
    
    X_test, test_indices = preprocessor.preprocess_test_data(config.TEST_DATA_PATH)
    
    print(f"Test samples: {len(X_test)}")
    print(f"Feature dimensions: {X_test.shape[1]}")
    
    # Step 3: Generate predictions
    print("\n[STEP 3/4] GENERATING PREDICTIONS")
    print("-"*70)
    
    predictions = model.predict(X_test)
    
    print(f"Generated {len(predictions)} predictions")
    print(f"Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    print(f"Mean prediction: {predictions.mean():.2f}")
    print(f"Median prediction: {np.median(predictions):.2f}")
    
    # Step 4: Create submission file
    print("\n[STEP 4/4] CREATING SUBMISSION FILE")
    print("-"*70)
    
    # Load sample submission to verify format
    sample_submission = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
    print(f"\nSample submission format:")
    print(f"  Columns: {list(sample_submission.columns)}")
    print(f"  Shape (sample): {sample_submission.shape}")
    
    # Create submission dataframe
    submission = pd.DataFrame({
        config.INDEX_COLUMN: test_indices,
        config.TARGET_COLUMN: predictions
    })
    
    # Verify submission format
    print(f"\nGenerated submission format:")
    print(f"  Columns: {list(submission.columns)}")
    print(f"  Shape: {submission.shape}")
    
    # Verify shape matches requirements (500 x 2)
    expected_shape = (500, 2)
    if submission.shape != expected_shape:
        print(f"\nWARNING: Submission shape {submission.shape} does not match expected {expected_shape}")
    else:
        print(f"\n✓ Submission shape matches requirements: {expected_shape}")
    
    # Save submission
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    
    print(f"\n{'='*70}")
    print(f" " * 20 + "PREDICTIONS COMPLETED!")
    print(f"{'='*70}")
    print(f"\nSubmission file saved to: {config.SUBMISSION_PATH}")
    print(f"File size: {submission.shape[0]} rows x {submission.shape[1]} columns")
    print(f"\nFirst few predictions:")
    print(submission.head(10).to_string(index=False))
    print(f"{'='*70}\n")


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

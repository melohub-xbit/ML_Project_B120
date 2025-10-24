# XGBoost Hyperparameter Tuning

This folder contains the hyperparameter tuning script for XGBoost using Optuna and K-Fold Cross-Validation.

## Requirements

Make sure you have the following packages installed:

```bash
pip install optuna xgboost scikit-learn pandas numpy category-encoders
```

## How to Run

1. Make sure the preprocessing module exists in `../preprocessing/preprocess_data.py`
2. Make sure the dataset exists in `../dataset/` (train.csv and test.csv)
3. Run the tuning script:

```bash
python tune_xgboost.py
```

## What it Does

1. **Data Loading**: Runs preprocessing and loads the data (WITHOUT applying TargetEncoder)
2. **Hyperparameter Tuning**: Uses Optuna to search for optimal hyperparameters with 5-Fold Cross-Validation
3. **Parameter Space Tuned**:
   - `learning_rate`: 0.01 to 0.3
   - `max_depth`: 4 to 10
   - `subsample`: 0.6 to 1.0
   - `colsample_bytree`: 0.6 to 1.0
   - `min_child_weight`: 1 to 10
   - `reg_alpha`: 1e-3 to 10.0 (L1 regularization, log scale)
   - `reg_lambda`: 1e-3 to 10.0 (L2 regularization, log scale)
   - `smoothing`: 1.0 to 5.0 (for TargetEncoder)
4. **Critical Feature**: TargetEncoder is applied INSIDE each CV fold to prevent data leakage
5. **Saves Results**: Best parameters saved to `./tuning_results/` in both JSON and TXT formats
6. **Final Training**: 
   - First trains on 90% with validation for early stopping
   - Then re-trains on 100% of data using the best iteration count
7. **Predictions**: Generates `submission_xgboost_tuned.csv`

## Output Files

- `./tuning_results/best_params_xgboost.json` - Best parameters in JSON format
- `./tuning_results/best_params_xgboost.txt` - Best parameters in human-readable format
- `submission_xgboost_tuned.csv` - Final predictions
- `./processed_data/` - Preprocessed data files

## Important Notes

- Default number of trials: 50 (can be adjusted in the script)
- **TargetEncoder is fitted inside each CV fold** - this is critical to prevent data leakage
- Early stopping is set in model parameters (`early_stopping_rounds=100`)
- Final model is re-trained on full data after determining best iteration count

## Why TargetEncoder Inside CV Loop?

TargetEncoder uses the target variable to encode categorical features. If we fit it on the full dataset before splitting, the validation fold would have information from the training fold, leading to overfitting and unrealistic CV scores.

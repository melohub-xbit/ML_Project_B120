# Random Forest Hyperparameter Tuning

This folder contains the hyperparameter tuning script for Random Forest using Optuna and K-Fold Cross-Validation.

## Requirements

Make sure you have the following packages installed:

```bash
pip install optuna scikit-learn pandas numpy category-encoders
```

## How to Run

1. Make sure the preprocessing module exists in `../preprocessing/preprocess_data.py`
2. Make sure the dataset exists in `../dataset/` (train.csv and test.csv)
3. Run the tuning script:

```bash
python tune_random_forest.py
```

## What it Does

1. **Data Loading**: Runs preprocessing and loads the data (WITHOUT applying TargetEncoder)
2. **Hyperparameter Tuning**: Uses Optuna to search for optimal hyperparameters with 5-Fold Cross-Validation
3. **Parameter Space Tuned**:
   - `n_estimators`: 100 to 1000 (number of trees)
   - `max_depth`: 10 to 30
   - `min_samples_split`: 2 to 20
   - `min_samples_leaf`: 1 to 10
   - `max_features`: 0.6 to 1.0 (fraction of features to consider)
   - `smoothing`: 1.0 to 5.0 (for TargetEncoder)
4. **Critical Feature**: TargetEncoder is applied INSIDE each CV fold to prevent data leakage
5. **Saves Results**: Best parameters saved to `./tuning_results/` in both JSON and TXT formats
6. **Final Training**: Trains a final model with best parameters on full dataset
7. **Predictions**: Generates `submission_random_forest_tuned.csv`

## Output Files

- `./tuning_results/best_params_random_forest.json` - Best parameters in JSON format
- `./tuning_results/best_params_random_forest.txt` - Best parameters in human-readable format
- `submission_random_forest_tuned.csv` - Final predictions
- `./processed_data/` - Preprocessed data files

## Important Notes

- Default number of trials: 30 (fewer than XGBoost/CatBoost because RF is slower)
- **TargetEncoder is fitted inside each CV fold** - this is critical to prevent data leakage
- Random Forest doesn't use early stopping, so we train on 100% of data directly
- All CPU cores are used (`n_jobs=-1`) for faster training

## Why Fewer Trials?

Random Forest is computationally more expensive than gradient boosting methods, especially with many trees. We use 30 trials instead of 50 to balance thoroughness with runtime. You can increase this if you have more computational resources.

## Why TargetEncoder Inside CV Loop?

TargetEncoder uses the target variable to encode categorical features. If we fit it on the full dataset before splitting, the validation fold would have information from the training fold, leading to overfitting and unrealistic CV scores.

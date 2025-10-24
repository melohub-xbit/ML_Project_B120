# CatBoost Hyperparameter Tuning

This folder contains the hyperparameter tuning script for CatBoost using Optuna and K-Fold Cross-Validation.

## Requirements

Make sure you have the following packages installed:

```bash
pip install optuna catboost scikit-learn pandas numpy
```

## How to Run

1. Make sure the preprocessing module exists in `../preprocessing/preprocess_data.py`
2. Make sure the dataset exists in `../dataset/` (train.csv and test.csv)
3. Run the tuning script:

```bash
python tune_catboost.py
```

## What it Does

1. **Data Loading**: Runs preprocessing and loads the data
2. **Hyperparameter Tuning**: Uses Optuna to search for optimal hyperparameters with 5-Fold Cross-Validation
3. **Parameter Space Tuned**:
   - `learning_rate`: 0.01 to 0.3
   - `depth`: 4 to 10
   - `l2_leaf_reg`: 1.0 to 10.0 (log scale)
   - `bagging_temperature`: 0.0 to 1.0
   - `random_strength`: 0.1 to 1.0
4. **Saves Results**: Best parameters saved to `./tuning_results/` in both JSON and TXT formats
5. **Final Training**: Trains a final model with best parameters on full dataset
6. **Predictions**: Generates `submission_catboost_tuned.csv`

## Output Files

- `./tuning_results/best_params_catboost.json` - Best parameters in JSON format
- `./tuning_results/best_params_catboost.txt` - Best parameters in human-readable format
- `submission_catboost_tuned.csv` - Final predictions
- `./processed_data/` - Preprocessed data files

## Notes

- Default number of trials: 50 (can be adjusted in the script)
- CatBoost handles categorical features internally, no need for manual encoding
- Early stopping is used during cross-validation (100 rounds)
- Final model trains for 5000 iterations

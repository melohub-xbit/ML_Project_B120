# Hyperparameter Tuning Guide

This document provides an overview of the hyperparameter tuning process for the three best-performing models.

## Overview

Three separate tuning folders have been created for hyperparameter optimization:

1. **tune_catboost/** - CatBoost hyperparameter tuning
2. **tune_xgboost/** - XGBoost hyperparameter tuning  
3. **tune_random_forest/** - Random Forest hyperparameter tuning

Each uses **Optuna** for hyperparameter search combined with **5-Fold Cross-Validation** for robust performance estimation.

## Installation

Before running any tuning scripts, install the required packages:

```bash
pip install optuna catboost xgboost scikit-learn pandas numpy category-encoders
```

## Quick Start

Navigate to each tuning folder and run:

```bash
# CatBoost
cd tune_catboost
python tune_catboost.py

# XGBoost
cd tune_xgboost
python tune_xgboost.py

# Random Forest
cd tune_random_forest
python tune_random_forest.py
```

## What Gets Tuned?

### CatBoost Parameters
- `learning_rate`: Learning step size (0.01 - 0.3)
- `depth`: Tree depth (4 - 10)
- `l2_leaf_reg`: L2 regularization (1.0 - 10.0)
- `bagging_temperature`: Bagging randomness (0.0 - 1.0)
- `random_strength`: Feature split randomness (0.1 - 1.0)

### XGBoost Parameters
- `learning_rate`: Learning step size (0.01 - 0.3)
- `max_depth`: Tree depth (4 - 10)
- `subsample`: Row sampling ratio (0.6 - 1.0)
- `colsample_bytree`: Column sampling ratio (0.6 - 1.0)
- `min_child_weight`: Minimum sum of instance weight (1 - 10)
- `reg_alpha`: L1 regularization (1e-3 - 10.0)
- `reg_lambda`: L2 regularization (1e-3 - 10.0)
- `smoothing`: TargetEncoder smoothing (1.0 - 5.0)

### Random Forest Parameters
- `n_estimators`: Number of trees (100 - 1000)
- `max_depth`: Tree depth (10 - 30)
- `min_samples_split`: Minimum samples to split (2 - 20)
- `min_samples_leaf`: Minimum samples in leaf (1 - 10)
- `max_features`: Feature fraction per split (0.6 - 1.0)
- `smoothing`: TargetEncoder smoothing (1.0 - 5.0)

## Key Features

### 1. K-Fold Cross-Validation
All models use 5-fold cross-validation to get a robust estimate of model performance and prevent overfitting to a single train/validation split.

### 2. Preventing Data Leakage
**Critical for XGBoost and Random Forest**: TargetEncoder is fitted inside each CV fold separately. This prevents target information from the validation fold leaking into the training fold.

**CatBoost** handles categorical features internally, so no TargetEncoder is needed.

### 3. Early Stopping
- **CatBoost**: Uses early stopping during CV (100 rounds)
- **XGBoost**: Uses early stopping during CV and final training (100 rounds)
- **Random Forest**: No early stopping (not applicable)

### 4. Automatic Parameter Saving
After tuning completes, best parameters are automatically saved in two formats:

- **JSON**: `./tuning_results/best_params_<model>.json` - Machine-readable
- **TXT**: `./tuning_results/best_params_<model>.txt` - Human-readable

Example TXT output:
```
================================================================================
XGBOOST HYPERPARAMETER TUNING RESULTS
================================================================================

Tuning Date: 2025-10-23 14:30:45
Number of Trials: 50
Best 5-Fold CV RMSE: 0.123456

Best Hyperparameters:
----------------------------------------
  learning_rate: 0.15
  max_depth: 6
  subsample: 0.8
  colsample_bytree: 0.85
  min_child_weight: 3
  reg_alpha: 0.5
  reg_lambda: 2.0
  smoothing: 2.5

================================================================================
```

## Runtime Expectations

- **CatBoost**: ~30-60 minutes (50 trials)
- **XGBoost**: ~20-40 minutes (50 trials)
- **Random Forest**: ~40-90 minutes (30 trials, slower per trial)

*Actual times depend on your hardware and data size.*

## Output Files

Each tuning folder will generate:

```
tune_<model>/
├── tune_<model>.py           # Tuning script
├── README.md                  # Model-specific documentation
├── processed_data/            # Preprocessed data
│   ├── train_processed.csv
│   ├── test_processed.csv
│   ├── label_encodings.json
│   └── preprocessing_metadata.txt
├── tuning_results/            # Best parameters
│   ├── best_params_<model>.json
│   └── best_params_<model>.txt
└── submission_<model>_tuned.csv  # Final predictions
```

## How It Works

### Phase 1: Hyperparameter Search
1. Optuna suggests a set of hyperparameters
2. 5-Fold CV is performed with these parameters
3. Average RMSE across folds is calculated
4. Process repeats for N trials (30-50)
5. Best parameters are those with lowest average CV RMSE

### Phase 2: Final Training
1. Best parameters are loaded
2. Model is trained on 100% of training data
3. For XGBoost: First train on 90% to find best iteration, then retrain on 100%
4. Predictions are generated for test set
5. Submission file is created

## Tips for Better Results

1. **Increase Trials**: If you have time, increase `n_trials` in the scripts
2. **Adjust Search Space**: Modify the `suggest_*` ranges if you want to explore different regions
3. **Use GPUs**: CatBoost and XGBoost support GPU acceleration (add `task_type='GPU'` for CatBoost)
4. **Parallel Trials**: Optuna supports parallel optimization (requires additional setup)

## Next Steps After Tuning

1. **Compare Results**: Check the CV RMSE across all three models
2. **Use Best Model**: Submit the predictions from the best-performing model
3. **Ensemble**: Consider averaging predictions from multiple tuned models
4. **Further Tuning**: Run additional trials on the most promising model

## Troubleshooting

**Import errors**: Make sure all packages are installed
```bash
pip install optuna catboost xgboost scikit-learn pandas numpy category-encoders
```

**Preprocessing errors**: Ensure `../preprocessing/preprocess_data.py` exists

**Dataset not found**: Ensure `../dataset/train.csv` and `../dataset/test.csv` exist

**Memory errors**: Reduce number of folds (use 3 instead of 5) or reduce n_estimators ranges

## Questions?

Refer to the README.md in each individual tuning folder for model-specific details.

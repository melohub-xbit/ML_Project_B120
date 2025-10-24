# Ensemble Model Creation Guide

## 📌 Overview

This guide walks you through creating an optimal ensemble of three models using **stacking with Ridge regression**. This method finds the best weights automatically using your validation data.

## 🎯 Models Being Ensembled

1. **CatBoost (Original)** - `exp_catboost_3_v2` - Untuned, but performs best
2. **XGBoost (Tuned)** - `tune_xgboost_v2` - Tuned with 5-fold CV
3. **Random Forest (Tuned)** - `tune_random_forest_v2` - Tuned with 5-fold CV

## ⚙️ Setup: Modifications Made

The following scripts have been **modified** to save necessary files for ensembling:

### 1. CatBoost Script (`exp_catboost_3_v2/train_and_predict_cb.py`)
**New saves:**
- `oof_cb_preds.npy` - Validation predictions (log scale)
- `oof_cb_true.npy` - True validation values (log scale)
- `catboost_model.pkl` - Trained model
- `log_preds_cb.npy` - Test predictions (log scale)

### 2. XGBoost Script (`tune_xgboost_v2/tune_xgboost.py`)
**New saves:**
- `xgboost_model.pkl` - Trained model
- `xgboost_encoder.pkl` - TargetEncoder
- `log_preds_xgb.npy` - Test predictions (log scale)

### 3. Random Forest Script (`tune_random_forest_v2/tune_random_forest.py`)
**New saves:**
- `random_forest_model.pkl` - Trained model
- `random_forest_encoder.pkl` - TargetEncoder
- `log_preds_rf.npy` - Test predictions (log scale)

## 🚀 Execution Steps

### Phase 1: Run Individual Models (If Not Already Done)

```bash
# Step 1: Run CatBoost
cd exp_catboost_3_v2
python train_and_predict_cb.py
cd ..

# Step 2: Run XGBoost Tuning (takes ~20-40 min)
cd tune_xgboost_v2
python tune_xgboost.py
cd ..

# Step 3: Run Random Forest Tuning (takes ~40-90 min)
cd tune_random_forest_v2
python tune_random_forest.py
cd ..
```

**What to check:**
- ✅ All three produce `submission_*.csv` files
- ✅ All three produce `log_preds_*.npy` files
- ✅ XGBoost and RF produce model and encoder `.pkl` files
- ✅ CatBoost produces `oof_cb_preds.npy` and `oof_cb_true.npy`

### Phase 2: Create the Ensemble

```bash
cd ensemble
```

#### Step 1: Generate OOF Predictions
```bash
python get_oof_preds.py
```

**What it does:**
- Recreates CatBoost's exact 80/20 validation split
- Loads XGBoost and Random Forest models
- Generates validation predictions on the same samples CatBoost used
- Saves `oof_xgb_preds.npy` and `oof_rf_preds.npy`

**Expected output:**
```
OOF PREDICTIONS GENERATED SUCCESSFULLY!

Individual model RMSE on validation set (log scale):
  CatBoost:      0.XXXXXX
  XGBoost:       0.XXXXXX
  Random Forest: 0.XXXXXX

  Simple Average Ensemble: 0.XXXXXX
```

**Time:** < 1 minute

---

#### Step 2: Find Optimal Weights
```bash
python find_weights.py
```

**What it does:**
- Loads all OOF predictions (CB, XGB, RF)
- Tests different Ridge regression alpha values (0.001 to 10.0)
- Finds weights that minimize validation RMSE
- Saves `optimal_weights.npy` and `optimal_weights.txt`

**Expected output:**
```
OPTIMAL WEIGHTS FOUND

Best alpha (regularization): 0.XXX
Best validation RMSE: 0.XXXXXX

Optimal Ensemble Weights:
  CatBoost:      0.XXXX  (XX.X%)
  XGBoost:       0.XXXX  (XX.X%)
  Random Forest: 0.XXXX  (XX.X%)

Improvement over best single: +X.XX%
```

**Time:** < 1 minute

---

#### Step 3: Generate Final Ensemble
```bash
python ensemble.py
```

**What it does:**
- Loads optimal weights from Step 2
- Loads log-scale test predictions from all three models
- Computes weighted average **on log scale**
- Inverse transforms to original scale
- Saves `submission_ensemble_optimal.csv`

**Expected output:**
```
ENSEMBLE COMPLETE!

Submit this file to Kaggle: submission_ensemble_optimal.csv

Ensemble Details:
  - Uses optimal weights from Ridge regression
  - Weighted averaging done on LOG SCALE
  - Based on XX.X% CatBoost + XX.X% XGBoost + XX.X% RF
```

**Time:** < 10 seconds

---

## 📊 Understanding the Results

### Interpreting Weights

The optimal weights reflect each model's validation performance:

```
Example Output:
  CatBoost:      0.6500 (65.0%)  ← Highest (best model)
  XGBoost:       0.2300 (23.0%)  ← Second-best
  Random Forest: 0.1200 (12.0%)  ← Weakest
```

**What this means:**
- CatBoost contributes 65% to the final prediction
- The ensemble trusts CatBoost the most
- But XGB and RF still add value (diversity)

### Performance Gains

Typical improvements:
- **Over best single model:** 0.5-2.0% RMSE reduction
- **Over simple (1/3, 1/3, 1/3) average:** 0.2-1.0% RMSE reduction

Even small improvements matter in competitions!

## 📁 Final Folder Structure

```
ML_Project_B120/
├── exp_catboost_3_v2/
│   ├── train_and_predict_cb.py        [MODIFIED]
│   ├── catboost_model.pkl             [NEW]
│   ├── log_preds_cb.npy               [NEW]
│   ├── oof_cb_preds.npy               [NEW]
│   ├── oof_cb_true.npy                [NEW]
│   └── submission_catboost.csv
│
├── tune_xgboost_v2/
│   ├── tune_xgboost.py                [MODIFIED]
│   ├── xgboost_model.pkl              [NEW]
│   ├── xgboost_encoder.pkl            [NEW]
│   ├── log_preds_xgb.npy              [NEW]
│   └── submission_xgboost_tuned.csv
│
├── tune_random_forest_v2/
│   ├── tune_random_forest.py          [MODIFIED]
│   ├── random_forest_model.pkl        [NEW]
│   ├── random_forest_encoder.pkl      [NEW]
│   ├── log_preds_rf.npy               [NEW]
│   └── submission_random_forest_tuned.csv
│
└── ensemble/                          [NEW FOLDER]
    ├── get_oof_preds.py
    ├── find_weights.py
    ├── ensemble.py
    ├── README.md
    ├── oof_xgb_preds.npy              [GENERATED]
    ├── oof_rf_preds.npy               [GENERATED]
    ├── optimal_weights.npy            [GENERATED]
    ├── optimal_weights.txt            [GENERATED]
    └── submission_ensemble_optimal.csv [FINAL OUTPUT ⭐]
```

## ⚠️ Common Issues & Solutions

### Issue 1: "FileNotFoundError: log_preds_cb.npy"
**Cause:** Individual model script not run yet  
**Solution:** 
```bash
cd exp_catboost_3_v2
python train_and_predict_cb.py
```

### Issue 2: "FileNotFoundError: xgboost_model.pkl"
**Cause:** XGBoost tuning not completed  
**Solution:**
```bash
cd tune_xgboost_v2
python tune_xgboost.py
```

### Issue 3: "Validation set size mismatch"
**Cause:** Different preprocessed data or random seed  
**Solution:** All scripts use the same `processed_data` folder from CatBoost and `random_state=42`

### Issue 4: Ensemble performs worse than best single model
**Analysis:** Check the weights - if one model has >90% weight, the ensemble won't help much  
**Solution:** Just use the best single model (usually CatBoost in your case)

## 🎓 Why This Method Works

### 1. Stacking vs. Simple Averaging
- **Simple Average:** Equal weights (1/3, 1/3, 1/3) - suboptimal
- **Stacking:** Learns optimal weights based on validation performance

### 2. Ridge Regression
- Prevents overfitting by adding L2 regularization
- `positive=True` ensures weights ≥ 0
- `fit_intercept=False` ensures weights sum to ~1.0

### 3. Log-Scale Averaging
Models were trained on log-transformed targets:
```python
# Training:
y_log = np.log1p(y + shift_value)
model.fit(X, y_log)

# Prediction:
pred_log = model.predict(X_test)

# Ensemble (CORRECT):
ensemble_log = w1*pred_log1 + w2*pred_log2 + w3*pred_log3
final = np.expm1(ensemble_log) - shift_value
```

## 🏆 Expected Kaggle Performance

Based on your validation results:
- If **validation RMSE decreases**: Kaggle score should also improve
- If **weights are balanced** (e.g., 40/30/30): Ensemble adds diversity
- If **one weight dominates** (e.g., 90/5/5): Just use that single model

## 📈 Next Steps

1. ✅ **Submit to Kaggle:** `submission_ensemble_optimal.csv`
2. ✅ **Compare scores:**
   - Individual CatBoost score
   - Individual XGBoost score  
   - Individual Random Forest score
   - Ensemble score
3. ✅ **Analyze:**
   - Did ensemble beat all singles?
   - How close is validation RMSE to Kaggle score?
4. ✅ **Decide:**
   - If ensemble wins: Use it!
   - If single model wins: Use that!

## 💡 Pro Tips

1. **Weights tell you everything:** High weight = trust that model more
2. **Small gains matter:** Even 0.5% improvement can be significant
3. **Trust your validation:** If it works locally, it should work on Kaggle
4. **Log scale is sacred:** Never average in original scale!
5. **Diversity helps:** If all models were identical, ensembling wouldn't help

---

**You're all set!** Follow the steps and create your optimal ensemble. Good luck! 🎉

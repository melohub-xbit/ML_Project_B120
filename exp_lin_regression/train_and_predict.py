import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Model & Metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- Step 1: Load Data ---
print("Loading data...")
train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')

print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# --- Step 2: Feature Engineering ---
print("\nApplying feature engineering...")

def engineer_features(df):
    # --- 1. Handle Dates ---
    df['Order_Placed_Date'] = pd.to_datetime(df['Order_Placed_Date'], format='mixed', dayfirst=False)
    df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'], format='mixed', dayfirst=False)
    
    # Calculate Delivery_Duration
    df['Delivery_Duration'] = (df['Delivery_Date'] - df['Order_Placed_Date']).dt.days
    
    # Extract date parts
    df['Order_Month'] = df['Order_Placed_Date'].dt.month
    df['Order_DayOfWeek'] = df['Order_Placed_Date'].dt.dayofweek
    df['Order_Year'] = df['Order_Placed_Date'].dt.year

    # --- 2. Create Numerical Features ---
    df['Equipment_Area'] = df['Equipment_Height'] * df['Equipment_Width']
    df['Equipment_Density'] = df['Equipment_Weight'] / (df['Equipment_Area'] + 1e-6)
    
    # --- 3. Handle Binary Features ---
    binary_cols = [
        'CrossBorder_Shipping', 'Urgent_Shipping', 'Installation_Service',
        'Fragile_Equipment', 'Rural_Hospital'
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({True: 1, False: 0})

    # --- 4. Drop Original/Unnecessary Columns ---
    df = df.drop(['Order_Placed_Date', 'Delivery_Date'], axis=1)
    
    return df

# Apply the feature engineering
train_features_df = engineer_features(train_df.copy())
test_features_df = engineer_features(test_df.copy())

# --- Step 3: Target Variable and Data Splitting ---
print("\nPreparing target variable and splitting data...")

# Remove rows with missing or invalid target values
print(f"Rows before cleaning: {len(train_features_df)}")
train_features_df = train_features_df.dropna(subset=['Transport_Cost'])
print(f"Rows after dropping NaN: {len(train_features_df)}")

# Remove rows with non-positive Transport_Cost (log1p requires non-negative values)
train_features_df = train_features_df[train_features_df['Transport_Cost'] >= 0]
print(f"Rows after removing negative costs: {len(train_features_df)}")

# Apply log transform to the target variable
y_train_log = np.log1p(train_features_df['Transport_Cost'])

# Check for any remaining NaN values after log transform
if y_train_log.isna().any():
    print(f"Warning: {y_train_log.isna().sum()} NaN values found after log transform")
    # Remove any remaining NaN
    valid_indices = ~y_train_log.isna()
    y_train_log = y_train_log[valid_indices]
    train_features_df = train_features_df[valid_indices]
    print(f"Rows after removing NaN from log transform: {len(train_features_df)}")

# Drop the target variable from the training features
X_train = train_features_df.drop('Transport_Cost', axis=1)

# The test set is just our processed test features
X_test = test_features_df

# Split the training data for validation
X_train_sub, X_val, y_train_sub_log, y_val_log = train_test_split(
    X_train, y_train_log, test_size=0.2, random_state=42
)

# --- Step 4: Define Preprocessing Pipelines ---
print("\nSetting up preprocessing pipelines...")

# Drop high-cardinality features
high_cardinality_cols = ['Hospital_Id', 'Supplier_Name', 'Hospital_Location']
X_train_sub = X_train_sub.drop(columns=high_cardinality_cols)
X_val = X_val.drop(columns=high_cardinality_cols)
X_test = X_test.drop(columns=high_cardinality_cols)

# Identify remaining numerical and categorical columns
numerical_cols = X_train_sub.select_dtypes(include=np.number).columns
categorical_cols = X_train_sub.select_dtypes(include=['object', 'category']).columns

print(f"Numerical columns: {list(numerical_cols)}")
print(f"Categorical columns: {list(categorical_cols)}")

# Pipeline for Numerical Features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline for Categorical Features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine Pipelines with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough'
)

# --- Step 5: Model Training ---
print("\nCreating model pipeline...")

# Create the Full Model Pipeline
lr_model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression(n_jobs=-1))
])

print("Starting model training...")
lr_model_pipeline.fit(X_train_sub, y_train_sub_log)
print("Model training complete.")

# --- Step 6: Evaluation and Coefficient Analysis ---
print("\nEvaluating model...")

# Make predictions on the validation set
preds_val_log = lr_model_pipeline.predict(X_val)

# Reverse the Log Transform
preds_val = np.expm1(preds_val_log)
y_val_orig = np.expm1(y_val_log)

# Calculate the RMSE
rmse = np.sqrt(mean_squared_error(y_val_orig, preds_val))
print(f"Validation RMSE: {rmse}")

# Analyze Coefficients
try:
    # Get the feature names created by the preprocessor
    feature_names = (
        list(numerical_cols) + 
        list(lr_model_pipeline.named_steps['preprocessor']
             .named_transformers_['cat']
             .named_steps['onehot']
             .get_feature_names_out(categorical_cols))
    )
    
    # Get the model's coefficients
    coefficients = lr_model_pipeline.named_steps['regressor'].coef_
    
    # Create a DataFrame of coefficients
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    }).sort_values('coefficient', ascending=False)
    
    print("\nTop 10 Positive Coefficients (Increase Cost):")
    print(coef_df.head(10))
    
    print("\nTop 10 Negative Coefficients (Decrease Cost):")
    print(coef_df.tail(10))

except Exception as e:
    print(f"\nCould not retrieve feature names for coefficients: {e}")

# --- Step 7: Final Prediction on Test Data ---
print("\nMaking final predictions on test data...")

# The pipeline automatically applies all the fitted imputers, scalers, and encoders
preds_test_log = lr_model_pipeline.predict(X_test)

# Reverse the Log Transform
preds_test = np.expm1(preds_test_log)

# Ensure no negative predictions
preds_test[preds_test < 0] = 0

# Create Submission File with Hospital_Id from test data
submission_df = pd.DataFrame({
    'Hospital_Id': test_df['Hospital_Id'],
    'Transport_Cost': preds_test
})
submission_df.to_csv('submission_linear_regression.csv', index=False)

print("\nSubmission file 'submission_linear_regression.csv' created successfully!")
print(f"Submission shape: {submission_df.shape}")
print("\nFirst few predictions:")
print(submission_df.head())
print("\nLast few predictions:")
print(submission_df.tail())

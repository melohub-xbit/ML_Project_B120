import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')

print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

def engineer_features(df):
    df['Order_Placed_Date'] = pd.to_datetime(df['Order_Placed_Date'])
    df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'])
    df['Delivery_Duration'] = (df['Delivery_Date'] - df['Order_Placed_Date']).dt.days
    df['Order_Month'] = df['Order_Placed_Date'].dt.month
    df['Order_DayOfWeek'] = df['Order_Placed_Date'].dt.dayofweek
    df['Order_Year'] = df['Order_Placed_Date'].dt.year

    df['Equipment_Area'] = df['Equipment_Height'] * df['Equipment_Width']
    df['Equipment_Density'] = df['Equipment_Weight'] / (df['Equipment_Area'] + 1e-6)
    
    binary_cols = [
        'CrossBorder_Shipping', 'Urgent_Shipping', 'Installation_Service',
        'Fragile_Equipment', 'Rural_Hospital'
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({True: 1, False: 0})

    df = df.drop(['Order_Placed_Date', 'Delivery_Date'], axis=1)
    
    return df

train_features_df = engineer_features(train_df.copy())
test_features_df = engineer_features(test_df.copy())

y_train = train_features_df['Transport_Cost']

X_train = train_features_df.drop('Transport_Cost', axis=1)

X_test = test_features_df



high_cardinality_cols = ['Hospital_Id', 'Supplier_Name', 'Hospital_Location']
X_train = X_train.drop(columns=high_cardinality_cols)
X_test = X_test.drop(columns=high_cardinality_cols)

numerical_cols = X_train.select_dtypes(include=np.number).columns
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns

print(f"Numerical columns: {list(numerical_cols)}")
print(f"Categorical columns: {list(categorical_cols)}")


numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough'
)

tree_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=42))
])

param_grid = {
    'regressor__max_depth': [5, 10, 15, None],
    'regressor__min_samples_leaf': [10, 20, 50]
}

grid_search = GridSearchCV(
    tree_pipeline, 
    param_grid, 
    cv=5, 
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,  
    verbose=1   
)

print("\nStarting model tuning (GridSearchCV)...")
grid_search.fit(X_train, y_train)

print("\nModel tuning complete.")
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best CV RMSE: {-grid_search.best_score_:.2f}")

best_tree_model = grid_search.best_estimator_

try:
    feature_names = (
        list(numerical_cols) + 
        list(best_tree_model.named_steps['preprocessor']
             .named_transformers_['cat']
             .named_steps['onehot']
             .get_feature_names_out(categorical_cols))
    )
    
    importances = best_tree_model.named_steps['regressor'].feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))

except Exception as e:
    print(f"\nCould not retrieve feature names for importances: {e}")

plt.figure(figsize=(20, 10))
plot_tree(
    best_tree_model.named_steps['regressor'],
    feature_names=feature_names,
    filled=True,
    max_depth=3, 
    fontsize=10
)
plt.title("Decision Tree (First 3 Levels)")
plt.savefig('decision_tree_visualization.png')
print("\nDecision tree visualization saved as 'decision_tree_visualization.png'")

print("\nMaking final predictions on test data...")

preds_test = grid_search.predict(X_test)


preds_test[preds_test < 0] = 0

submission_df = pd.DataFrame({
    'Hospital_Id': test_df['Hospital_Id'],
    'Transport_Cost': preds_test
})
submission_df.to_csv('submission_decision_tree.csv', index=False)

print("\nSubmission file 'submission_decision_tree.csv' created successfully!")
print(f"Submission shape: {submission_df.shape}")
print(submission_df.head())

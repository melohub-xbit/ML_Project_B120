import numpy as np
import pandas as pd
import re
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# DATA PREPROCESSING PIPELINE
# ============================================================================

def parse_location(location_str):
    """Parse hospital location to extract state, zip, and military status"""
    if pd.isna(location_str) or location_str == 'Missing':
        return 'Missing', 'Missing', 0
    
    # Check for military addresses (APO, FPO, DPO with AA, AE, AP)
    is_military = 1 if re.search(r'\b(APO|FPO|DPO)\b', str(location_str), re.IGNORECASE) else 0
    is_military = is_military or (1 if re.search(r'\b(AA|AE|AP)\b', str(location_str)) else 0)
    
    # Extract state (2 letter code) - usually before the last comma and zip
    state_match = re.search(r'\b([A-Z]{2})\b\s+\d{5}', str(location_str))
    if state_match:
        state = state_match.group(1)
    else:
        state = 'Missing'
    
    # Extract zip code (5 digits)
    zip_match = re.search(r'\b(\d{5})\b', str(location_str))
    if zip_match:
        zip_code = zip_match.group(1)
    else:
        zip_code = 'Missing'
    
    return state, zip_code, is_military


def engineer_physical_features(df):
    """Create physical dimension features"""
    # Handle zeros and NaNs before division
    df['Equipment_Area'] = df['Equipment_Height'] * df['Equipment_Width']
    
    # Weight per Area (density proxy)
    df['Weight_per_Area'] = np.where(
        df['Equipment_Area'] > 0,
        df['Equipment_Weight'] / df['Equipment_Area'],
        0
    )
    
    # Value per Weight
    df['Value_per_Weight'] = np.where(
        df['Equipment_Weight'] > 0,
        df['Equipment_Value'] / df['Equipment_Weight'],
        0
    )
    
    # Value per Area
    df['Value_per_Area'] = np.where(
        df['Equipment_Area'] > 0,
        df['Equipment_Value'] / df['Equipment_Area'],
        0
    )
    
    # Replace inf values with 0
    df['Weight_per_Area'].replace([np.inf, -np.inf], 0, inplace=True)
    df['Value_per_Weight'].replace([np.inf, -np.inf], 0, inplace=True)
    df['Value_per_Area'].replace([np.inf, -np.inf], 0, inplace=True)
    
    return df


def engineer_date_features(df):
    """Extract date features from Order_Placed_Date and Delivery_Date"""
    # Convert to datetime
    df['Order_Placed_Date'] = pd.to_datetime(df['Order_Placed_Date'], format='%m/%d/%y', errors='coerce')
    df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'], format='%m/%d/%y', errors='coerce')
    
    # Extract date components
    df['Order_Year'] = df['Order_Placed_Date'].dt.year
    df['Order_Month'] = df['Order_Placed_Date'].dt.month
    df['Order_Day'] = df['Order_Placed_Date'].dt.day
    df['Order_DayOfWeek'] = df['Order_Placed_Date'].dt.dayofweek
    df['Order_Quarter'] = df['Order_Placed_Date'].dt.quarter
    
    df['Delivery_Year'] = df['Delivery_Date'].dt.year
    df['Delivery_Month'] = df['Delivery_Date'].dt.month
    df['Delivery_Day'] = df['Delivery_Date'].dt.day
    df['Delivery_DayOfWeek'] = df['Delivery_Date'].dt.dayofweek
    df['Delivery_Quarter'] = df['Delivery_Date'].dt.quarter
    
    # Calculate delivery time (days)
    df['Delivery_Time'] = (df['Delivery_Date'] - df['Order_Placed_Date']).dt.days
    
    # Drop original date columns
    df.drop(['Order_Placed_Date', 'Delivery_Date'], axis=1, inplace=True)
    
    return df


def preprocess_data(train_path, test_path, output_dir):
    """
    Main preprocessing pipeline
    """
    print("=" * 80)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # Load data
    print("\n[1/9] Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Store IDs and target
    train_ids = train_df['Hospital_Id'].copy()
    test_ids = test_df['Hospital_Id'].copy()
    
    has_target = 'Transport_Cost' in train_df.columns
    if has_target:
        y_train = train_df['Transport_Cost'].copy()
        
        # Log transform target (handling negative values by shifting)
        min_cost = y_train.min()
        print(f"\nTarget variable stats:")
        print(f"  Min: {min_cost:.2f}")
        print(f"  Max: {y_train.max():.2f}")
        print(f"  Mean: {y_train.mean():.2f}")
        print(f"  Negative values: {(y_train < 0).sum()}")
        
        # Shift to make all values positive, then apply log1p
        shift_value = abs(min_cost) + 1 if min_cost < 0 else 0
        y_train_log = np.log1p(y_train + shift_value)
        
        print(f"  Shift applied: {shift_value:.2f}")
        print(f"  Log-transformed target range: [{y_train_log.min():.4f}, {y_train_log.max():.4f}]")
    
    # Drop target from train
    if has_target:
        train_df = train_df.drop(['Transport_Cost'], axis=1)
    
    # Combine train and test for consistent preprocessing
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    print(f"\nCombined shape: {combined_df.shape}")
    
    # [2/9] Engineer date features
    print("\n[2/9] Engineering date features...")
    combined_df = engineer_date_features(combined_df)
    
    # [3/9] Engineer location features
    print("\n[3/9] Engineering location features...")
    location_data = combined_df['Hospital_Location'].apply(parse_location)
    combined_df['Location_State'] = location_data.apply(lambda x: x[0])
    combined_df['Location_Zip'] = location_data.apply(lambda x: x[1])
    combined_df['Location_Is_Military'] = location_data.apply(lambda x: x[2])
    
    print(f"  Unique states: {combined_df['Location_State'].nunique()}")
    print(f"  Military addresses: {combined_df['Location_Is_Military'].sum()}")
    
    # [4/9] Drop high-cardinality and ID columns
    print("\n[4/9] Dropping unnecessary columns...")
    cols_to_drop = ['Hospital_Id', 'Hospital_Location', 'Supplier_Name']
    combined_df = combined_df.drop(cols_to_drop, axis=1)
    print(f"  Dropped: {cols_to_drop}")
    
    # [5/9] Engineer physical features (before imputation)
    print("\n[5/9] Engineering physical features...")
    combined_df = engineer_physical_features(combined_df)
    
    # [6/9] Identify column types
    print("\n[6/9] Identifying column types...")
    categorical_cols = combined_df.select_dtypes(include=['object']).columns.tolist()
    if 'is_train' in categorical_cols:
        categorical_cols.remove('is_train')  # This is a marker, not a feature
    
    # Get numerical columns (excluding the is_train marker)
    numerical_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'is_train' in numerical_cols:
        numerical_cols.remove('is_train')
    
    print(f"  Categorical columns: {len(categorical_cols)}")
    print(f"  Numerical columns: {len(numerical_cols)}")
    
    # [7/9] Impute categorical NaNs with 'Missing'
    print("\n[7/9] Imputing categorical missing values...")
    for col in categorical_cols:
        missing_count = combined_df[col].isna().sum()
        if missing_count > 0:
            print(f"  {col}: {missing_count} missing values")
            combined_df[col].fillna('Missing', inplace=True)
    
    # [8/9] Map binary Yes/No columns to 1/0
    print("\n[8/9] Encoding binary categorical variables...")
    binary_mapping = {'Yes': 1, 'No': 0, 'Missing': -1}
    binary_cols = ['Rural_Hospital']  # Add more if identified
    
    for col in binary_cols:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].map(binary_mapping)
            print(f"  {col}: Mapped Yes/No to 1/0")
    
    # [8.5/9] Encode categorical columns to numerical (label encoding)
    print("\n[8.5/9] Encoding categorical columns to numerical values...")
    label_encodings = {}
    
    for col in categorical_cols:
        if col not in binary_cols:  # Skip binary columns already encoded
            unique_values = combined_df[col].unique()
            # Create mapping dictionary
            value_to_num = {val: idx for idx, val in enumerate(sorted(unique_values))}
            combined_df[col] = combined_df[col].map(value_to_num)
            label_encodings[col] = value_to_num
            print(f"  {col}: {len(unique_values)} unique values mapped to 0-{len(unique_values)-1}")
    
    # [9/9] Impute numerical NaNs using IterativeImputer
    print("\n[9/9] Imputing numerical missing values with IterativeImputer...")
    numerical_cols_to_impute = [col for col in numerical_cols if combined_df[col].isna().any()]
    
    if numerical_cols_to_impute:
        print(f"  Columns to impute: {len(numerical_cols_to_impute)}")
        for col in numerical_cols_to_impute:
            print(f"    {col}: {combined_df[col].isna().sum()} missing")
        
        # Use IterativeImputer
        imputer = IterativeImputer(random_state=42, max_iter=10, verbose=0)
        combined_df[numerical_cols] = imputer.fit_transform(combined_df[numerical_cols])
        print("  IterativeImputer fitting completed")
    else:
        print("  No numerical columns require imputation")
    
    # Split back into train and test
    print("\n" + "=" * 80)
    print("SPLITTING AND SAVING")
    print("=" * 80)
    
    train_processed = combined_df[combined_df['is_train'] == 1].drop(['is_train'], axis=1).copy()
    test_processed = combined_df[combined_df['is_train'] == 0].drop(['is_train'], axis=1).copy()
    
    # Add Hospital_Id back to test (original values, not preprocessed)
    test_processed.insert(0, 'Hospital_Id', test_ids.values)
    
    # Add target back to train
    if has_target:
        train_processed['Transport_Cost'] = y_train
        train_processed['Transport_Cost_Log'] = y_train_log
        train_processed['Target_Shift_Value'] = shift_value
    
    print(f"\nProcessed train shape: {train_processed.shape}")
    print(f"Processed test shape: {test_processed.shape}")
    
    # Save processed data
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    train_output = os.path.join(output_dir, 'train_processed.csv')
    test_output = os.path.join(output_dir, 'test_processed.csv')
    
    train_processed.to_csv(train_output, index=False)
    test_processed.to_csv(test_output, index=False)
    
    print(f"\nSaved processed train to: {train_output}")
    print(f"Saved processed test to: {test_output}")
    
    # Save metadata
    metadata = {
        'original_train_shape': train_df.shape,
        'original_test_shape': test_df.shape,
        'processed_train_shape': train_processed.shape,
        'processed_test_shape': test_processed.shape,
        'categorical_columns': categorical_cols,
        'numerical_columns': numerical_cols,
        'target_shift_value': shift_value if has_target else 0,
        'features_engineered': [
            'Location_State', 'Location_Zip', 'Location_Is_Military',
            'Equipment_Area', 'Weight_per_Area', 'Value_per_Weight', 'Value_per_Area',
            'Order_Year', 'Order_Month', 'Order_Day', 'Order_DayOfWeek', 'Order_Quarter',
            'Delivery_Year', 'Delivery_Month', 'Delivery_Day', 'Delivery_DayOfWeek', 
            'Delivery_Quarter', 'Delivery_Time'
        ],
        'columns_dropped': cols_to_drop
    }
    
    metadata_output = os.path.join(output_dir, 'preprocessing_metadata.txt')
    with open(metadata_output, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Saved metadata to: {metadata_output}")
    
    # Save label encodings
    import json
    encodings_output = os.path.join(output_dir, 'label_encodings.json')
    with open(encodings_output, 'w') as f:
        json.dump(label_encodings, f, indent=2)
    print(f"Saved label encodings to: {encodings_output}")
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return train_processed, test_processed


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    train_path = '../dataset/train.csv'
    test_path = '../dataset/test.csv'
    output_dir = '../processed_data'
    
    preprocess_data(train_path, test_path, output_dir)

"""
Data preprocessing utilities for the regression experiment
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import config


class DataPreprocessor:
    """Handles all data preprocessing operations"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.categorical_features = config.CATEGORICAL_FEATURES
        self.date_features = config.DATE_FEATURES
        
    def load_data(self, filepath):
        """Load data from CSV file"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def handle_missing_values(self, df, is_training=True):
        """Handle missing values in the dataset"""
        print("Handling missing values...")
        
        # Store strategies for test data
        if is_training:
            self.fill_values = {}
        
        for col in df.columns:
            if col in [config.INDEX_COLUMN, config.TARGET_COLUMN]:
                continue
                
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                print(f"  {col}: {missing_count} missing values")
                
                if col in self.categorical_features:
                    # Fill categorical with mode or 'Unknown'
                    if is_training:
                        mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                        self.fill_values[col] = mode_val
                    df[col].fillna(self.fill_values[col], inplace=True)
                else:
                    # Fill numerical with median
                    if is_training:
                        median_val = df[col].median()
                        self.fill_values[col] = median_val
                    df[col].fillna(self.fill_values[col], inplace=True)
        
        return df
    
    def engineer_date_features(self, df):
        """Extract features from date columns"""
        print("Engineering date features...")
        
        for date_col in self.date_features:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                # Extract year, month, day
                df[f'{date_col}_Year'] = df[date_col].dt.year
                df[f'{date_col}_Month'] = df[date_col].dt.month
                df[f'{date_col}_Day'] = df[date_col].dt.day
                df[f'{date_col}_DayOfWeek'] = df[date_col].dt.dayofweek
                
                # Drop original date column
                df.drop(date_col, axis=1, inplace=True)
        
        # Calculate delivery time if both dates exist
        if 'Order_Placed_Date_Year' in df.columns and 'Delivery_Date_Year' in df.columns:
            # Reconstruct dates for calculation
            order_date = pd.to_datetime(df[['Order_Placed_Date_Year', 'Order_Placed_Date_Month', 'Order_Placed_Date_Day']].rename(
                columns={'Order_Placed_Date_Year': 'year', 'Order_Placed_Date_Month': 'month', 'Order_Placed_Date_Day': 'day'}))
            delivery_date = pd.to_datetime(df[['Delivery_Date_Year', 'Delivery_Date_Month', 'Delivery_Date_Day']].rename(
                columns={'Delivery_Date_Year': 'year', 'Delivery_Date_Month': 'month', 'Delivery_Date_Day': 'day'}))
            
            df['Delivery_Days'] = (delivery_date - order_date).dt.days
        
        return df
    
    def encode_categorical_features(self, df, is_training=True):
        """Encode categorical features"""
        print("Encoding categorical features...")
        
        for col in self.categorical_features:
            if col in df.columns:
                if is_training:
                    # Create and fit label encoder
                    le = LabelEncoder()
                    # Handle any remaining NaN
                    df[col] = df[col].astype(str)
                    df[col] = le.fit_transform(df[col])
                    self.label_encoders[col] = le
                else:
                    # Use existing encoder
                    df[col] = df[col].astype(str)
                    # Handle unseen categories
                    le = self.label_encoders[col]
                    df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    df[col] = le.transform(df[col])
        
        return df
    
    def engineer_features(self, df):
        """Create additional engineered features"""
        print("Engineering additional features...")
        
        # Volume feature
        if all(col in df.columns for col in ['Equipment_Height', 'Equipment_Width', 'Equipment_Weight']):
            df['Equipment_Volume'] = df['Equipment_Height'] * df['Equipment_Width']
            df['Volume_Weight_Ratio'] = df['Equipment_Volume'] / (df['Equipment_Weight'] + 1)
        
        # Value to weight ratio
        if 'Equipment_Value' in df.columns and 'Equipment_Weight' in df.columns:
            df['Value_Weight_Ratio'] = df['Equipment_Value'] / (df['Equipment_Weight'] + 1)
        
        # Value to base fee ratio
        if 'Equipment_Value' in df.columns and 'Base_Transport_Fee' in df.columns:
            df['Value_Fee_Ratio'] = df['Equipment_Value'] / (df['Base_Transport_Fee'] + 1)
        
        return df
    
    def prepare_features(self, df, is_training=True):
        """Prepare final feature set"""
        print("Preparing features...")
        
        # Drop index column and target if present
        features_df = df.drop(columns=[config.INDEX_COLUMN], errors='ignore')
        if config.TARGET_COLUMN in features_df.columns:
            features_df = features_df.drop(columns=[config.TARGET_COLUMN])
        
        # Drop any specified features
        features_df = features_df.drop(columns=config.DROP_FEATURES, errors='ignore')
        
        # Drop any remaining non-numeric columns
        non_numeric = features_df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            print(f"Dropping non-numeric columns: {list(non_numeric)}")
            features_df = features_df.drop(columns=non_numeric)
        
        # Handle any inf or extremely large values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        if is_training:
            self.fill_values.update({col: features_df[col].median() for col in features_df.columns if features_df[col].isna().any()})
        
        for col in features_df.columns:
            if features_df[col].isna().any():
                features_df[col].fillna(self.fill_values.get(col, 0), inplace=True)
        
        if is_training:
            self.feature_names = features_df.columns.tolist()
            print(f"Total features: {len(self.feature_names)}")
        
        return features_df
    
    def scale_features(self, X, is_training=True):
        """Scale features using StandardScaler"""
        print("Scaling features...")
        
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def preprocess_train_data(self, train_path):
        """Complete preprocessing pipeline for training data"""
        print("\n" + "="*50)
        print("PREPROCESSING TRAINING DATA")
        print("="*50)
        
        # Load data
        df = self.load_data(train_path)
        
        # Store target and index
        y = df[config.TARGET_COLUMN].values
        indices = df[config.INDEX_COLUMN].values
        
        # Preprocessing steps
        df = self.handle_missing_values(df, is_training=True)
        df = self.engineer_date_features(df)
        df = self.encode_categorical_features(df, is_training=True)
        df = self.engineer_features(df)
        
        # Prepare features
        X_df = self.prepare_features(df, is_training=True)
        X = X_df.values
        
        # Scale features
        X_scaled = self.scale_features(X, is_training=True)
        
        print(f"\nFinal data shape: {X_scaled.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
        
        return X_scaled, y, indices
    
    def preprocess_test_data(self, test_path):
        """Complete preprocessing pipeline for test data"""
        print("\n" + "="*50)
        print("PREPROCESSING TEST DATA")
        print("="*50)
        
        # Load data
        df = self.load_data(test_path)
        
        # Store index
        indices = df[config.INDEX_COLUMN].values
        
        # Preprocessing steps (using training parameters)
        df = self.handle_missing_values(df, is_training=False)
        df = self.engineer_date_features(df)
        df = self.encode_categorical_features(df, is_training=False)
        df = self.engineer_features(df)
        
        # Prepare features
        X_df = self.prepare_features(df, is_training=False)
        
        # Ensure same features as training
        missing_cols = set(self.feature_names) - set(X_df.columns)
        if missing_cols:
            print(f"Adding missing columns: {missing_cols}")
            for col in missing_cols:
                X_df[col] = 0
        
        # Reorder columns to match training
        X_df = X_df[self.feature_names]
        X = X_df.values
        
        # Scale features
        X_scaled = self.scale_features(X, is_training=False)
        
        print(f"\nFinal data shape: {X_scaled.shape}")
        
        return X_scaled, indices
    
    def save(self, filepath):
        """Save preprocessor state"""
        state = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'fill_values': self.fill_values
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Preprocessor saved to {filepath}")
    
    def load(self, filepath):
        """Load preprocessor state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.scaler = state['scaler']
        self.label_encoders = state['label_encoders']
        self.feature_names = state['feature_names']
        self.fill_values = state['fill_values']
        print(f"Preprocessor loaded from {filepath}")


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = DataPreprocessor()
    X_train, y_train, train_indices = preprocessor.preprocess_train_data(config.TRAIN_DATA_PATH)
    print(f"\nTraining data processed successfully!")
    print(f"X shape: {X_train.shape}, y shape: {y_train.shape}")

"""
Configuration file for the regression experiment
"""

# Data paths
TRAIN_DATA_PATH = '../dataset/train.csv'
TEST_DATA_PATH = '../dataset/test.csv'
SAMPLE_SUBMISSION_PATH = '../dataset/sample_submission.csv'

# Output paths
MODEL_SAVE_PATH = 'model_weights.pkl'
SUBMISSION_PATH = 'submission.csv'
PLOTS_DIR = 'plots'

# Model hyperparameters
HIDDEN_LAYERS = [128, 64, 32]  # Architecture of hidden layers
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 200
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Training parameters
EARLY_STOPPING_PATIENCE = 20
MIN_DELTA = 0.0001

# Data preprocessing
TARGET_COLUMN = 'Transport_Cost'
INDEX_COLUMN = 'Hospital_Id'

# Features to drop (besides target and index)
DROP_FEATURES = []

# Categorical features (will be one-hot encoded)
CATEGORICAL_FEATURES = [
    'Supplier_Name',
    'Equipment_Type',
    'CrossBorder_Shipping',
    'Urgent_Shipping',
    'Installation_Service',
    'Transport_Method',
    'Fragile_Equipment',
    'Hospital_Info',
    'Rural_Hospital'
]

# Date features
DATE_FEATURES = ['Order_Placed_Date', 'Delivery_Date']

# Numerical features will be automatically identified

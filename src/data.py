"""
Data loading and processing utilities
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def load_raw_data(filepath='data/raw/hepatitis_data.csv'):
    """Load the raw hepatitis dataset"""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Dataset loaded successfully: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        print("Please download the dataset from Kaggle and place it in data/raw/")
        return None

def get_data_info(df):
    """Get basic information about the dataset"""
    if df is None:
        return None
    
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'missing_values': df.isnull().sum(),
        'target_distribution': df['Category'].value_counts() if 'Category' in df.columns else None,
        'data_types': df.dtypes
    }
    
    return info

def clean_data(df):
    """Clean and prepare the dataset"""
    if df is None:
        return None
    
    # Make a copy
    data = df.copy()
    
    # Remove index column if present
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
    
    # Simplify target variable (binary classification)
    def simplify_category(category):
        if category in ['0=Blood Donor', '0s=suspect Blood Donor']:
            return 0  # Healthy
        else:
            return 1  # Hepatitis C
    
    data['target'] = data['Category'].apply(simplify_category)
    
    # Encode sex
    sex_encoder = LabelEncoder()
    data['sex_encoded'] = sex_encoder.fit_transform(data['Sex'])
    
    print(f"✅ Data cleaned successfully")
    print(f"   Healthy: {sum(data['target'] == 0)} samples")
    print(f"   Hepatitis C: {sum(data['target'] == 1)} samples")
    
    return data, sex_encoder

def prepare_features(data):
    """Prepare features for modeling"""
    if data is None:
        return None
    
    # Select feature columns
    feature_columns = ['Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT', 'sex_encoded']
    
    X = data[feature_columns]
    y = data['target']
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X), 
        columns=X.columns, 
        index=X.index
    )
    
    print(f"✅ Features prepared: {X_imputed.shape}")
    print(f"   Missing values after imputation: {X_imputed.isnull().sum().sum()}")
    
    return X_imputed, y, imputer

def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """Split and scale the data"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Data split and scaled:")
    print(f"   Training set: {X_train_scaled.shape}")
    print(f"   Test set: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
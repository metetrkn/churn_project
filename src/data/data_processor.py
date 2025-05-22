import pandas as pd
import numpy as np
from typing import Tuple

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the Telco Customer Churn dataset.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset by handling missing values and converting data types.
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    # Convert TotalCharges to numeric, replacing empty strings with NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing values in TotalCharges with 0
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Convert SeniorCitizen to string type for consistency
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    
    return df

def split_data(df: pd.DataFrame, target_col: str = 'Churn', 
               test_size: float = 0.2, val_size: float = 0.2,
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split the dataset into training, validation, and test sets.
    
    Args:
        df (pd.DataFrame): Preprocessed dataset
        target_col (str): Name of the target column
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of training data to use for validation
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple containing:
        - X_train: Training features
        - X_val: Validation features
        - X_test: Test features
        - y_train: Training target
        - y_val: Validation target
        - y_test: Test target
    """
    from sklearn.model_selection import train_test_split
    
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate validation set from training set
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test 
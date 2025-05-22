import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, List

def create_tenure_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create tenure cohorts based on customer tenure.
    
    Args:
        df (pd.DataFrame): Input dataframe with tenure column
        
    Returns:
        pd.DataFrame: Dataframe with added tenure_cohort column
    """
    def get_cohort(tenure):
        if tenure <= 12:
            return '0-12 months'
        elif tenure <= 24:
            return '13-24 months'
        elif tenure <= 36:
            return '25-36 months'
        elif tenure <= 48:
            return '37-48 months'
        else:
            return '49+ months'
    
    df['tenure_cohort'] = df['tenure'].apply(get_cohort)
    return df

def encode_categorical_features(df: pd.DataFrame, categorical_columns: List[str]) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical features using LabelEncoder.
    
    Args:
        df (pd.DataFrame): Input dataframe
        categorical_columns (List[str]): List of categorical column names
        
    Returns:
        Tuple containing:
        - pd.DataFrame: Dataframe with encoded features
        - dict: Dictionary of label encoders for each categorical column
    """
    encoded_df = df.copy()
    encoders = {}
    
    for col in categorical_columns:
        if col in df.columns:
            encoder = LabelEncoder()
            encoded_df[col] = encoder.fit_transform(df[col])
            encoders[col] = encoder
    
    return encoded_df, encoders

def scale_numerical_features(df: pd.DataFrame, numerical_columns: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numerical features using StandardScaler.
    
    Args:
        df (pd.DataFrame): Input dataframe
        numerical_columns (List[str]): List of numerical column names
        
    Returns:
        Tuple containing:
        - pd.DataFrame: Dataframe with scaled features
        - StandardScaler: Fitted scaler object
    """
    scaled_df = df.copy()
    scaler = StandardScaler()
    
    scaled_df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    return scaled_df, scaler

def prepare_features(df: pd.DataFrame, categorical_columns: List[str], 
                    numerical_columns: List[str]) -> Tuple[pd.DataFrame, dict, StandardScaler]:
    """
    Prepare features by encoding categorical variables and scaling numerical variables.
    
    Args:
        df (pd.DataFrame): Input dataframe
        categorical_columns (List[str]): List of categorical column names
        numerical_columns (List[str]): List of numerical column names
        
    Returns:
        Tuple containing:
        - pd.DataFrame: Processed dataframe
        - dict: Dictionary of label encoders
        - StandardScaler: Fitted scaler object
    """
    # Create tenure cohorts
    df = create_tenure_cohorts(df)
    
    # Encode categorical features
    df, encoders = encode_categorical_features(df, categorical_columns)
    
    # Scale numerical features
    df, scaler = scale_numerical_features(df, numerical_columns)
    
    return df, encoders, scaler 
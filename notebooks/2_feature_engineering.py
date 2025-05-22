# Telco Customer Churn Analysis - Feature Engineering
# This script focuses on feature engineering for the Telco Customer Churn dataset.
# We'll create new features, encode categorical variables, and prepare the data for modeling.

# 1. Setup and Data Loading
import pandas as pd
import numpy as np
from src.data.data_processor import load_data, preprocess_data

# Load and preprocess data
df = load_data('../data/Telco-Customer-Churn.csv')
df = preprocess_data(df)

print(f"Dataset shape: {df.shape}")
print(df.head())

# 2. Feature Engineering
# Create tenure cohorts
df['tenure_cohort'] = pd.qcut(df['tenure'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# Create total charges per month
df['charges_per_month'] = df['TotalCharges'] / df['tenure']

# Create binary features for services
service_columns = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in service_columns:
    df[f'{col}_binary'] = (df[col] == 'Yes').astype(int)

# 3. Encoding Categorical Variables
# One-hot encoding for categorical variables
categorical_cols = ['Contract', 'InternetService', 'PaymentMethod']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 4. Scaling Numerical Features
from sklearn.preprocessing import StandardScaler

numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charges_per_month']
scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

# 5. Final Data Preparation
# Drop original categorical columns
df_encoded = df_encoded.drop(columns=service_columns)

# Save the processed data
df_encoded.to_csv('../data/processed_data.csv', index=False)

print("Feature engineering completed. Processed data saved to '../data/processed_data.csv'.") 
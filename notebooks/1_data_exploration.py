# Telco Customer Churn Analysis - Data Exploration
# This script focuses on exploring and understanding the Telco Customer Churn dataset.
# We'll perform initial data quality checks, statistical analysis, and create visualizations to understand the data distribution and relationships between features.

# 1. Setup and Data Loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.data_processor import load_data, preprocess_data

# Set plot style
plt.style.use('seaborn')
sns.set_palette('husl')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Load and preprocess data
df = load_data('../data/Telco-Customer-Churn.csv')
df = preprocess_data(df)

print(f"Dataset shape: {df.shape}")
print(df.head())

# 2. Data Quality Assessment
# Check for missing values
missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values[missing_values > 0])

# Data types and basic statistics
print("\nData types:")
print(df.dtypes)

print("\nNumerical columns statistics:")
print(df.describe())

# 3. Target Variable Analysis
# Churn distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Churn')
plt.title('Distribution of Customer Churn')
plt.show()

# Calculate churn rate
churn_rate = df['Churn'].value_counts(normalize=True) * 100
print(f"Churn rate: {churn_rate['Yes']:.2f}%")
print(f"Retention rate: {churn_rate['No']:.2f}%")

# 4. Feature Analysis
# Numerical features distribution
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, col in enumerate(numerical_cols):
    sns.histplot(data=df, x=col, hue='Churn', ax=axes[idx])
    axes[idx].set_title(f'{col} Distribution')
plt.tight_layout()
plt.show()

# Categorical features analysis
categorical_cols = ['Contract', 'InternetService', 'PaymentMethod']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, col in enumerate(categorical_cols):
    sns.countplot(data=df, x=col, hue='Churn', ax=axes[idx])
    axes[idx].set_title(f'{col} vs Churn')
    axes[idx].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

# 5. Correlation Analysis
# Correlation between numerical features
correlation = df[numerical_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# 6. Key Findings and Insights
# Based on our exploration, we've identified several key insights:
# 1. Data Quality:
#    - The dataset is well-structured with no significant missing values
#    - All features are properly formatted
# 2. Churn Distribution:
#    - The dataset shows an imbalanced distribution of churn
#    - This will need to be addressed in the modeling phase
# 3. Key Features:
#    - Contract type appears to be strongly related to churn
#    - Monthly charges show different distributions for churned vs retained customers
#    - Tenure is likely to be an important predictor
# 4. Next Steps:
#    - Feature engineering to create tenure cohorts
#    - Encoding categorical variables
#    - Scaling numerical features
#    - Addressing class imbalance 
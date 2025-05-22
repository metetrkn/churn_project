# Telco Customer Churn Prediction

## Project Overview
This project focuses on predicting customer churn in a telecommunications company using machine learning techniques. The analysis uses the Telco Customer Churn dataset to build and evaluate various tree-based models for churn prediction.

## Dataset Description
The Telco Customer Churn dataset contains 7,032 customer records with 21 features including:
- Customer demographics (gender, senior citizen status, partner, dependents)
- Service information (phone service, multiple lines, internet service)
- Additional services (online security, online backup, device protection, tech support)
- Contract details (contract type, paperless billing, payment method)
- Billing information (monthly charges, total charges)
- Target variable: Churn status

## Project Structure
```
churn_project/
├── data/
│   └── Telco-Customer-Churn.csv
├── notebooks/
│   ├── 1_data_exploration.ipynb
│   ├── 2_feature_engineering.ipynb
│   └── 3_model_development.ipynb
├── src/
│   ├── data/
│   │   └── data_processor.py
│   ├── features/
│   │   └── feature_engineering.py
│   └── models/
│       └── model_trainer.py
├── requirements.txt
└── README.md
```

## Analysis Workflow
1. **Data Exploration**
   - Initial data quality assessment
   - Statistical analysis of features
   - Correlation analysis
   - Distribution analysis of key variables

2. **Feature Engineering**
   - Handling missing values
   - Encoding categorical variables
   - Creating tenure cohorts
   - Feature scaling and normalization

3. **Model Development**
   - Implementation of tree-based models:
     - Decision Tree
     - Random Forest
     - AdaBoost
     - Gradient Boosting
   - Model evaluation and comparison
   - Feature importance analysis

## Results
The project demonstrates the effectiveness of tree-based models in predicting customer churn, with Gradient Boosting showing the best performance. Key findings include:
- Most important features for churn prediction
- Impact of contract type on customer retention
- Relationship between service usage and churn probability

## Setup and Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Jupyter notebooks in sequence

## Dependencies
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

## Author
Mete Turkan
- LinkedIn: [linkedin.com/in/mete-turkan](https://linkedin.com/in/mete-turkan)

## License
This project is licensed under the [MIT License](LICENSE).

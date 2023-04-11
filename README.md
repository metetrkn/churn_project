# Supervised Learning Capstone Project - Tree Methods Focus

## Objective

    In this capstone project, our goal is to create a model to predict whether or not a customer will churn using the Telco-Customer-Churn dataset. We will focus on tree-based methods, such as a single Decision Tree, Random Forest, AdaBoost, and Gradient Boosting.
    Dataset

    The dataset consists of 7032 records with 21 columns. Each row represents a unique customer and their respective features, such as demographic information, contract type, billing information, and the target variable Churn.
    Workflow

    The project is divided into several parts:

        Quick Data Check
        Exploratory Data Analysis (EDA)
        Creating Cohorts based on Tenure
        Predictive Modeling

### Quick Data Check

    We start by examining the dataset using methods such as df.info() and df.describe(). The dataset contains 17 categorical features and 4 numerical features. No missing values are observed.
    Exploratory Data Analysis (EDA)

    The EDA section involves several visualization tasks, including:

        Distribution of TotalCharges between Churn categories using Box Plots and Violin Plots
        Bar plot showing the correlation of features to the class label
        Histogram displaying the distribution of the 'tenure' column
        Scatter plot of Total Charges versus Monthly Charges, colored by Churn

### Creating Cohorts based on Tenure

    We create cohorts based on the tenure column values and perform several visualization tasks such as:

        Calculating the Churn rate per cohort
        Scatterplot of Total Charges versus Monthly Charges, colored by Tenure Cohort
        Count plot showing the churn count per cohort
        Grid of Count Plots showing counts per Tenure Cohort, separated by contract type and colored by the Churn hue

### Predictive Modeling

    We explore four different tree-based methods: A Single Decision Tree, Random Forest, AdaBoost, and Gradient Boosting. The modeling process includes the following steps:

        Splitting the dataset into training, validation, and test sets
        Training each model using GridSearchCV for optimal hyperparameters
        Evaluating performance metrics such as classification report and plotting a confusion matrix
        Calculating feature importances for each model

### Results

    Through the analysis, we found that the Gradient Boosting model provides the best performance in predicting customer churn. The most important features in predicting customer churn are the contract type, monthly charges, and tenure.

#### Contributing

    Contributions are welcome. Please open an issue or submit a pull request to suggest changes or improvements.


#### Credits

    Mete Turkan
    linkedIn : linkedin.com/in/mete-turkan
    Inst : m_trkn46


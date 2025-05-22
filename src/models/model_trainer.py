import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple

def train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series, 
                       param_grid: Dict[str, Any] = None) -> Tuple[DecisionTreeClassifier, Dict[str, Any]]:
    """
    Train a Decision Tree model with hyperparameter tuning.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        param_grid (Dict[str, Any]): Grid of parameters to search
        
    Returns:
        Tuple containing:
        - DecisionTreeClassifier: Trained model
        - Dict[str, Any]: Best parameters
    """
    if param_grid is None:
        param_grid = {
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    model = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                       param_grid: Dict[str, Any] = None) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Train a Random Forest model with hyperparameter tuning.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        param_grid (Dict[str, Any]): Grid of parameters to search
        
    Returns:
        Tuple containing:
        - RandomForestClassifier: Trained model
        - Dict[str, Any]: Best parameters
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

def train_adaboost(X_train: pd.DataFrame, y_train: pd.Series,
                  param_grid: Dict[str, Any] = None) -> Tuple[AdaBoostClassifier, Dict[str, Any]]:
    """
    Train an AdaBoost model with hyperparameter tuning.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        param_grid (Dict[str, Any]): Grid of parameters to search
        
    Returns:
        Tuple containing:
        - AdaBoostClassifier: Trained model
        - Dict[str, Any]: Best parameters
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0]
        }
    
    model = AdaBoostClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

def train_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series,
                           param_grid: Dict[str, Any] = None) -> Tuple[GradientBoostingClassifier, Dict[str, Any]]:
    """
    Train a Gradient Boosting model with hyperparameter tuning.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        param_grid (Dict[str, Any]): Grid of parameters to search
        
    Returns:
        Tuple containing:
        - GradientBoostingClassifier: Trained model
        - Dict[str, Any]: Best parameters
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 1.0],
            'max_depth': [3, 5, 7]
        }
    
    model = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        
    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    return metrics

def plot_feature_importance(model: Any, feature_names: list, title: str = 'Feature Importance'):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
        title (str): Plot title
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, title: str = 'Confusion Matrix'):
    """
    Plot confusion matrix.
    
    Args:
        y_true (pd.Series): True labels
        y_pred (np.ndarray): Predicted labels
        title (str): Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show() 
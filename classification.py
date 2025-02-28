# classification.py

import itertools
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import joblib
from joblib import load
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tabular_data import load_airbnb
from typing import Tuple, Dict, Any, List


class ClassificationModel:
    """
    A class for implementing and evaluating classification models for Airbnb property listing data.
    """

    def __init__(self, model: Any):
        """
        Initialize the ClassificationModel with a given model.

        Args:
            model (Any): The classification model to be used 
                         (e.g., LogisticRegression, DecisionTreeClassifier).
        """
        self.model = model

    def import_and_standarise_data(self, data_file: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Import and standardize the Airbnb dataset from a CSV file.

        Reads data from the CSV file, extracts features & target via load_airbnb(), 
        selects numeric columns, imputes missing values, and standardizes them.

        Args:
            data_file (str): The path to the CSV file containing the dataset.

        Returns:
            (X_final, y): A tuple of the standardized DataFrame and the target Series.
        """
        df = pd.read_csv(data_file)
        X, y = load_airbnb(df)  # Custom function from tabular_data.py
        numeric_columns = X.select_dtypes(include=[np.number])
        df_numeric = pd.DataFrame(numeric_columns)

        # Impute missing numeric values with mean
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X_imputed = imputer.fit_transform(df_numeric)

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        X_final = pd.DataFrame(X_scaled, columns=numeric_columns.columns)
        return X_final, y

    def splited_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split the dataset into train (80%), validation (10%), test (10%).

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target labels

        Returns:
            X_train, X_validation, X_test, y_train, y_validation, y_test
        """
        np.random.seed(10)
        # 80% train, 20% test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=12
        )
        # half of the 20% => 10% for final test, 10% for validation
        X_test, X_validation, y_test, y_validation = train_test_split(
            X_test, y_test, test_size=0.5
        )
        return X_train, X_validation, X_test, y_train, y_validation, y_test

    def classification_metrics_performance(
        self, 
        y_test: pd.Series, 
        y_pred: pd.Series
    ) -> Dict[str, float]:
        """
        Return accuracy, precision, recall, and F1 (macro avg).
        """
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}

    def confusion_matrix(
        self, 
        model: Any, 
        X_test: pd.DataFrame, 
        y_test: pd.Series, 
        y_pred: pd.Series
    ) -> ConfusionMatrixDisplay:
        """
        Generate a confusion matrix display (for debugging/visualization).
        """
        disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
        disp.ax_.set_title("Confusion Matrix")
        return disp

    def tune_hyperparameters(
        self, 
        model: Any,
        X_train: pd.DataFrame, 
        X_val: pd.DataFrame, 
        X_test: pd.DataFrame,
        y_train: pd.Series, 
        y_val: pd.Series, 
        y_test: pd.Series,
        hyperparameters: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """
        Perform GridSearchCV over `hyperparameters` for the given `model`.
        Evaluate best model on train/val/test sets, return dict with results.
        """
        tuned_model = GridSearchCV(
            estimator=model,
            param_grid=hyperparameters,
            cv=3,
            scoring='accuracy'
        )
        tuned_model.fit(X_train, y_train)

        best_hyperparameters = tuned_model.best_params_
        best_model_accuracy = tuned_model.best_score_
        best_model_estimator = tuned_model.best_estimator_

        # Evaluate on train, val, test
        train_y_pred = best_model_estimator.predict(X_train)
        valid_y_pred = best_model_estimator.predict(X_val)
        test_y_pred = best_model_estimator.predict(X_test)

        train_metrics = self.classification_metrics_performance(y_train, train_y_pred)
        val_metrics = self.classification_metrics_performance(y_val, valid_y_pred)
        test_metrics = self.classification_metrics_performance(y_test, test_y_pred)

        result_dict = {
            "Best_Model": best_model_estimator,
            "Best_Hyperparameters": best_hyperparameters,
            "Best_Metrics": best_model_accuracy,  # best cross-val accuracy
            "Train_Metrics": train_metrics,
            "Validation_Metrics": val_metrics,
            "Test_Metrics": test_metrics
        }
        return result_dict

    def save_classification_model(
        self, 
        folder_name: str, 
        result_dict: Dict[str, Any]
    ) -> None:
        """
        Save best model, its hyperparams, and performance metrics in:
            ./models/classification/{folder_name}/
        """
        classification_dir = os.path.join(os.getcwd(), 'models', 'classification')
        folder_path = os.path.join(classification_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        best_model = result_dict["Best_Model"]
        best_hyperparameters = result_dict["Best_Hyperparameters"]
        performance_metric = {
            "Train_Metrics": result_dict["Train_Metrics"],
            "Validation_Metrics": result_dict["Validation_Metrics"],
            "Test_Metrics": result_dict["Test_Metrics"]
        }

        # Save model
        joblib.dump(best_model, os.path.join(folder_path, "model.joblib"))
        # Save hyperparams
        with open(os.path.join(folder_path, "hyperparameters.json"), 'w') as json_file:
            json.dump(best_hyperparameters, json_file)
        # Save metrics
        with open(os.path.join(folder_path, "metrics.json"), 'w') as json_file:
            json.dump(performance_metric, json_file)

    def evaluate_model(
        self, 
        model: Any, 
        hyperparameters_dict: Dict[str, List[Any]], 
        folder_name: str
    ) -> None:
        """
        Full pipeline: load & standardize data, split, hyperparameter-tune,
        then save the best model with metrics into the given folder_name.
        """
        data_file = os.path.join("tabular_data", "listing.csv")
        X, y = self.import_and_standarise_data(data_file)
        X_train, X_val, X_test, y_train, y_val, y_test = self.splited_data(X, y)

        # GridSearchCV, get best model & metrics
        result_dict = self.tune_hyperparameters(
            model,
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            hyperparameters_dict
        )
        # Save to disk
        self.save_classification_model(folder_name, result_dict)


def find_best_model(
    model_configs: List[Tuple[str, Any]]
) -> Tuple[Any, Dict[str, Any], Dict[str, float]]:
    """
    Find the best classification model among the model_configs,
    by comparing stored test-set accuracy in metrics.json.
    """
    best_classification_model = None
    best_hyperparameters_dict = {}
    best_metrics_dict = {"Accuracy": 0, "Precision": 0, "Recall": 0, "F1": 0}

    classification_dir = os.path.join(os.getcwd(), 'models', 'classification')
    for folder_name, _ in model_configs:
        model_dir = os.path.join(classification_dir, folder_name)
        model_file = os.path.join(model_dir, 'model.joblib')
        if not os.path.exists(model_file):
            print(f"Skipping {folder_name}: {model_file} does not exist.")
            continue

        # Load saved model
        loaded_model = load(model_file)

        # Load hyperparameters
        with open(os.path.join(model_dir, 'hyperparameters.json'), 'r') as f:
            hyperparameters = json.load(f)
        # Load test metrics
        with open(os.path.join(model_dir, 'metrics.json'), 'r') as f:
            metrics = json.load(f)["Test_Metrics"]  # e.g. {"Accuracy":..., "Precision":..., ...}

        # Compare test-set Accuracy
        if best_classification_model is None or metrics["Accuracy"] > best_metrics_dict["Accuracy"]:
            best_classification_model = loaded_model
            best_hyperparameters_dict = hyperparameters
            best_metrics_dict = metrics

    return best_classification_model, best_hyperparameters_dict, best_metrics_dict


# ------------------------------------------------------------------------------
# 1) Define hyperparameter grids
# ------------------------------------------------------------------------------

LogisticRegression_param = {
    'C': [1.0],
    'class_weight': ['balanced', None],
    'dual': [False],
    'fit_intercept': [True],
    'intercept_scaling': [1],
    'max_iter': [2000, 5000],
    'n_jobs': [None],
    'penalty': ['l2'],
    'random_state': [0],
    'solver': ['lbfgs', 'saga'],
    'tol': [0.0001],
    'verbose': [0],
    'warm_start': [False]
}

DecisionTreeClassifier_param = {
    'max_depth': [1, 3, 5, None],
    'min_samples_split': [3, 5, 10],
    'random_state': [10, 20, None],
    'splitter': ['best', 'random']
}

GradientBoostingClassifier_param = {
    'loss': ['log_loss', 'exponential'],
    'learning_rate': [0.0, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'criterion': ['friedman_mse', 'squared_error']
}

RandomForestClassifier_param = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt', 'log2', None]
}

# ------------------------------------------------------------------------------
# 2) Define model_configs at the top level (important for `import` statements!)
# ------------------------------------------------------------------------------

from typing import Union  # just an example if needed

model_configs: List[Tuple[str, Tuple[Any, Dict[str, List[Any]]]]] = [
    (
        "DecisionTreeClassifier",
        (
            DecisionTreeClassifier(),
            DecisionTreeClassifier_param
        )
    ),
    (
        "LogisticRegression",
        (
            LogisticRegression(),
            LogisticRegression_param
        )
    ),
    (
        "GradientBoostingClassifier",
        (
            GradientBoostingClassifier(),
            GradientBoostingClassifier_param
        )
    ),
    (
        "RandomForestClassifier",
        (
            RandomForestClassifier(),
            RandomForestClassifier_param
        )
    )
]


def logistic_regression(
    X: pd.DataFrame, 
    y: pd.Series
) -> Tuple[pd.Series, pd.Series, LogisticRegression, pd.DataFrame]:
    """
    Example function that trains a simple LogisticRegression (without GridSearch).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    clf = LogisticRegression(random_state=0, max_iter=5000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"LogisticRegression Accuracy: {accuracy:.2f}")
    return y_test, y_pred, clf, X_test


# ------------------------------------------------------------------------------
# 3) Optional main block for testing standalone
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Evaluate each model config, saving best version
    for folder_name, (model, params) in model_configs:
        print(f"Evaluating and saving model: {folder_name}")
        classification_obj = ClassificationModel(model)
        classification_obj.evaluate_model(model, params, folder_name)

    # Once done, pick best among them
    best_model, best_hparams, best_metrics = find_best_model(model_configs)
    print("Best Model:", best_model)
    print("Best Hyperparameters:", best_hparams)
    print("Best Metrics:", best_metrics)

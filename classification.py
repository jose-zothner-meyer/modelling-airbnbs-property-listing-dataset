# Import necessary libraries and modules
import itertools  # Provides tools for creating iterators for efficient looping.
import json  # Provides functions for parsing and creating JSON data.
import math  # Provides access to mathematical functions.
import matplotlib.pyplot as plt  # Used for plotting graphs and visualizations.
import numpy as np  # Provides support for large, multi-dimensional arrays and matrices.
import os  # Provides functions to interact with the operating system.
import pandas as pd  # Provides data structures and data analysis tools.
from joblib import load  # Used to load pre-saved models.
import joblib  # Provides tools for efficient serialization of Python objects.
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # Import ensemble classifiers.
from sklearn.impute import SimpleImputer  # Provides a simple imputation transformer for missing values.
from sklearn.linear_model import LogisticRegression  # Import logistic regression classifier.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay  # Metrics for model evaluation.
from sklearn.model_selection import GridSearchCV, train_test_split  # Tools for cross-validation and splitting data.
from sklearn.preprocessing import StandardScaler  # Used to standardize features.
from sklearn.tree import DecisionTreeClassifier  # Import decision tree classifier.
from tabular_data import load_airbnb  # Custom function to load Airbnb dataset.
from typing import Tuple, Dict, Any, List  # Provides type hinting for better code clarity.


class ClassificationModel:
    """
    A class for implementing and evaluating classification models for Airbnb property listing data.

    Attributes:
        model (Any): The classification model instance to be used for training and evaluation.
    """
    def __init__(self, model: Any):
        """
        Initialize the ClassificationModel with a given model.

        Args:
            model (Any): The classification model to be used (e.g., LogisticRegression, DecisionTreeClassifier).
        """
        self.model = model  # Store the model in an instance variable.

    def import_and_standarise_data(self, data_file: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Import and standardize the Airbnb dataset from a CSV file.

        This function reads data from a CSV file, extracts features and target variable using a custom loader,
        selects numeric columns, imputes missing values using the mean, and standardizes the numeric features.

        Args:
            data_file (str): The file path to the CSV file containing the dataset.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: A tuple containing:
                - X_final: A DataFrame of standardized numeric features.
                - y: A Series representing the target variable.
        """
        df = pd.read_csv(data_file)  # Read the CSV file into a DataFrame.
        X, y = load_airbnb(df)  # Load features and target variable using the custom load_airbnb function.
        numeric_columns = X.select_dtypes(include=[np.number])  # Select only numeric columns from the features.
        df_numeric = pd.DataFrame(numeric_columns)  # Create a new DataFrame with the numeric columns.
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  # Initialize imputer to fill missing values with the mean.
        X_imputed = imputer.fit_transform(df_numeric)  # Fit the imputer and transform the numeric data.
        scaler = StandardScaler()  # Initialize the StandardScaler for feature scaling.
        X_scaled = scaler.fit_transform(X_imputed)  # Standardize the imputed data.
        X_final = pd.DataFrame(X_scaled, columns=numeric_columns.columns)  # Create a DataFrame with scaled data using original column names.
        return X_final, y  # Return the standardized features and the target variable.

    def splited_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split the dataset into training, validation, and test sets.

        The dataset is first split into 80% training and 20% testing sets, and then the test set is further split
        equally into validation and final test sets.

        Args:
            X (pd.DataFrame): The DataFrame containing features.
            y (pd.Series): The Series containing the target variable.

        Returns:
            Tuple containing:
                - X_train (pd.DataFrame): Training features.
                - X_validation (pd.DataFrame): Validation features.
                - X_test (pd.DataFrame): Test features.
                - y_train (pd.Series): Training target values.
                - y_validation (pd.Series): Validation target values.
                - y_test (pd.Series): Test target values.
        """
        np.random.seed(10)  # Set the random seed for reproducibility.
        # Split data into training (80%) and testing (20%) sets using a fixed random state.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
        # Further split the test set into validation and final test sets (50% each of the test set).
        X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)
        return X_train, X_validation, X_test, y_train, y_validation, y_test  # Return all the dataset splits.

    def classification_metrics_performance(self, y_test: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """
        Calculate classification performance metrics.

        Computes accuracy, precision, recall, and F1 score using macro averaging and handles any division by zero.

        Args:
            y_test (pd.Series): The true target values.
            y_pred (pd.Series): The predicted target values.

        Returns:
            Dict[str, float]: A dictionary containing accuracy, precision, recall, and F1 score.
        """
        accuracy = accuracy_score(y_test, y_pred)  # Calculate the accuracy score.
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)  # Calculate precision with macro averaging.
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)  # Calculate recall with macro averaging.
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)  # Calculate the F1 score with macro averaging.
        return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}  # Return the performance metrics.

    def confusion_matrix(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, y_pred: pd.Series) -> ConfusionMatrixDisplay:
        """
        Generate a confusion matrix display using the provided model and test data.

        Args:
            model (Any): The trained classification model.
            X_test (pd.DataFrame): The test set features.
            y_test (pd.Series): The true labels for the test set.
            y_pred (pd.Series): The predicted labels for the test set (not directly used in this implementation).

        Returns:
            ConfusionMatrixDisplay: The display object for the confusion matrix.
        """
        disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)  # Generate the confusion matrix display from the model.
        disp.ax_.set_title("Confusion Matrix")  # Set the title for the confusion matrix plot.
        return disp  # Return the confusion matrix display.

    def tune_hyperparameters(self, model: Any, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, 
                             y_train: pd.Series, y_val: pd.Series, y_test: pd.Series, hyperparameters: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Tune hyperparameters of the model using grid search and evaluate performance.

        Performs grid search cross-validation on the training data, selects the best hyperparameters,
        evaluates the model on training, validation, and test sets, and returns the best model along with performance metrics.

        Args:
            model (Any): The classification model to be tuned.
            X_train (pd.DataFrame): Training features.
            X_val (pd.DataFrame): Validation features.
            X_test (pd.DataFrame): Test features.
            y_train (pd.Series): Training target values.
            y_val (pd.Series): Validation target values.
            y_test (pd.Series): Test target values.
            hyperparameters (Dict[str, List[Any]]): Dictionary of hyperparameters to search over.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - Best_Model: The best estimator found by GridSearchCV.
                - Best_Hyperparameters: The best hyperparameters.
                - Best_Metrics: The best cross-validated accuracy score.
                - Train_Metrics: Performance metrics on the training set.
                - Validation_Metrics: Performance metrics on the validation set.
                - Test_Metrics: Performance metrics on the test set.
        """
        # Initialize GridSearchCV with the specified model, hyperparameters, 3-fold cross-validation, and accuracy as the scoring metric.
        tuned_model = GridSearchCV(estimator=model, param_grid=hyperparameters, cv=3, scoring='accuracy')
        tuned_model.fit(X_train, y_train)  # Perform grid search on the training data.
        best_hyperparameters = tuned_model.best_params_  # Retrieve the best hyperparameters from grid search.
        best_model_accuracy = tuned_model.best_score_  # Retrieve the best cross-validated accuracy score.
        best_classification_model = tuned_model.best_estimator_  # Retrieve the best model estimator.
        train_y_pred = best_classification_model.predict(X_train)  # Predict on the training set.
        valid_y_pred = best_classification_model.predict(X_val)  # Predict on the validation set.
        test_y_pred = best_classification_model.predict(X_test)  # Predict on the test set.
        train_metrics = self.classification_metrics_performance(y_train, train_y_pred)  # Calculate training performance metrics.
        val_metrics = self.classification_metrics_performance(y_val, valid_y_pred)  # Calculate validation performance metrics.
        test_metrics = self.classification_metrics_performance(y_test, test_y_pred)  # Calculate test performance metrics.
        # Compile the results into a dictionary.
        result_dict = {
            "Best_Model": best_classification_model,
            "Best_Hyperparameters": best_hyperparameters,
            "Best_Metrics": best_model_accuracy,
            "Train_Metrics": train_metrics,
            "Validation_Metrics": val_metrics,
            "Test_Metrics": test_metrics
        }
        return result_dict  # Return the dictionary containing the best model and metrics.

    def save_classification_model(self, folder_name: str, result_dict: Dict[str, Any]) -> None:
        """
        Save the best classification model, hyperparameters, and performance metrics to disk.

        The model is saved in a folder structure under the current working directory:
        ./models/classification/{folder_name}/

        Args:
            folder_name (str): The name of the folder where the model will be saved.
            result_dict (Dict[str, Any]): Dictionary containing the best model, hyperparameters, and metrics.
        """
        classification_dir = os.path.join(os.getcwd(), 'models', 'classification')  # Build the base directory path.
        folder_path = os.path.join(classification_dir, folder_name)  # Build the folder path for the current model.
        os.makedirs(folder_path, exist_ok=True)  # Create the directory if it doesn't already exist.
        best_model = result_dict["Best_Model"]  # Extract the best model from the results.
        best_hyperparameters = result_dict["Best_Hyperparameters"]  # Extract the best hyperparameters.
        performance_metric = {  # Prepare a dictionary with performance metrics.
            "Train_Metrics": result_dict["Train_Metrics"],
            "Validation_Metrics": result_dict["Validation_Metrics"],
            "Test_Metrics": result_dict["Test_Metrics"]
        }
        # Save the best model to disk using joblib.
        joblib.dump(best_model, os.path.join(folder_path, "model.joblib"))
        # Save the best hyperparameters to a JSON file.
        with open(os.path.join(folder_path, "hyperparameters.json"), 'w') as json_file:
            json.dump(best_hyperparameters, json_file)
        # Save the performance metrics to a JSON file.
        with open(os.path.join(folder_path, "metrics.json"), 'w') as json_file:
            json.dump(performance_metric, json_file)

    def evaluate_model(self, model: Any, hyperparameters_dict: Dict[str, List[Any]], folder_name: str) -> None:
        """
        Evaluate the given model using Airbnb data and save the results.

        This function imports and standardizes the dataset, splits it into training, validation, and test sets,
        tunes hyperparameters using grid search, and saves the best model along with its performance metrics.

        Args:
            model (Any): The classification model to be evaluated.
            hyperparameters_dict (Dict[str, List[Any]]): Dictionary of hyperparameters for grid search.
            folder_name (str): The folder name for saving the model and results.
        """
        data_file = os.path.join("tabular_data", "listing.csv")  # Define the path to the data file.
        X, y = self.import_and_standarise_data(data_file)  # Import and standardize the data.
        X_train, X_val, X_test, y_train, y_val, y_test = self.splited_data(X, y)  # Split the data into training, validation, and test sets.
        # Tune hyperparameters and obtain the best model and metrics.
        result_dict = self.tune_hyperparameters(model, X_train, X_val, X_test, y_train, y_val, y_test, hyperparameters_dict)
        self.save_classification_model(folder_name, result_dict)  # Save the best model and associated data to disk.


def find_best_model(model_configs: List[Tuple[str, Any]]) -> Tuple[Any, Dict[str, Any], Dict[str, float]]:
    """
    Find and return the best classification model based on test set accuracy.

    Iterates through the provided model configurations, loads each saved model and its metrics,
    and selects the one with the highest test set accuracy.

    Args:
        model_configs (List[Tuple[str, Any]]): A list of tuples. Each tuple contains:
            - folder_name (str): The folder name where the model is saved.
            - (model, hyperparameters): A tuple containing the model instance and its hyperparameters.

    Returns:
        Tuple containing:
            - best_classification_model (Any): The best performing model.
            - best_hyperparameters_dict (Dict[str, Any]): Hyperparameters for the best model.
            - best_metrics_dict (Dict[str, float]): Performance metrics for the best model.
    """
    best_classification_model = None  # Initialize variable to store the best model.
    best_hyperparameters_dict = {}  # Initialize dictionary to store the best hyperparameters.
    best_metrics_dict = {"Accuracy": 0, "Precision": 0, "Recall": 0, "F1": 0}  # Initialize dictionary for best metrics.
    classification_dir = os.path.join(os.getcwd(), 'models', 'classification')  # Define the base directory for models.
    # Loop through each model configuration.
    for folder_name, _ in model_configs:
        model_dir = os.path.join(classification_dir, folder_name)  # Build the directory path for the current model.
        model_file = os.path.join(model_dir, 'model.joblib')  # Define the file path for the saved model.
        if not os.path.exists(model_file):  # Check if the model file exists.
            print(f"Skipping {folder_name}: {model_file} does not exist.")  # Print a message if it does not exist.
            continue  # Skip this model configuration.
        loaded_model = load(model_file)  # Load the saved model.
        with open(os.path.join(model_dir, 'hyperparameters.json'), 'r') as f:  # Open the hyperparameters file.
            hyperparameters = json.load(f)  # Load the hyperparameters from JSON.
        with open(os.path.join(model_dir, 'metrics.json'), 'r') as f:  # Open the metrics file.
            metrics = json.load(f)["Test_Metrics"]  # Load the test metrics from JSON.
        # Update the best model if the current one has a higher accuracy.
        if best_classification_model is None or metrics["Accuracy"] > best_metrics_dict["Accuracy"]:
            best_classification_model = loaded_model  # Set the current model as the best.
            best_hyperparameters_dict = hyperparameters  # Set the current hyperparameters as the best.
            best_metrics_dict = metrics  # Set the current metrics as the best.
    return best_classification_model, best_hyperparameters_dict, best_metrics_dict  # Return the best model and its details.


def logistic_regression(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.Series, pd.Series, LogisticRegression, pd.DataFrame]:
    """
    Train and evaluate a Logistic Regression model.

    Splits the dataset into training and test sets, trains the logistic regression model,
    predicts on the test set, prints the accuracy, and returns the predictions and trained model.

    Args:
        X (pd.DataFrame): The DataFrame containing features.
        y (pd.Series): The Series containing the target variable.

    Returns:
        Tuple containing:
            - y_test (pd.Series): True labels of the test set.
            - y_pred (pd.Series): Predicted labels for the test set.
            - clf (LogisticRegression): The trained logistic regression model.
            - X_test (pd.DataFrame): The test set features.
    """
    # Split data into training (90%) and test (10%) sets using a fixed random state.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    # Initialize LogisticRegression with an increased maximum number of iterations to ensure convergence.
    clf = LogisticRegression(random_state=0, max_iter=5000)
    clf.fit(X_train, y_train)  # Fit the logistic regression model on the training data.
    y_pred = clf.predict(X_test)  # Predict the target values for the test set.
    accuracy = accuracy_score(y_test, y_pred)  # Calculate the accuracy of the model.
    print(f"LogisticRegression Accuracy: {accuracy:.2f}")  # Print the accuracy score.
    return y_test, y_pred, clf, X_test  # Return the test labels, predictions, the trained model, and test features.


# Updated hyperparameters for different classification models

# Hyperparameters for LogisticRegression model.
LogisticRegression_param = {
    'C': [1.0],  # Inverse regularization strength.
    'class_weight': ['balanced', None],  # Options for weighting classes.
    'dual': [False],  # Flag to decide if dual formulation is used.
    'fit_intercept': [True],  # Whether to calculate the intercept.
    'intercept_scaling': [1],  # Scaling for the intercept.
    'max_iter': [2000, 5000],  # Maximum number of iterations for convergence.
    'n_jobs': [None],  # Number of parallel jobs.
    'penalty': ['l2'],  # Norm used in the penalization.
    'random_state': [0],  # Seed for randomness.
    'solver': ['lbfgs', 'saga'],  # Optimization algorithms to use.
    'tol': [0.0001],  # Tolerance for stopping criteria.
    'verbose': [0],  # Verbosity mode.
    'warm_start': [False]  # Whether to reuse the solution of the previous call.
}

# Hyperparameters for DecisionTreeClassifier model.
DecisionTreeClassifier_param = {
    'max_depth': [1, 3, 5, None],  # Maximum depth of the decision tree.
    'min_samples_split': [3, 5, 10],  # Minimum samples required to split an internal node.
    'random_state': [10, 20, None],  # Seed for randomness.
    'splitter': ['best', 'random'],  # Strategy used to choose the split at each node.
}

# Hyperparameters for GradientBoostingClassifier model.
GradientBoostingClassifier_param = {      
    'loss': ['log_loss', 'exponential'],  # Loss function to be optimized.
    'learning_rate': [0.0, 0.1, 0.2],  # Learning rate that shrinks the contribution of each tree.
    'n_estimators': [100, 200, 300],  # Number of boosting stages.
    'criterion': ['friedman_mse', 'squared_error']  # Function to measure the quality of a split.
}

# Hyperparameters for RandomForestClassifier model.
RandomForestClassifier_param = {
    'criterion': ['gini', 'entropy', 'log_loss'],  # Function to measure the quality of a split.
    'max_depth': [None, 10, 20],  # Maximum depth of each tree.
    'min_samples_leaf': [1, 2, 3],  # Minimum samples required to be at a leaf node.
    'max_features': ['sqrt', 'log2', None]  # Number of features to consider when looking for the best split.
}


if __name__ == "__main__":
    # Define model configurations as tuples with the folder name and a tuple of (model instance, hyperparameters dictionary).
    model_configs = [
        ("DecisionTreeClassifier", (DecisionTreeClassifier(), DecisionTreeClassifier_param)),  # Configuration for DecisionTreeClassifier.
        ("LogisticRegression", (LogisticRegression(), LogisticRegression_param)),  # Configuration for LogisticRegression.
        ("GradientBoostingClassifier", (GradientBoostingClassifier(), GradientBoostingClassifier_param)),  # Configuration for GradientBoostingClassifier.
        ("RandomForestClassifier", (RandomForestClassifier(), RandomForestClassifier_param))  # Configuration for RandomForestClassifier.
    ]
    
    # Evaluate and save each model based on the configurations.
    for folder_name, (model, params) in model_configs:
        print(f"Evaluating and saving model: {folder_name}")  # Inform which model is currently being processed.
        classification = ClassificationModel(model)  # Instantiate the ClassificationModel with the current model.
        classification.evaluate_model(model, params, folder_name)  # Evaluate the model and save results.
    
    # Find the best model among those saved based on test set accuracy.
    best_model, best_hyperparameters, best_metrics = find_best_model(model_configs)
    print("Best Model:")  # Print header for the best model details.
    print(best_model)  # Print the best model.
    print("Best Hyperparameters:")  # Print header for hyperparameters.
    print(best_hyperparameters)  # Print the best hyperparameters.
    print("Best Metrics:")  # Print header for performance metrics.
    print(best_metrics)  # Print the best performance metrics.
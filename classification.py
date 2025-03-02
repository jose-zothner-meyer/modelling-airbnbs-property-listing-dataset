# classification.py

# Import standard libraries for various functionalities
import itertools  # Provides functions for efficient looping
import json  # For encoding and decoding JSON data
import math  # Mathematical functions (e.g., for advanced calculations)
import os  # Operating system interfaces (e.g., file paths)
# Import plotting library for visualization
import matplotlib.pyplot as plt  # For creating visualizations
# Import numerical and data processing libraries
import numpy as np  # For numerical operations on arrays
import pandas as pd  # For data manipulation and analysis
# Import joblib for saving and loading models
import joblib  # For persisting Python objects (e.g., models)
from joblib import load  # Specific function to load saved objects
# Import classifiers from scikit-learn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # Ensemble classifiers
from sklearn.impute import SimpleImputer  # For handling missing data
from sklearn.linear_model import LogisticRegression  # Logistic regression classifier
# Import various metrics for model evaluation
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
# Import functions for model selection and splitting the dataset
from sklearn.model_selection import GridSearchCV, train_test_split
# Import scaler for standardizing features
from sklearn.preprocessing import StandardScaler
# Import decision tree classifier
from sklearn.tree import DecisionTreeClassifier
# Import custom data loader for Airbnb data
from tabular_data import load_airbnb
# Import type hints from typing module
from typing import Tuple, Dict, Any, List

class ClassificationModel:
    """
    A class for implementing and evaluating classification models using Airbnb property listing data.
    This class handles data import, preprocessing, splitting, hyperparameter tuning, and saving the model.
    """

    def __init__(self, model: Any) -> None:
        """
        Initialize the ClassificationModel with a specified model.

        Args:
            model (Any): The classification model to be used (e.g., LogisticRegression, DecisionTreeClassifier).
        """
        self.model = model  # Store the provided model instance for later use

    def import_and_standarise_data(self, data_file: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Import data from a CSV file and standardize its numeric features.

        This function reads the CSV file, extracts features and the target using a custom function,
        selects numeric features, imputes missing values with the column mean, and standardizes the data.

        Args:
            data_file (str): The file path to the CSV data file.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: A tuple containing the standardized features DataFrame and the target Series.
        """
        df: pd.DataFrame = pd.read_csv(data_file)  # Read the CSV file into a DataFrame
        X, y = load_airbnb(df)  # Extract features (X) and target (y) using custom function from tabular_data.py
        numeric_columns: pd.DataFrame = X.select_dtypes(include=[np.number])  # Select only numeric columns from features
        df_numeric: pd.DataFrame = pd.DataFrame(numeric_columns)  # Create a DataFrame with numeric data

        # Impute missing numeric values with the mean of each column
        imputer: SimpleImputer = SimpleImputer(missing_values=np.nan, strategy='mean')  # Create imputer object
        X_imputed: np.ndarray = imputer.fit_transform(df_numeric)  # Impute missing values and return as NumPy array

        # Standardize the imputed numeric data to have zero mean and unit variance
        scaler: StandardScaler = StandardScaler()  # Create scaler object
        X_scaled: np.ndarray = scaler.fit_transform(X_imputed)  # Scale the data using the standard scaler
        X_final: pd.DataFrame = pd.DataFrame(X_scaled, columns=numeric_columns.columns)  # Convert scaled data back to DataFrame
        return X_final, y  # Return the standardized features and the target labels

    def splited_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split the dataset into training, validation, and test sets.

        The split is done in two stages: first into 80% training and 20% testing,
        then the 20% test set is equally divided into validation and final test sets.

        Args:
            X (pd.DataFrame): Feature data.
            y (pd.Series): Target labels.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
                - X_train: Training features.
                - X_validation: Validation features.
                - X_test: Test features.
                - y_train: Training target labels.
                - y_validation: Validation target labels.
                - y_test: Test target labels.
        """
        np.random.seed(10)  # Set a seed for reproducibility of random operations

        # Split the data into 80% training and 20% testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=12  # random_state ensures reproducibility
        )
        # Split the 20% test set into two equal parts: validation (10%) and test (10%)
        X_test, X_validation, y_test, y_validation = train_test_split(
            X_test, y_test, test_size=0.5  # Splitting equally into validation and test sets
        )
        return X_train, X_validation, X_test, y_train, y_validation, y_test  # Return the split datasets

    def classification_metrics_performance(
        self, 
        y_test: pd.Series, 
        y_pred: pd.Series
    ) -> Dict[str, float]:
        """
        Compute performance metrics for classification.

        Metrics calculated include accuracy, precision, recall, and F1 score using macro averaging.

        Args:
            y_test (pd.Series): Actual target labels.
            y_pred (pd.Series): Predicted target labels from the model.

        Returns:
            Dict[str, float]: Dictionary containing computed accuracy, precision, recall, and F1 score.
        """
        accuracy: float = accuracy_score(y_test, y_pred)  # Calculate overall accuracy
        precision: float = precision_score(y_test, y_pred, average="macro", zero_division=0)  # Calculate macro-average precision
        recall: float = recall_score(y_test, y_pred, average="macro", zero_division=0)  # Calculate macro-average recall
        f1: float = f1_score(y_test, y_pred, average="macro", zero_division=0)  # Calculate macro-average F1 score
        return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}  # Return metrics as a dictionary

    def confusion_matrix(
        self, 
        model: Any, 
        X_test: pd.DataFrame, 
        y_test: pd.Series, 
        y_pred: pd.Series
    ) -> ConfusionMatrixDisplay:
        """
        Generate and display a confusion matrix using the provided model and test data.

        Note: The y_pred parameter is not used in this implementation since the confusion matrix
        is directly derived from the model's predictions via the estimator.

        Args:
            model (Any): The trained classification model.
            X_test (pd.DataFrame): Features from the test set.
            y_test (pd.Series): True target labels.
            y_pred (pd.Series): Predicted labels (not used here).

        Returns:
            ConfusionMatrixDisplay: A display object of the confusion matrix.
        """
        # Create a confusion matrix display from the estimator using test data
        disp: ConfusionMatrixDisplay = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
        disp.ax_.set_title("Confusion Matrix")  # Set the title of the confusion matrix plot
        return disp  # Return the display object

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
        Tune hyperparameters using GridSearchCV and evaluate the best model.

        This function performs grid search over the specified hyperparameters using 3-fold cross-validation,
        then evaluates the best model on training, validation, and test sets.

        Args:
            model (Any): The classification model to be tuned.
            X_train (pd.DataFrame): Training features.
            X_val (pd.DataFrame): Validation features.
            X_test (pd.DataFrame): Test features.
            y_train (pd.Series): Training target labels.
            y_val (pd.Series): Validation target labels.
            y_test (pd.Series): Test target labels.
            hyperparameters (Dict[str, List[Any]]): Dictionary with hyperparameter names and lists of values to try.

        Returns:
            Dict[str, Any]: Dictionary containing the best model, its hyperparameters, best cross-validation accuracy,
                            and performance metrics for training, validation, and test sets.
        """
        # Initialize GridSearchCV with the provided model and hyperparameters, using 3-fold cross-validation
        tuned_model: GridSearchCV = GridSearchCV(
            estimator=model,
            param_grid=hyperparameters,
            cv=3,  # Perform 3-fold cross-validation
            scoring='accuracy'  # Use accuracy as the metric for evaluation during grid search
        )
        tuned_model.fit(X_train, y_train)  # Fit the grid search on the training data

        best_hyperparameters: Dict[str, Any] = tuned_model.best_params_  # Retrieve the best hyperparameters found
        best_model_accuracy: float = tuned_model.best_score_  # Retrieve the best cross-validation accuracy
        best_model_estimator: Any = tuned_model.best_estimator_  # Retrieve the best model estimator

        # Make predictions on training, validation, and test sets using the best estimator
        train_y_pred: np.ndarray = best_model_estimator.predict(X_train)  # Predictions on training set
        valid_y_pred: np.ndarray = best_model_estimator.predict(X_val)  # Predictions on validation set
        test_y_pred: np.ndarray = best_model_estimator.predict(X_test)  # Predictions on test set

        # Calculate performance metrics for each dataset split
        train_metrics: Dict[str, float] = self.classification_metrics_performance(y_train, train_y_pred)
        val_metrics: Dict[str, float] = self.classification_metrics_performance(y_val, valid_y_pred)
        test_metrics: Dict[str, float] = self.classification_metrics_performance(y_test, test_y_pred)

        # Create a dictionary to store the tuning results and performance metrics
        result_dict: Dict[str, Any] = {
            "Best_Model": best_model_estimator,
            "Best_Hyperparameters": best_hyperparameters,
            "Best_Metrics": best_model_accuracy,  # Best cross-validation accuracy
            "Train_Metrics": train_metrics,
            "Validation_Metrics": val_metrics,
            "Test_Metrics": test_metrics
        }
        return result_dict  # Return the result dictionary

    def save_classification_model(
        self, 
        folder_name: str, 
        result_dict: Dict[str, Any]
    ) -> None:
        """
        Save the best classification model along with its hyperparameters and performance metrics.

        The model, hyperparameters, and metrics are saved in the following folder structure:
            ./models/classification/{folder_name}/

        Args:
            folder_name (str): Name of the folder to save the model files.
            result_dict (Dict[str, Any]): Dictionary containing the best model, hyperparameters, and metrics.
        """
        # Construct the directory path for classification models
        classification_dir: str = os.path.join(os.getcwd(), 'models', 'classification')
        folder_path: str = os.path.join(classification_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)  # Create the directory if it doesn't already exist

        best_model: Any = result_dict["Best_Model"]  # Extract the best model from the result dictionary
        best_hyperparameters: Dict[str, Any] = result_dict["Best_Hyperparameters"]  # Extract hyperparameters
        performance_metric: Dict[str, Any] = {
            "Train_Metrics": result_dict["Train_Metrics"],
            "Validation_Metrics": result_dict["Validation_Metrics"],
            "Test_Metrics": result_dict["Test_Metrics"]
        }  # Combine performance metrics into one dictionary

        # Save the best model using joblib for persistence
        joblib.dump(best_model, os.path.join(folder_path, "model.joblib"))
        # Save hyperparameters to a JSON file
        with open(os.path.join(folder_path, "hyperparameters.json"), 'w') as json_file:
            json.dump(best_hyperparameters, json_file)
        # Save performance metrics to a JSON file
        with open(os.path.join(folder_path, "metrics.json"), 'w') as json_file:
            json.dump(performance_metric, json_file)

    def evaluate_model(
        self, 
        model: Any, 
        hyperparameters_dict: Dict[str, List[Any]], 
        folder_name: str
    ) -> None:
        """
        Execute the complete evaluation pipeline for a classification model.

        This includes:
            - Loading and standardizing the data.
            - Splitting the data into training, validation, and test sets.
            - Performing hyperparameter tuning using GridSearchCV.
            - Saving the best model and its evaluation metrics.

        Args:
            model (Any): The classification model to evaluate.
            hyperparameters_dict (Dict[str, List[Any]]): Hyperparameter grid for tuning.
            folder_name (str): Folder name to save the evaluation results.
        """
        # Define the data file path
        data_file: str = os.path.join("tabular_data", "listing.csv")
        # Import and standardize the data from CSV
        X, y = self.import_and_standarise_data(data_file)
        # Split the data into train, validation, and test sets
        X_train, X_val, X_test, y_train, y_val, y_test = self.splited_data(X, y)

        # Tune hyperparameters using GridSearchCV and retrieve the best model and metrics
        result_dict: Dict[str, Any] = self.tune_hyperparameters(
            model,
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            hyperparameters_dict
        )
        # Save the best model, hyperparameters, and performance metrics to disk
        self.save_classification_model(folder_name, result_dict)

def find_best_model(
    model_configs: List[Tuple[str, Any]]
) -> Tuple[Any, Dict[str, Any], Dict[str, float]]:
    """
    Identify and return the best classification model from saved models based on test set accuracy.

    The function iterates through each model configuration folder, loads the saved model,
    hyperparameters, and test metrics, and selects the one with the highest accuracy.

    Args:
        model_configs (List[Tuple[str, Any]]): A list of tuples where each tuple contains a folder name
                                               and a model configuration.

    Returns:
        Tuple[Any, Dict[str, Any], Dict[str, float]]:
            - Best model instance.
            - Dictionary of best hyperparameters.
            - Dictionary of best performance metrics from the test set.
    """
    best_classification_model: Any = None  # Initialize variable to store the best model
    best_hyperparameters_dict: Dict[str, Any] = {}  # Initialize dictionary for best hyperparameters
    # Initialize metrics with zeros to compare later; higher accuracy will replace these values
    best_metrics_dict: Dict[str, float] = {"Accuracy": 0, "Precision": 0, "Recall": 0, "F1": 0}

    # Define the directory where classification models are saved
    classification_dir: str = os.path.join(os.getcwd(), 'models', 'classification')
    # Iterate over each model configuration provided
    for folder_name, _ in model_configs:
        model_dir: str = os.path.join(classification_dir, folder_name)  # Build the path to the model folder
        model_file: str = os.path.join(model_dir, 'model.joblib')  # Path to the saved model file
        if not os.path.exists(model_file):  # Check if the model file exists
            print(f"Skipping {folder_name}: {model_file} does not exist.")  # Inform user if file is missing
            continue  # Skip this model configuration if file is not found

        loaded_model: Any = load(model_file)  # Load the saved model from disk

        # Load the hyperparameters used for this model from a JSON file
        with open(os.path.join(model_dir, 'hyperparameters.json'), 'r') as f:
            hyperparameters: Dict[str, Any] = json.load(f)
        # Load the test metrics from a JSON file and extract the test metrics dictionary
        with open(os.path.join(model_dir, 'metrics.json'), 'r') as f:
            metrics: Dict[str, Any] = json.load(f)["Test_Metrics"]

        # Compare the accuracy of the current model with the best one so far
        if best_classification_model is None or metrics["Accuracy"] > best_metrics_dict["Accuracy"]:
            best_classification_model = loaded_model  # Update best model
            best_hyperparameters_dict = hyperparameters  # Update best hyperparameters
            best_metrics_dict = metrics  # Update best metrics

    # Return the best model along with its hyperparameters and performance metrics
    return best_classification_model, best_hyperparameters_dict, best_metrics_dict

# ------------------------------------------------------------------------------
# 1) Define hyperparameter grids for various classifiers
# ------------------------------------------------------------------------------

# Hyperparameter grid for Logistic Regression
LogisticRegression_param: Dict[str, List[Any]] = {
    'C': [1.0],  # Inverse of regularization strength
    'class_weight': ['balanced', None],  # Weighting of classes
    'dual': [False],  # Dual formulation flag; typically False for l2 penalty
    'fit_intercept': [True],  # Whether to calculate the intercept for the model
    'intercept_scaling': [1],  # Scaling of the intercept term
    'max_iter': [2000, 5000],  # Maximum number of iterations for solver convergence
    'n_jobs': [None],  # Number of CPU cores used during the training
    'penalty': ['l2'],  # Norm used in penalization (l2 for Ridge)
    'random_state': [0],  # Seed used by the random number generator
    'solver': ['lbfgs', 'saga'],  # Algorithms to use in the optimization problem
    'tol': [0.0001],  # Tolerance for stopping criteria
    'verbose': [0],  # Verbosity level for the solver
    'warm_start': [False]  # Whether to reuse the solution of the previous call to fit
}

# Hyperparameter grid for Decision Tree Classifier
DecisionTreeClassifier_param: Dict[str, List[Any]] = {
    'max_depth': [1, 3, 5, None],  # Maximum depth of the tree
    'min_samples_split': [3, 5, 10],  # Minimum number of samples required to split an internal node
    'random_state': [10, 20, None],  # Seed for randomness in tree building
    'splitter': ['best', 'random']  # Strategy to choose the split at each node
}

# Hyperparameter grid for Gradient Boosting Classifier
GradientBoostingClassifier_param: Dict[str, List[Any]] = {
    'loss': ['log_loss', 'exponential'],  # Loss function to optimize
    'learning_rate': [0.0, 0.1, 0.2],  # Learning rate shrinks the contribution of each tree
    'n_estimators': [100, 200, 300],  # Number of boosting stages
    'criterion': ['friedman_mse', 'squared_error']  # Function to measure the quality of a split
}

# Hyperparameter grid for Random Forest Classifier
RandomForestClassifier_param: Dict[str, List[Any]] = {
    'criterion': ['gini', 'entropy', 'log_loss'],  # Function to measure the quality of a split
    'max_depth': [None, 10, 20],  # Maximum depth of the tree
    'min_samples_leaf': [1, 2, 3],  # Minimum number of samples required at a leaf node
    'max_features': ['sqrt', 'log2', None]  # Number of features to consider when looking for best split
}

# ------------------------------------------------------------------------------
# 2) Define model_configs at the top level (important for `import` statements!)
# ------------------------------------------------------------------------------

from typing import Union  # Import Union for potential advanced type hints

# List of model configurations.
# Each tuple consists of a string identifier and a tuple of:
#   (model instance, hyperparameter grid for that model)
model_configs: List[Tuple[str, Tuple[Any, Dict[str, List[Any]]]]] = [
    (
        "DecisionTreeClassifier",  # Identifier for Decision Tree Classifier
        (
            DecisionTreeClassifier(),  # Create an instance of DecisionTreeClassifier
            DecisionTreeClassifier_param  # Corresponding hyperparameter grid
        )
    ),
    (
        "LogisticRegression",  # Identifier for Logistic Regression
        (
            LogisticRegression(),  # Create an instance of LogisticRegression
            LogisticRegression_param  # Corresponding hyperparameter grid
        )
    ),
    (
        "GradientBoostingClassifier",  # Identifier for Gradient Boosting Classifier
        (
            GradientBoostingClassifier(),  # Create an instance of GradientBoostingClassifier
            GradientBoostingClassifier_param  # Corresponding hyperparameter grid
        )
    ),
    (
        "RandomForestClassifier",  # Identifier for Random Forest Classifier
        (
            RandomForestClassifier(),  # Create an instance of RandomForestClassifier
            RandomForestClassifier_param  # Corresponding hyperparameter grid
        )
    )
]

def logistic_regression(
    X: pd.DataFrame, 
    y: pd.Series
) -> Tuple[pd.Series, pd.Series, LogisticRegression, pd.DataFrame]:
    """
    Train and evaluate a simple Logistic Regression model without hyperparameter tuning.

    This function splits the dataset into training and testing sets, trains a Logistic Regression model,
    makes predictions on the test set, computes accuracy, and prints the accuracy.

    Args:
        X (pd.DataFrame): The feature data.
        y (pd.Series): The target labels.

    Returns:
        Tuple[pd.Series, pd.Series, LogisticRegression, pd.DataFrame]:
            - y_test: True labels for the test set.
            - y_pred: Predicted labels for the test set.
            - clf: The trained Logistic Regression model.
            - X_test: Test set features.
    """
    # Split data into 90% training and 10% testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    clf: LogisticRegression = LogisticRegression(random_state=0, max_iter=5000)  # Initialize Logistic Regression with a fixed random state and max iterations
    clf.fit(X_train, y_train)  # Train the model using the training data
    y_pred: np.ndarray = clf.predict(X_test)  # Predict the target labels for the test set
    accuracy: float = accuracy_score(y_test, y_pred)  # Compute accuracy score on the test set
    print(f"LogisticRegression Accuracy: {accuracy:.2f}")  # Print the accuracy in a formatted string
    return y_test, y_pred, clf, X_test  # Return the test labels, predictions, trained model, and test features

# ------------------------------------------------------------------------------
# 3) Optional main block for testing standalone
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Loop through each model configuration in model_configs
    for folder_name, (model, params) in model_configs:
        print(f"Evaluating and saving model: {folder_name}")  # Notify which model is currently being processed
        classification_obj: ClassificationModel = ClassificationModel(model)  # Create an instance of ClassificationModel for the current model
        # Evaluate the model using the complete pipeline and save results in the specified folder
        classification_obj.evaluate_model(model, params, folder_name)

    # After processing all models, determine which saved model performs best on the test set
    best_model, best_hparams, best_metrics = find_best_model(model_configs)
    print("Best Model:", best_model)  # Output the best model instance
    print("Best Hyperparameters:", best_hparams)  # Output the hyperparameters of the best model
    print("Best Metrics:", best_metrics)  # Output the performance metrics of the best model

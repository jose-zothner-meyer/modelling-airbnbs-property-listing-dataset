from datetime import datetime, timedelta  # Import datetime and timedelta for time handling
import itertools  # Import for creating hyperparameter combinations
import json  # Import for saving configurations and metrics in JSON format
import numpy as np  # Import numpy for numerical operations
import os  # Import os for file path operations
from collections import OrderedDict  # Import OrderedDict for maintaining layer order in models
import pandas as pd  # Import pandas for data manipulation
import time  # Import time for timing inference latency
import torch  # Import PyTorch for building and training neural networks
import torch.nn as nn  # Import torch.nn for neural network modules and layers
import torch.nn.functional as F  # Import functional API (not used explicitly here but often useful)
import torch.optim as optim  # Import torch.optim for optimization algorithms
from sklearn.linear_model import LinearRegression  # Import scikit-learn linear regression (not used here)
from sklearn.metrics import mean_squared_error, r2_score  # Import metrics for model evaluation
from sklearn.model_selection import train_test_split  # Import for splitting data into train/test sets
from torch.utils.data import Dataset, DataLoader  # Import Dataset and DataLoader for handling data batches
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter for TensorBoard logging
import yaml  # Import yaml for reading configuration files
from typing import Any, Dict, List, Tuple  # Import type hints


# Define the dataset class for Airbnb nightly price regression
class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Initializes the dataset with features and target variable.

        Args:
            X (pd.DataFrame): Features for the dataset.
            y (pd.Series): Target variable (e.g., nightly price).
        """
        super().__init__()
        # Ensure the features and target have the same number of samples.
        assert len(X) == len(y)
        self.X: pd.DataFrame = X
        self.y: pd.Series = y
    
    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the feature and target at the given index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the feature tensor and the target tensor.
        """
        # Convert the feature row to a numpy array with data type float32.
        x_val: np.ndarray = np.array(self.X.iloc[index].values, dtype=np.float32)
        # Convert the target value to a float32 scalar.
        y_val: np.float32 = np.float32(self.y.iloc[index])
        # Return the features and target as torch tensors.
        return torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)


# Define a simple neural network model for regression tasks.
class NN(torch.nn.Module):
    """
    Neural network model for regression.

    This class defines a simple feed-forward neural network architecture for regression.
    
    Attributes:
        layers (torch.nn.Sequential): Sequential container of layers.
    """
    def __init__(self) -> None:
        super().__init__()
        # Define a sequential model with an input linear layer, ReLU activation, and output linear layer.
        self.layers: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Linear(11, 16),  # Input layer: maps 11 features to 16 neurons.
            torch.nn.ReLU(),          # Activation function.
            torch.nn.Linear(16, 1)     # Output layer: maps 16 neurons to 1 output.
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output prediction from the network.
        """
        return self.layers(X)


# Define a neural network model with a custom linear structure.
class LinearRegressionStructure(torch.nn.Module):
    """
    Linear regression model with a custom structure.

    This model is built using a user-defined configuration for the network layers.

    Attributes:
        layers (torch.nn.Sequential): The custom layers of the model.
    """
    def __init__(self, config_model_structure: OrderedDict) -> None:
        """
        Initializes the linear regression model with a custom structure.

        Args:
            config_model_structure (OrderedDict): An ordered dictionary specifying the model layers.
        """
        super().__init__()
        # Create a sequential container with the provided configuration.
        self.layers: nn.Sequential = nn.Sequential(config_model_structure)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom model.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model output.
        """
        return self.layers(X)


# Define a simple linear regression model using a single linear layer.
class LinearRegressionModel(torch.nn.Module):
    """
    Simple linear regression model.

    This model uses a single linear layer to map inputs to an output.
    
    Attributes:
        linear_layer (torch.nn.Linear): The linear layer performing the regression.
    """
    def __init__(self, input_size: int) -> None:
        """
        Initializes the linear regression model.

        Args:
            input_size (int): The number of input features.
        """
        super().__init__()
        self.linear_layer: torch.nn.Linear = torch.nn.Linear(input_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear regression model.

        Args:
            features (torch.Tensor): Input feature tensor.

        Returns:
            torch.Tensor: Predicted output.
        """
        return self.linear_layer(features)


def build_model_structure(config: Dict[str, Any]) -> OrderedDict:
    """
    Constructs the model structure based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing:
            - 'hidden_layer_width': int, the width of hidden layers.
            - 'depth': int, the number of hidden layers.

    Returns:
        OrderedDict: An ordered dictionary representing the model structure.
    """
    hidden_layer_size: int = config['hidden_layer_width']
    linear_depth: int = config['depth']

    # Create an ordered dictionary to maintain layer order.
    config_dict: OrderedDict = OrderedDict()

    # Define the input layer mapping 11 input features to the hidden layer size.
    config_dict['input'] = nn.Linear(11, hidden_layer_size)

    # Loop through the desired depth to add ReLU activations and linear layers.
    for idx in range(linear_depth):
        config_dict[f'relu{idx}'] = nn.ReLU()  # Activation layer.
        config_dict[f'layer{idx}'] = nn.Linear(hidden_layer_size, hidden_layer_size)  # Hidden layer.

    # Add an extra layer after the loop to reshape features.
    config_dict[f'layer{linear_depth}'] = nn.Linear(hidden_layer_size, 11)
    config_dict[f'relu{linear_depth}'] = nn.ReLU()  # Activation after the extra layer.

    # Define the output layer mapping from 11 features to a single output.
    config_dict['output'] = nn.Linear(11, 1)
    return config_dict


def train_neural_network(model: torch.nn.Module,
                           train_dataloader: DataLoader,
                           val_dataloader: DataLoader,
                           nn_config: Dict[str, Any],
                           epochs: int = 40) -> Tuple[torch.nn.Module, timedelta, float, str]:
    """
    Trains the neural network model.

    Args:
        model (torch.nn.Module): Neural network model to be trained.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        val_dataloader (DataLoader): DataLoader for the validation dataset.
        nn_config (dict): Configuration for training, containing:
            - 'lr': Learning rate.
            - 'optimiser': Choice of optimiser ("SGD", "Adam", or "RMSProp").
        epochs (int, optional): Number of training epochs. Defaults to 40.

    Returns:
        Tuple[torch.nn.Module, timedelta, float, str]: A tuple containing:
            - The trained model.
            - The training duration.
            - The average inference latency per sample.
            - A timestamp string for saving purposes.
    """
    lr: float = nn_config['lr']
    
    # Select the optimiser based on the provided configuration.
    if nn_config['optimiser'] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr)
    elif nn_config['optimiser'] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr)
    elif nn_config['optimiser'] == "RMSProp":
        optimizer = torch.optim.RMSProp(model.parameters(), lr)
    else:
        raise ValueError("Unsupported optimiser type provided in configuration.")

    # Define the mean squared error loss function.
    criterion = nn.MSELoss()
    writer: SummaryWriter = SummaryWriter()  # Initialize TensorBoard writer for logging.
    batch_idx: int = 0  # Counter for training batches.
    _batch_idx: int = 0  # Counter for validation batches.
    
    starting_time: datetime = datetime.now()  # Record the start time of training.
    for epoch in range(epochs):
        # Training phase: iterate over training batches.
        for batch in train_dataloader:
            features, labels = batch  # Unpack the batch into features and labels.
            optimizer.zero_grad()  # Reset gradients before backpropagation.
            predictions = model(features)  # Compute model predictions.
            # Adjust target tensor shape to match predictions ([batch_size, 1]).
            loss = criterion(predictions, labels.unsqueeze(1))
            loss.backward()  # Perform backpropagation to compute gradients.
            optimizer.step()  # Update model parameters.
            writer.add_scalar('loss', loss.item(), batch_idx)  # Log training loss to TensorBoard.
            batch_idx += 1  # Increment training batch counter.

        prediction_time_list: List[float] = []  # List to record inference latency for validation.
        # Validation phase: iterate over validation batches.
        for _batch in val_dataloader:
            features, labels = _batch
            optimizer.zero_grad()  # Reset gradients (though not used in inference).
            timer_start_ = time.time()  # Start timing inference.
            predictions = model(features)  # Compute predictions.
            timer_end_ = time.time()  # End timing inference.
            # Calculate average time per sample in the current batch.
            batch_prediction_time = (timer_end_ - timer_start_) / len(features)
            prediction_time_list.append(batch_prediction_time)
            loss = criterion(predictions, labels.unsqueeze(1))  # Compute validation loss.
            writer.add_scalar('loss_val', loss.item(), _batch_idx)  # Log validation loss.
            _batch_idx += 1  # Increment validation batch counter.

    ending_time: datetime = datetime.now()  # Record the end time of training.
    training_duration: timedelta = ending_time - starting_time  # Compute total training time.
    time_filename: str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")  # Create a timestamp for file naming.
    # Compute the average inference latency over all validation batches.
    interference_latency: float = sum(prediction_time_list) / len(prediction_time_list)

    return model, training_duration, interference_latency, time_filename


def get_nn_config() -> Dict[str, Any]:
    """
    Reads the neural network configuration from a YAML file.

    Returns:
        dict: Neural network configuration as a dictionary.
    """
    with open('nn_config.yaml', 'r') as file:
        hyperparameter: Dict[str, Any] = yaml.safe_load(file)
    return hyperparameter


def generate_nn_config() -> List[Dict[str, Any]]:
    """
    Generates all possible configurations for neural network training.

    Returns:
        list: List of dictionaries, each containing a unique configuration.
    """
    # Define a dictionary of hyperparameters and their possible values.
    combined_dictionary: Dict[str, List[Any]] = {
        'optimiser': ['SGD', 'Adam', 'RMSProp'],
        'lr': [0.0005, 0.0008, 0.00001],
        'hidden_layer_width': [16, 11, 10],
        'depth': [5, 3, 1]
    }

    config_dict_list: List[Dict[str, Any]] = []
    # Generate the Cartesian product of all hyperparameter values.
    for iteration in itertools.product(*combined_dictionary.values()):
        config_dict: Dict[str, Any] = {
            'optimiser': iteration[0],
            'lr': iteration[1],
            'hidden_layer_width': iteration[2],
            'depth': iteration[3]
        }
        config_dict_list.append(config_dict)

    return config_dict_list
    

def find_best_nn(config_dict_list: List[Dict[str, Any]],
                 train_dataloader: DataLoader,
                 validation_dataloader: DataLoader,
                 test_dataloader: DataLoader) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Finds the best neural network configuration by training and evaluating each.

    Args:
        config_dict_list (list): List of hyperparameter configurations.
        train_dataloader (DataLoader): DataLoader for the training set.
        validation_dataloader (DataLoader): DataLoader for the validation set.
        test_dataloader (DataLoader): DataLoader for the test set.

    Returns:
        Tuple[dict, dict]: A tuple containing the best evaluation metrics and the corresponding hyperparameters.
    """
    best_metrics_: Any = None
    best_model_: Any = None
    best_hyperparameters_: Any = None

    # Iterate over each hyperparameter configuration.
    for i, nn_config in enumerate(config_dict_list):
        # Build the model structure using the current configuration.
        model_structure: OrderedDict = build_model_structure(nn_config)
        # Initialize the model with the custom structure.
        model: LinearRegressionStructure = LinearRegressionStructure(model_structure)
        # Train the model using the training and validation data.
        best_model, training_duration, inference_latency, time_stamp = train_neural_network(
            model, train_dataloader, validation_dataloader, nn_config
        )

        # Calculate evaluation metrics on training, validation, and test datasets.
        train_RMSE_loss, validation_RMSE_loss, test_RMSE_loss, train_R_squared, validation_R_squared, test_R_squared = calculate_metrics(
            best_model, train_dataloader, validation_dataloader, test_dataloader
        )

        # Organize metrics into a dictionary.
        best_metrics: Dict[str, Any] = {
            'RMSE_loss': [train_RMSE_loss, validation_RMSE_loss, test_RMSE_loss],
            'R_squared': [train_R_squared, validation_R_squared, test_R_squared],
            'training_duration': training_duration,
            'inference_latency': inference_latency,
        }

        # Select the best configuration based on the highest validation R² score.
        if best_metrics_ is None or best_metrics['R_squared'][1] > best_metrics_['R_squared'][1]:
            best_model_ = best_model
            best_hyperparameters_ = nn_config
            best_metrics_ = best_metrics

        # Limit the search to 21 configurations (index 0 to 20) for efficiency.
        if i >= 20:
            break
      
    # Save the best model along with its hyperparameters and metrics.
    save_model(best_model_, best_hyperparameters_, best_metrics_, time_stamp)
    
    return best_metrics_, best_hyperparameters_
        
        
def calculate_metrics(best_model: torch.nn.Module,
                      train_loader: DataLoader,
                      validation_loader: DataLoader,
                      test_loader: DataLoader) -> Tuple[float, float, float, float, float, float]:
    """
    Calculates evaluation metrics (RMSE loss and R²) for the trained model.

    Args:
        best_model (torch.nn.Module): Trained model.
        train_loader (DataLoader): DataLoader for the training set.
        validation_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set.

    Returns:
        Tuple[float, float, float, float, float, float]: RMSE and R² for train, validation, and test sets.
    """
    def calculate_metrics_for_loader(loader: DataLoader) -> Tuple[float, float]:
        # Initialize empty arrays for true and predicted values.
        y_true: np.ndarray = np.array([]) 
        y_pred: np.ndarray = np.array([])

        # Iterate over the data loader.
        for features, labels in loader:
            features = features.to(torch.float32)
            # Flatten labels for metric calculations.
            labels = labels.to(torch.float32).flatten()
            prediction: torch.Tensor = best_model(features).flatten()
            # Collect the true and predicted values as numpy arrays.
            y_true = np.concatenate((y_true, labels.detach().numpy()))
            y_pred = np.concatenate((y_pred, prediction.detach().numpy()))
        # If any predictions are NaN, assign a high loss and zero R².
        if np.isnan(y_pred).any():
            RMSE_loss: float = 1000
            R_squared: float = 0
        else:
            # Calculate mean squared error and then derive RMSE.
            mse: float = mean_squared_error(y_true, y_pred)
            RMSE_loss = np.sqrt(mse)
            R_squared = r2_score(y_true, y_pred)

        return RMSE_loss, R_squared
    
    # Calculate metrics for each dataset.
    train_RMSE_loss, train_R_squared = calculate_metrics_for_loader(train_loader)
    validation_RMSE_loss, validation_R_squared = calculate_metrics_for_loader(validation_loader)
    test_RMSE_loss, test_R_squared = calculate_metrics_for_loader(test_loader)

    return train_RMSE_loss, validation_RMSE_loss, test_RMSE_loss, train_R_squared, validation_R_squared, test_R_squared    


def save_model(model: torch.nn.Module,
               best_hyperparameters_: Dict[str, Any],
               best_metrics: Dict[str, Any],
               time_stamp: str) -> None:
    """
    Saves the trained model, hyperparameters, and evaluation metrics to disk.

    The model's state dictionary is saved as 'model.pt', and the hyperparameters and metrics
    are saved as JSON files.

    Args:
        model (torch.nn.Module): Trained neural network model.
        best_hyperparameters_ (dict): Best hyperparameter configuration.
        best_metrics (dict): Best evaluation metrics.
        time_stamp (str): Timestamp used to create a unique directory.
    """
    # Define the destination directory for saving the model.
    dest: str = 'models/neural_networks/regression'
    # Create a new directory path that includes the timestamp.
    new_path: str = os.path.join(dest, time_stamp)
    # Create the directory along with any necessary parent directories.
    os.makedirs(new_path, exist_ok=True)
    
    # Save the model's state dictionary to 'model.pt'.
    torch.save(model.state_dict(), os.path.join(new_path, 'model.pt'))
    
    # Save the hyperparameters to a JSON file.
    hyperparameter: Dict[str, Any] = {
        'optimiser': best_hyperparameters_['optimiser'],
        'lr': best_hyperparameters_['lr'],
        'hidden_layer_width': best_hyperparameters_['hidden_layer_width'],
        'depth': best_hyperparameters_['depth']
    }
    with open(os.path.join(new_path, "hyperparameters.json"), 'w') as json_file:
        json.dump(hyperparameter, json_file)
    
    # Convert training duration to seconds (if possible).
    training_duration = best_metrics['training_duration']
    training_duration_seconds: float = training_duration.total_seconds() if hasattr(training_duration, 'total_seconds') else training_duration
    
    # Prepare the metrics data for saving.
    metrics_data: Dict[str, Any] = {
        'RMSE_loss_train': best_metrics['RMSE_loss'][0],
        'RMSE_loss_validation': best_metrics['RMSE_loss'][1],
        'RMSE_loss_test': best_metrics['RMSE_loss'][2],
        'R_squared_train': best_metrics['R_squared'][0],
        'R_squared_validation': best_metrics['R_squared'][1],
        'R_squared_test': best_metrics['R_squared'][2],
        'training_duration_seconds': training_duration_seconds,
        'inference_latency': best_metrics['inference_latency']
    }
    with open(os.path.join(new_path, "metrics.json"), 'w') as json_file:
        json.dump(metrics_data, json_file)
      
    return  # Explicit return for clarity


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads the Airbnb property listing data from 'clean_tabular_data.csv'.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing:
            - X (DataFrame): Features matrix with only numeric columns.
            - y (Series): Target variable 'Price_Night'.
    """
    # Read the CSV file into a pandas DataFrame.
    dataframe: pd.DataFrame = pd.read_csv('clean_tabular_data.csv')
    # Drop the target column to obtain the feature set.
    X: pd.DataFrame = dataframe.drop('Price_Night', axis=1)
    # Select only numeric columns for the features.
    X_numeric: pd.DataFrame = X.select_dtypes(include=[np.number])
    # Extract the target variable.
    y: pd.Series = dataframe['Price_Night']
    return X_numeric, y


def load_data_2() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads the Airbnb property listing data from 'clean_tabular_data.csv' using a different target.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing:
            - X (DataFrame): Features matrix with only numeric columns.
            - y (Series): Target variable 'bedrooms'.
    """
    dataframe: pd.DataFrame = pd.read_csv('clean_tabular_data.csv')
    # Drop the target column 'bedrooms' from the features.
    X: pd.DataFrame = dataframe.drop('bedrooms', axis=1)
    # Select only numeric columns for the features.
    X_numeric: pd.DataFrame = X.select_dtypes(include=[np.number])
    # Extract the target variable.
    y: pd.Series = dataframe['bedrooms']
    return X_numeric, y


# Load data using the defined function.
X, y = load_data_2()
dataset: AirbnbNightlyPriceRegressionDataset = AirbnbNightlyPriceRegressionDataset(X, y)

# Split the data into training (80%), testing (10%), and validation (10%) sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Further split the test set into equal parts for testing and validation.
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)

# Create dataset objects for training, validation, and testing.
train_dataset: AirbnbNightlyPriceRegressionDataset = AirbnbNightlyPriceRegressionDataset(X_train, y_train)
val_dataset: AirbnbNightlyPriceRegressionDataset = AirbnbNightlyPriceRegressionDataset(X_validation, y_validation)
test_dataset: AirbnbNightlyPriceRegressionDataset = AirbnbNightlyPriceRegressionDataset(X_test, y_test)

# Define DataLoaders for batching and shuffling data.
batch_size: int = 16 
train_dataloader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader: DataLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader: DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

if __name__ == "__main__":
    # Generate all possible neural network configurations.
    config_dict_list: List[Dict[str, Any]] = generate_nn_config()
    # Find the best configuration, train the model, and save the results.
    find_best_nn(config_dict_list, train_dataloader, val_dataloader, test_dataloader)

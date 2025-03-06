# Modelling Airbnb's Property Listing Dataset

<div style="display: flex; flex-direction: column; align-items: center;">
  <div style="width: 80%;">
    <a href="https://www.youtube.com/watch?v=Tub0xAsNzk8" target="_blank">
      <img src="https://img.youtube.com/vi/Tub0xAsNzk8/0.jpg" alt="Watch project overview on YouTube" style="width: 100%; margin-bottom: 20px;">
  </div>
  <div style="width: 80%;">
    <img src="./readme_images/Results.png" alt="Hangman lost Image" style="width: 100%;">
    </a>
  </div>
</div>

<div style="text-align: right; font-size: small;">
    <p><i>Clickable link to YT for project overview AND results on terminal after running main.py.</i></p>
</div>


## Table of Contents
- [Modelling Airbnb's Property Listing Dataset](#modelling-airbnbs-property-listing-dataset)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Regression Modelling](#regression-modelling)
  - [Classification Modelling](#classification-modelling)
  - [Neural Network Modelling](#neural-network-modelling)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Model Evaluation](#model-evaluation)
  - [License](#license)

## Description
This project aims to build a comprehensive framework for modelling Airbnb's property listing dataset. The primary objectives include data preprocessing, <b>training machine learning models for regression and classification</b> tasks, hyperparameter tuning, model evaluation, and deployment. The project <b>also focuses</b> on utilizing PyTorch <b>for neural network modelling</b> and integration of TensorBoard for visualization.

The ultimate aim of this project is to gain insights into the factors affecting property prices on Airbnb and to develop predictive models that can assist hosts in setting competitive prices and improve the overall user experience for guests.

Through this project, I learned about various aspects of data science and machine learning pipeline, including data cleaning, feature engineering, model selection, hyperparameter tuning, and model evaluation. Additionally, working with PyTorch provided valuable experience in building neural network architectures and optimizing model performance.

## Installation
To get started with the project, follow these steps:
1. Clone the repository: `git clone <repository_url>`
2. Navigate to the project directory: `cd modelling-airbnb`
3. Install the required dependencies: `pip install -r requirements.txt`

## Usage
After installing the dependencies, follow these instructions to run the project:
1. Data Preprocessing:
   - Run `python tabular_data.py` to preprocess the tabular dataset.
   - Execute `python modelling.py` to train and evaluate regression and classification models.
2. Neural Network Modelling:
   - Set up configurations in `nn_config.yaml`.
   - Run `python modelling.py` to train and evaluate PyTorch neural network models.
3. Visualization:
   - Start TensorBoard server using `tensorboard --logdir=logs` to visualize training curves and model performance.

## Project Structure
The project directory is organized as follows:
- **tabular_data.py**: Contains functions for data preprocessing and cleaning.
- **modelling.py**: Includes code for training, tuning, and evaluating machine learning models.
- **neural_networks/**: Directory for PyTorch neural network models.
- **models/**: Directory to save trained models, hyperparameters, and metrics.
- **logs/**: Directory to store TensorBoard logs.
- **README.md**: Overview of the project, instructions, and project structure.

## Regression Modelling
In the regression modelling phase, various regression models such as linear regression, decision trees, random forests, and gradient boosting are trained using sklearn. Hyperparameters for each model are fine-tuned using custom hyperparameter tuning functions and sklearn's GridSearchCV. Model performance is evaluated using key metrics such as RMSE and R^2 scores on both training and test sets.

## Classification Modelling
For classification modelling, logistic regression, decision trees, random forests, and gradient boosting classifiers are trained using sklearn. Similar to regression modelling, hyperparameters are fine-tuned, and model performance is evaluated using metrics such as F1 score, precision, recall, and accuracy.

## Neural Network Modelling
PyTorch is used for building and training neural network models. The architecture of the neural network is defined based on configurations specified in `nn_config.yaml`. Training and validation sets are used to train the model, and performance is evaluated using RMSE loss, R^2 score, and training duration.

## Hyperparameter Tuning
Hyperparameter tuning is a crucial step in model development to optimize model performance. Custom hyperparameter tuning functions are implemented for regression and classification models, which perform grid search over a range of hyperparameter values. Additionally, sklearn's GridSearchCV is utilized for fine-tuning hyperparameters.

## Model Evaluation
Model evaluation involves assessing the performance of trained models using various metrics specific to the task at hand. For regression tasks, metrics such as RMSE and R^2 score are computed, while for classification tasks, metrics like F1 score, precision, recall, and accuracy are calculated. The performance of each model is evaluated on both training and test datasets to ensure robustness and generalization.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Based on the printed results, **all three modeling pipelines (regression, classification, neural net)** are training and returning some best‐found models, but the actual metrics suggest that performance is rather poor—especially for the classification and neural‐network parts.

---

from regression import evaluate_all_models, find_best_model, models as reg_models, hyperparameters_dict
from classification import find_best_model as find_best_cls_model, model_configs
from neural_network import generate_nn_config, find_best_nn, train_dataloader, val_dataloader, test_dataloader

def main():
    # ---------------------------------------------------------------------
    # 1. Evaluate Regression Models
    # ---------------------------------------------------------------------
    print("Evaluating Regression Models...")
    # This will train each regression model (SGD, DecisionTree, RF, GBoost) 
    # across all specified hyperparameters and save the best one.
    evaluate_all_models(reg_models, hyperparameters_dict)

    # Retrieve the best regression model from disk based on the highest R^2
    reg_best_model, reg_best_hyperparams, reg_best_metrics = find_best_model(reg_models)
    print("\n--- Best Regression Model ---")
    print("Hyperparameters:", reg_best_hyperparams)
    print("Metrics:", reg_best_metrics)

    # ---------------------------------------------------------------------
    # 2. Evaluate Classification Models
    # ---------------------------------------------------------------------
    print("\nEvaluating Classification Models...")
    # This will train each classification model (DecisionTree, Logistic, GradientBoost, RF)
    # across the specified hyperparameters and save the best one.
    cls_best_model, cls_best_hyperparams, cls_best_metrics = find_best_cls_model(model_configs)
    print("\n--- Best Classification Model ---")
    print("Hyperparameters:", cls_best_hyperparams)
    print("Metrics:", cls_best_metrics)

    # ---------------------------------------------------------------------
    # 3. Evaluate Neural Network Models
    # ---------------------------------------------------------------------
    print("\nEvaluating Neural Network Models...")
    # Generate a list of possible NN configs (e.g., varying optimizers, 
    # learning rates, network depth, etc.)
    nn_config_list = generate_nn_config()

    # find_best_nn will train each configuration, save the best model, 
    # and return its metrics & hyperparameters.
    nn_best_metrics, nn_best_hyperparams = find_best_nn(
        nn_config_list, 
        train_dataloader, 
        val_dataloader, 
        test_dataloader
    )
    print("\n--- Best Neural Network Model ---")
    print("Hyperparameters:", nn_best_hyperparams)
    print("Metrics:", nn_best_metrics)

    # ---------------------------------------------------------------------
    # 4. Compare All Model Metrics
    # ---------------------------------------------------------------------
    print("\n--- Comparing All Best Model Metrics ---")
    
    # You can store them in a dictionary keyed by model-type for quick comparison
    combined_metrics = {
        "Regression": reg_best_metrics,        # e.g., { "validation_RMSE": ..., "R^2": ... }
        "Classification": cls_best_metrics,    # e.g., { "Accuracy": ..., "Precision": ..., "Recall": ..., "F1": ... }
        "Neural Network": nn_best_metrics      # e.g., { "RMSE_loss": [...], "R_squared": [...], ... }
    }
    
    # Print or process as you like:
    print("Combined Best Metrics from Each Family:")
    for model_type, metrics in combined_metrics.items():
        print(f"\nModel Family: {model_type}")
        print(metrics)

if __name__ == "__main__":
    main()

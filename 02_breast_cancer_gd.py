#!/usr/bin/env python
# Created by "Thieu" at 08:45, 24/12/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from metaperceptron import DataTransformer, MlpClassifier
from data_util import get_breast_cancer


# Function to train, test, and evaluate a model for a single seed
def run_trial(model, seed, data, metrics):
    X_train, X_test, y_train, y_test = data

    # Initialize model
    if model["optim"] != "SGD":
        net = MlpClassifier(
            hidden_layers=(15,), act_names="Tanh", dropout_rates=None, act_output=None,
            epochs=model["epoch"], batch_size=16, optim=model["optim"], optim_paras=None,
            early_stopping=False, seed=seed, verbose=False
        )
    else:
        net = MlpClassifier(
            hidden_layers=(15,), act_names="Tanh", dropout_rates=None, act_output=None,
            epochs=model["epoch"], batch_size=len(X_train), optim=model["optim"], optim_paras=None,
            early_stopping=False, seed=seed, verbose=False
        )

    # Train the model
    net.fit(X=X_train, y=y_train)

    # Predict and evaluate
    y_pred = net.predict(X_test)
    res = net.evaluate(y_test, y_pred, list_metrics=metrics)
    result = {"model_name": model["name"], "seed": seed, **res}
    return result


if __name__ == "__main__":
    ## Load data object
    X_train, X_test, y_train, y_test = get_breast_cancer()

    ## Scaling dataset
    dt = DataTransformer(scaling_methods=("minmax",))
    X_train_scaled = dt.fit_transform(X_train)
    X_test_scaled = dt.transform(X_test)

    data = (X_train_scaled, X_test_scaled, y_train, y_test)
    DATA_NAME = "breast_cancer"
    EPOCH = 750
    LIST_SEEDS = [10, 11, 12, 15, 18, 21, 24, 25, 30, 42]
    LIST_METRICS = ["AS", "PS", "RS", "F1S", "SS", "NPV"]
    PATH_SAVE = "history_gd"
    N_WORKERS = 6

    LIST_MODELS = [
        {"name": "GD-MLP", "optim": "SGD", "epoch": EPOCH, },
        {"name": "Adadelta-MLP", "optim": "Adadelta", "epoch": EPOCH, },
        {"name": "RMSprop-MLP", "optim": "RMSprop", "epoch": EPOCH, },
        {"name": "Adam-MLP", "optim": "Adam", "epoch": EPOCH, },
        {"name": "ASGD-MLP", "optim": "ASGD", "epoch": EPOCH, },
        {"name": "Adagrad-MLP", "optim": "Adagrad", "epoch": EPOCH, },
    ]
    Path(f"{PATH_SAVE}/{DATA_NAME}").mkdir(parents=True, exist_ok=True)

    # Run trials in parallel for all models and seeds
    all_epoch_losses = []
    all_results = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = []
        for model in LIST_MODELS:
            for seed in LIST_SEEDS:
                futures.append(executor.submit(run_trial, model, seed, data, LIST_METRICS))

        # Collect results as they complete
        for future in as_completed(futures):
            # epoch_losses, result = future.result()
            res = future.result()
            # all_epoch_losses.extend(epoch_losses)  # Add all epoch-wise losses for this trial
            all_results.append(res)  # Add evaluation result for this trial

    # Create DataFrames with headers
    # df_loss = pd.DataFrame(all_epoch_losses)  # Each row is a single epoch loss for a model/seed
    df_result = pd.DataFrame(all_results)  # Each row is a summary of metrics for a model/seed

    # Save DataFrames to CSV with headers
    # df_loss.to_csv(f"{PATH_SAVE}/{DATA_NAME}/df_loss.csv", index=False, header=True)
    df_result.to_csv(f"{PATH_SAVE}/{DATA_NAME}/df_result.csv", index=False, header=True)

    print(f"Done with data: {DATA_NAME}.")

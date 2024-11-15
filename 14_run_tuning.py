#!/usr/bin/env python
# Created by "Thieu" at 18:38, 14/11/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def draw_scatter(path_read, col_name, title, xlabel, exts=(".png", ), path_save=None, verbose=False):
    Path(f"{path_save}").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(f"{path_read}")

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot each split's test score
    splits = ['split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score']
    for split in splits:
        plt.plot(df[col_name], df[split], marker='o', linestyle='-', label=split)

    # Customize the chart
    plt.title(title, fontsize=17)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel('Test Score', fontsize=16)
    plt.grid(alpha=0.3)
    plt.xticks(df[col_name], rotation=45)  # Rotate x-axis labels
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()

    for ext in exts:
        plt.savefig(f"{path_save}/scatter-{col_name}{ext}", bbox_inches="tight")
    if verbose:
        plt.show()


path = "history"

draw_scatter(path_read=f"{path}/dropout-tuning.csv", col_name="param_dropout_rates",
             title="Ablation study about Dropout Rates hyper-parameter",
             xlabel="Dropout Rate", path_save=f"{path}/visual/tuner")

draw_scatter(path_read=f"{path}/activations-tuning.csv", col_name="param_act_names",
             title="Ablation study about Activation Function hyper-parameter",
             xlabel="Activation Function", path_save=f"{path}/visual/tuner")

draw_scatter(path_read=f"{path}/hidden_layers-tuning.csv", col_name="param_hidden_layers",
             title="Ablation study about Hidden Layers hyper-parameter",
             xlabel="#Number of Nodes in Hidden Layer", path_save=f"{path}/visual/tuner")





# def get_metrics(path_read, col_name, path_save):
#
#     df = pd.read_csv(f"{path_read}.csv", columns=[col_name,
#         "split0_test_score", "split1_test_score", "split2_test_score", "split3_test_score"])
#
#     # Save the results to a CSV file
#     result_df.to_csv(f"{path_save}/{data_name}/df_metrics_summary.csv")
#
#
# path = "history"
#
#
#
#
# # Example results
# param_values = [0.1, 1, 10, 100]
# mean_test_scores = [0.7, 0.85, 0.9, 0.88]
#
#
# # Line plot
# plt.figure(figsize=(8, 5))
# plt.plot(param_values, mean_test_scores, marker="o", linestyle="-", label="Test Score")
# plt.xscale('log')  # Log scale is common for hyperparameters like C or learning rates
# plt.title("Grid Search Results", fontsize=16)
# plt.xlabel("Hyperparameter Value (C)", fontsize=14)
# plt.ylabel("Mean Test Score", fontsize=14)
# plt.grid(alpha=0.3)
# plt.legend(fontsize=12)
# plt.tight_layout()
# plt.show()

















#
#
# import matplotlib.pyplot as plt
# from sklearn.model_selection import validation_curve
# from sklearn.svm import SVC
# from sklearn.datasets import load_iris
#
# # Example data
# X, y = load_iris(return_X_y=True)
# param_range = [0.1, 1, 10, 100]
# train_scores, test_scores = validation_curve(SVC(), X, y, param_name="C", param_range=param_range, cv=5)
#
# # Calculate mean and std
# train_mean = train_scores.mean(axis=1)
# train_std = train_scores.std(axis=1)
# test_mean = test_scores.mean(axis=1)
# test_std = test_scores.std(axis=1)
#
# # Plot validation curve
# plt.figure(figsize=(8, 5))
# plt.plot(param_range, train_mean, label="Training Score", color="blue", marker="o")
# plt.plot(param_range, test_mean, label="Validation Score", color="green", marker="o")
# plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.2)
# plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="green", alpha=0.2)
# plt.xscale('log')
# plt.title("Validation Curve", fontsize=16)
# plt.xlabel("C (Hyperparameter)", fontsize=14)
# plt.ylabel("Score", fontsize=14)
# plt.legend(fontsize=12)
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.show()

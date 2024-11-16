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

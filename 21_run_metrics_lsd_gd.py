#!/usr/bin/env python
# Created by "Thieu" at 08:50, 08/01/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd


def get_metrics(data_name, path_read, path_save):

    df = pd.read_csv(f"{path_read}/df_result.csv")

    # Group by 'model_name' and calculate the mean and standard deviation for each metric
    result_df = df.groupby("model_name").agg(["mean", "std"])

    # Save the results to a CSV file
    result_df.to_csv(f"{path_save}/{data_name}/df_metrics_summary.csv")


path = "history_lsd_gd"

get_metrics(data_name="cdc_diabetes", path_read=f"{path}/cdc_diabetes", path_save=path)
get_metrics(data_name="PhiUSIIL", path_read=f"{path}/PhiUSIIL", path_save=path)
get_metrics(data_name="rt_iot2022", path_read=f"{path}/rt_iot2022", path_save=path)
get_metrics(data_name="sepsis_survival", path_read=f"{path}/sepsis_survival", path_save=path)
get_metrics(data_name="skin", path_read=f"{path}/skin", path_save=path)

#!/usr/bin/env python
# Created by "Thieu" at 10:25, 14/11/2024 ----------%                                                                               
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


path = "history"

get_metrics(data_name="iris", path_read=f"{path}/iris", path_save=path)
get_metrics(data_name="breast_cancer", path_read=f"{path}/breast_cancer", path_save=path)
get_metrics(data_name="digits", path_read=f"{path}/digits", path_save=path)
get_metrics(data_name="wine", path_read=f"{path}/wine", path_save=path)
get_metrics(data_name="phoneme", path_read=f"{path}/phoneme", path_save=path)
get_metrics(data_name="waveform", path_read=f"{path}/waveform", path_save=path)
get_metrics(data_name="magic_telescope", path_read=f"{path}/magic_telescope", path_save=path)
get_metrics(data_name="diabetes", path_read=f"{path}/diabetes", path_save=path)
get_metrics(data_name="boston", path_read=f"{path}/boston", path_save=path)
get_metrics(data_name="california", path_read=f"{path}/california", path_save=path)

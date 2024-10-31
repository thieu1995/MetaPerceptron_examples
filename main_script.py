#!/usr/bin/env python
# Created by "Thieu" at 18:30, 30/10/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from metaperceptron import MlpRegressor, MlpClassifier, MhaMlpRegressor, MhaMlpClassifier

## Use standard MLP model for regression problem
regressor = MlpRegressor(hidden_size=50, act1_name="tanh", act2_name="sigmoid", obj_name="MSE",
                 max_epochs=1000, batch_size=32, optimizer="SGD", optimizer_paras=None, verbose=False)

## Use standard MLP model for classification problem
classifier = MlpClassifier(hidden_size=50, act1_name="tanh", act2_name="sigmoid", obj_name="NLLL",
                 max_epochs=1000, batch_size=32, optimizer="SGD", optimizer_paras=None, verbose=False)

## Use Metaheuristic-optimized MLP model for regression problem
print(MhaMlpClassifier.SUPPORTED_OPTIMIZERS)
print(MhaMlpClassifier.SUPPORTED_REG_OBJECTIVES)

opt_paras = {"name": "WOA", "epoch": 100, "pop_size": 30}
regressor = MhaMlpRegressor(hidden_size=50, act1_name="tanh", act2_name="sigmoid",
                 obj_name="MSE", optimizer="OriginalWOA", optimizer_paras=opt_paras, verbose=True)

## Use Metaheuristic-optimized MLP model for classification problem
print(MhaMlpClassifier.SUPPORTED_OPTIMIZERS)
print(MhaMlpClassifier.SUPPORTED_CLS_OBJECTIVES)

opt_paras = {"name": "WOA", "epoch": 100, "pop_size": 30}
classifier = MhaMlpClassifier(hidden_size=50, act1_name="tanh", act2_name="softmax",
                 obj_name="CEL", optimizer="OriginalWOA", optimizer_paras=opt_paras, verbose=True)





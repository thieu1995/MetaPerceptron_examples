# MetaPerceptron Experiments

Seaborn color:

https://www.practicalpythonfordatascience.com/ap_seaborn_palette


### Setup environment

#### Create new environment
```shell
python -m venv pve
```

*) On Windows:
```shell
.\pve\Scripts\activate
```

*) On Linux-based
```shell
source pve/bin/activate
```

#### Install requirement file
```bash
pip install -r requirements.txt
```

### Run scripts to get results

*) On Windows:
```bash
python 01_iris.py
python 02_breast_cancer.py
....
python 10_california.py
python 11_tuner.py
```

*) On Linux:
```bash
chmod +x main_run.sh
./main_run.sh
```

### Run scripts to get tables and figures
```bash
python 11_run_metrics.py
python 12_run_boxplot.py
python 13_run_convergence.py
python 14_run_tuning.py
```

# Large-scale dataset

### 1. CDC Diabetes Health Indicators
+ samples: 253680
+ features: 21
+ feature type: Categorical, Integer
+ task: Classification
+ subject: Health and Medicine
+ dataset characteristics: Tabular, Multivariate
+ link: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

### 2. PhiUSIIL Phishing URL (Website)
+ samples: 235795
+ features: 54
+ feature type: Real, Categorical, Integer
+ task: Classification
+ subject: Computer Science
+ dataset characteristics: Tabular
+ link: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
+ https://www.kaggle.com/code/rutufz/phiusiil-phishing-url-using-deep-learning-model



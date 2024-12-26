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

### 3. RT-IoT2022
+ samples: 123117
+ features: 83
+ feature type: Real, Categorical
+ task: Classification, Regression, Clustering
+ subject: Engineering
+ dataset characteristics: Tabular, Sequential, Multivariate
+ link: https://archive.ics.uci.edu/dataset/942/rt-iot2022
+ https://www.kaggle.com/code/azimuddink/azimuddink-rt-iot

### 4. Sepsis Survival Minimal Clinical Records
+ samples: 110341
+ features: 3
+ feature type: Integer
+ task: classification
+ subject: heath and medicine
+ dataset characteristics: multivariate
+ link: https://archive.ics.uci.edu/dataset/827/sepsis+survival+minimal+clinical+records

### 5. Skin Segmentation
+ samples: 245057
+ features: 3
+ feature type: Real
+ task: Classification
+ subject: Computer Science
+ dataset characteristics: Univariate
+ link: https://archive.ics.uci.edu/dataset/229/skin+segmentation

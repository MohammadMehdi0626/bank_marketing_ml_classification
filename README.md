# Bank Marketing Classification

A complete machine learning classification project on the **Bank Marketing** dataset.

The goal is to predict whether a client will subscribe to a **term deposit** (`y`) based on customer information and marketing campaign data.

## Project Overview

This project follows a complete machine-learning workflow:

1. Data preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature transformation and normalization
4. Handling class imbalance
5. Training multiple classification models
6. Ensemble learning
7. Final evaluation on unseen test data

## Dataset

The dataset contains information from direct marketing campaigns of a Portuguese banking institution. The campaigns were conducted through phone calls, and the classification target is whether the client subscribed to a term deposit.

The full dataset contains:

- **45,211 instances**
- **16 input attributes**
- **1 binary target variable (`y`)**

The official dataset description states that `bank-full.csv` contains all examples, while `bank.csv` contains a randomly selected 10% subset (4,521 instances).

### Target

`y` indicates whether the client subscribed to a term deposit:

- `yes`
- `no`

### Main feature groups

- Client information: `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`
- Current campaign information: `contact`, `day`, `month`, `duration`
- Previous campaign information: `campaign`, `pdays`, `previous`, `poutcome`

Dataset details and the original citation are provided in `bank-names.txt`.

## Project Structure

```text
bank-marketing-classification/
│
├── notebooks/
│   ├── 01_Preprocessing_EDA.ipynb
│   ├── 02_Logistic_Models.ipynb
│   ├── 03_MLP_Network.ipynb
│   ├── 04_SVC.ipynb
│   ├── 05_KNN.ipynb
│   ├── 06_Decision_Tree.ipynb
│   ├── 07_Bagging.ipynb
│   ├── 08_Random_Forest.ipynb
│   ├── 09_ExtraTrees.ipynb
│   ├── 10_AdaBoost.ipynb
│   ├── 11_GradientBoost.ipynb
│   ├── 12_XGBoost.ipynb
│   ├── 13_LightBoost.ipynb
│   ├── 14_CatBoost.ipynb
│   ├── 15_Voting.ipynb
│   ├── 16_Stack.ipynb
│   └── 17_Final_Evaluation.ipynb
│
├── bank-full.csv
├── bank-names.txt
├── requirements.txt
├── .gitignore
└── README.md
```

## Models

The project explores several classification and ensemble methods:

- Logistic Regression
- Multi-Layer Perceptron (MLP)
- SVC / NuSVC
- K-Nearest Neighbors
- Decision Tree
- Bagging
- Random Forest
- Extra Trees
- AdaBoost
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- Voting
- Stacking

## Final Evaluation

The final evaluation notebook evaluates models on the held-out test set using:

- Accuracy
- Confusion Matrix
- ROC analysis
- Recall / Sensitivity
- Precision
- Specificity
- F1-score

For example, the current final evaluation reports the following test-set results for the models included in that evaluation section:

| Model | Accuracy | Recall | Precision | Specificity | F1 |
|---|---:|---:|---:|---:|---:|
| KNN | 0.8901 | 0.3129 | 0.5535 | 0.9666 | 0.3998 |
| Logistic Regression | 0.8331 | 0.8554 | 0.4003 | 0.8302 | 0.5453 |
| MLP | 0.8831 | 0.0019 | 0.6667 | 0.9999 | 0.0038 |
| NuSVC | 0.8831 | 0.0019 | 0.6667 | 0.9999 | 0.0038 |
| RBF SVC | 0.8868 | 0.2420 | 0.5356 | 0.9722 | 0.3333 |
| Decision Tree | 0.8821 | 0.4518 | 0.4959 | 0.9391 | 0.4728 |

These values are included as a record of the current notebook results; they should be interpreted together with the class distribution and the chosen evaluation objective.

## Dataset Citation

The dataset documentation requests citation of:

> S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. Proceedings of the European Simulation and Modelling Conference (ESM'2011), pp. 117-121, Guimarães, Portugal, October 2011.

See `bank-names.txt` for the dataset's original documentation and citation information.

## Technologies

- Python
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Matplotlib
- Seaborn
- Missingno
- XGBoost
- LightGBM
- CatBoost
- MLxtend
- Joblib
- CVXPY

## Author

Mohammad Mehdi

This repository is part of a machine learning learning and portfolio path, with emphasis on understanding classification algorithms and ensemble learning through implementation and experimentation.

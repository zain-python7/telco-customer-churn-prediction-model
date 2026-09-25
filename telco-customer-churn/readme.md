# Telco Customer Churn Prediction

## Overview

This project builds a machine learning model to predict whether a telecommunications customer is likely to churn.

The project follows a complete machine learning workflow, including data cleaning, exploratory data analysis, preprocessing, model comparison, cross-validation, hyperparameter tuning, and model evaluation.

## Problem Statement

Customer churn is an important business problem for telecommunications companies. Predicting customers who are likely to leave can help businesses identify potential churners and take appropriate retention actions.

The objective of this project is to build and evaluate classification models that predict customer churn.

## Dataset

The dataset contains **7,043 customer records** and information about:

* Customer demographics
* Tenure
* Contract type
* Internet services
* Payment methods
* Monthly charges
* Total charges
* Additional services
* Churn status

The target variable is `Churn`:

* `0` = No Churn
* `1` = Churn

The dataset is imbalanced, with approximately 73% non-churn customers and 27% churn customers.

## Project Workflow

### 1. Data Cleaning

The following preprocessing steps were performed:

* Converted `TotalCharges` to numeric format
* Handled missing values created during conversion
* Removed the `customerID` identifier
* Converted the target variable into binary values
* Separated features and target
* Used stratified train/test splitting

### 2. Exploratory Data Analysis

Several visualizations were created to understand patterns in customer churn, including:

* Churn distribution
* Churn by contract type
* Tenure distribution by churn
* Monthly charges by churn
* Churn by internet service

Some important patterns observed were:

* Customers with shorter tenure showed higher churn levels.
* Contract type was strongly associated with churn.
* Customers with higher monthly charges tended to show more churn.
* Fiber optic customers represented a substantial portion of churn cases.

These observations describe relationships in the dataset and do not by themselves establish causation.

## Machine Learning Models

Four classification models were compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

The models were evaluated using **5-fold cross-validation**.

### Model Comparison

| Model               | Accuracy | Precision | Recall |        F1 |   ROC-AUC |
| ------------------- | -------: | --------: | -----: | --------: | --------: |
| Logistic Regression |    0.804 |     0.658 |  0.547 | **0.597** |     0.846 |
| Gradient Boosting   |    0.800 |     0.654 |  0.521 |     0.580 | **0.846** |
| Random Forest       |    0.788 |     0.629 |  0.490 |     0.551 |     0.823 |
| Decision Tree       |    0.717 |     0.468 |  0.479 |     0.473 |     0.642 |

For this baseline comparison, Logistic Regression produced the highest F1 score and accuracy. Gradient Boosting produced an almost identical ROC-AUC.

## Preprocessing

A `ColumnTransformer` was used to process numerical and categorical features.

### Numerical features

* SeniorCitizen
* tenure
* MonthlyCharges
* TotalCharges

Numerical features were standardized using `StandardScaler`.

### Categorical features

Categorical variables were encoded using `OneHotEncoder`.

The preprocessing and model were combined into a scikit-learn pipeline to keep preprocessing consistent and prevent data leakage.

## Hyperparameter Tuning

Logistic Regression was tuned using `GridSearchCV`.

The tested values of `C` were:

```text
0.001, 0.01, 0.1, 1, 10, 100
```

The best parameter found was:

```text
C = 10
```

The best cross-validation accuracy from the earlier tuning run was approximately:

```text
0.805
```

## Model Evaluation

The final Logistic Regression model was evaluated on the test set using:

* Confusion matrix
* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* ROC curve

The test ROC-AUC was approximately:

```text
0.84
```

At the default classification threshold of `0.5`, the model achieved approximately:

* Accuracy: **80.6%**
* Churn precision: **66%**
* Churn recall: **56%**
* Churn F1: **60%**

The confusion matrix showed that the model correctly identified many churn and non-churn customers, but it also missed a meaningful number of actual churners.

## Threshold Analysis

The project also explored changing the classification threshold.

Lowering the threshold increased the model's ability to identify churners but also increased false positives.

For example, a threshold around `0.266` produced approximately:

* Churn recall: **79%**
* Churn F1: **62%**
* Accuracy: **74%**

This demonstrates the trade-off between precision and recall in an imbalanced classification problem.

The threshold should ultimately be selected according to the business cost of false positives versus missed churners.

## Key Takeaways

* Logistic Regression performed strongly on this dataset despite being a relatively simple model.
* More complex models did not automatically perform better.
* ROC-AUC showed that the model had useful predictive separation between churn and non-churn customers.
* Accuracy alone is not sufficient for evaluating an imbalanced churn problem.
* Changing the classification threshold can substantially change recall and precision.
* Feature quality and available information can limit model performance regardless of the algorithm used.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## Project Structure

```text
telco-customer-churn/
│
├── data/
│   └── customer_churn.csv
│
├── telco_customer_churn.ipynb
├── README.md
└── requirements.txt
```

## How to Run

Clone the repository and install the required libraries:

```bash
pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook
```

Then run the cells in `telco_customer_churn.ipynb`.

## Conclusion

This project demonstrates a complete end-to-end classification workflow for customer churn prediction, from data preparation and exploratory analysis to model comparison, tuning, and evaluation.

The project also demonstrates why model evaluation should consider multiple metrics and business objectives rather than relying on accuracy alone.

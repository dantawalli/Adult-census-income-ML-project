# Adult Income Classification — Machine Learning Project

A comprehensive and reproducible machine learning project for predicting whether an individual’s annual income exceeds $50K using the Adult Census Income dataset.

This project was developed as part of a Machine Learning and Data Science project and demonstrates the complete application of the machine learning lifecycle, including data preprocessing, exploratory data analysis, feature engineering, model training, hyperparameter optimization, interpretability analysis, fairness evaluation, and responsible AI considerations.

---

# Project Objectives

The project aims to:

* develop a complete end-to-end machine learning pipeline for income classification
* perform exploratory data analysis (EDA) to understand socioeconomic patterns
* preprocess and transform structured tabular data for machine learning
* engineer meaningful features to improve predictive performance
* compare multiple supervised classification algorithms
* optimize models using hyperparameter tuning
* evaluate predictive performance using multiple classification metrics
* interpret model behavior using feature importance and SHAP analysis
* analyze fairness across sensitive demographic groups
* demonstrate reproducible and responsible machine learning practices

---

# Dataset

Dataset: Adult Census Income Dataset

Source:
[Adult Income Dataset on Kaggle](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset?utm_source=chatgpt.com)

The dataset contains demographic, educational, occupational, and financial information collected from the 1994 U.S. Census database.

## Target Variable

* `income`

  * `<=50K`
  * `>50K`

---

# Project Structure

```text
adult-income-ml-project/
│
├── data/
│
├── models/
│
├── notebooks/
│   └── adult_income_classification.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── evaluation.py
│   ├── fairness.py
│   ├── interpretability.py
│   ├── persistence.py
│   ├── prediction.py
│   └── visualization.py
│
├── requirements.txt
├── main.py
└── README.md
```

---

# Exploratory Data Analysis (EDA)

Exploratory data analysis was conducted to better understand the structure and characteristics of the dataset.

The analysis included:

* missing-value analysis
* duplicate detection
* class distribution analysis
* categorical feature analysis
* numerical feature correlation analysis
* income distribution across demographic groups
* socioeconomic relationship visualization

Several visualizations were created to identify important patterns related to income level, including relationships between education, occupation, marital status, work hours, capital gain, and demographic characteristics.

---

# Feature Engineering

Feature engineering techniques were applied to improve model performance and better represent socioeconomic patterns within the dataset.

The engineered features included:

* grouped education categories
* grouped native-country categories
* binary capital-gain indicators
* binary capital-loss indicators
* logarithmic transformations for skewed financial variables

Examples:

* `capital-gain-log`
* `capital-loss-log`
* `has-capital-gain`
* `has-capital-loss`

These transformations improved the ability of machine learning algorithms to capture non-linear income-related relationships.

---

# Data Preprocessing

A reusable Scikit-learn preprocessing pipeline was implemented to ensure consistent and reproducible data transformation.

## Numerical Features

Processed using:

* median imputation
* standard scaling

## Categorical Features

Processed using:

* most-frequent imputation
* one-hot encoding

The preprocessing pipeline was applied only on the training data to prevent data leakage and ensure valid model evaluation.

After one-hot encoding, the dataset dimensionality increased from the original feature space to **117 transformed numerical features**.

---

# Machine Learning Workflow

The project follows a complete machine learning workflow:

1. Dataset loading
2. Data cleaning
3. Exploratory Data Analysis (EDA)
4. Feature engineering
5. Data preprocessing
6. Train-test splitting
7. Baseline model construction
8. Model training
9. Hyperparameter tuning
10. Model evaluation
11. Fairness analysis
12. Model interpretability analysis
13. Model comparison
14. Best-model selection
15. Model persistence

---

# Models Implemented

The following classification algorithms were trained and evaluated:

* Baseline Classifier
* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting

Hyperparameter tuning was performed using `GridSearchCV`.

---

# Evaluation Metrics

The models were evaluated using multiple classification metrics to provide a balanced assessment of predictive performance:

* Accuracy
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)
* ROC-AUC

Confusion matrices and ROC curves were also used for deeper evaluation of classification behavior and class-separation capability.

---

# Model Interpretability

Model interpretability was analyzed using multiple approaches:

## Logistic Regression Coefficients

Used to understand the direction and magnitude of feature influence on income predictions.

## Tree-Based Feature Importance

Used to identify the most influential features within Decision Tree, Random Forest, and Gradient Boosting models.

## SHAP Analysis

SHAP (SHapley Additive exPlanations) was used to provide both global and local interpretability insights.

The analysis revealed that features related to:

* marital status
* education level
* capital gain
* age
* work hours

were among the strongest predictors of income level.

---

# Fairness and Responsible AI Analysis

The project also explored fairness-aware evaluation by analyzing model performance across demographic groups.

Fairness analysis was conducted across:

* gender groups
* race groups

The results demonstrated that predictive performance varied across demographic populations, highlighting the importance of responsible AI evaluation when working with sensitive socioeconomic data.

This analysis emphasized that strong aggregate model performance does not necessarily imply equitable performance across all groups.

---

# Best Performing Model

Among all evaluated models, the tuned Gradient Boosting classifier achieved the strongest overall predictive performance.

Performance metrics:

* Accuracy: **0.8692**
* Precision: **0.7735**
* Recall: **0.6417**
* F1 Score: **0.7015**
* MCC: **0.6231**
* ROC-AUC: **0.9252**

These results demonstrate the effectiveness of boosting-based ensemble learning methods for capturing complex non-linear relationships within structured socioeconomic datasets.

---

# Technologies and Libraries

The project was implemented using:

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* SHAP
* Joblib
* Jupyter Notebook

---

# Installation

## 1. Clone repository

```bash
git clone https://github.com/dantawalli/Adult-census-income-ML-project.git
```

## 2. Navigate to project directory

```bash
cd Adult-census-income-ML-project
```

## 3. Create virtual environment

```bash
python -m venv .venv
```

## 4. Activate virtual environment

Mac/Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

## 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Run Jupyter Notebook

```bash
jupyter notebook
```

Then open:

```text
notebooks/adult_income_classification.ipynb
```

---

## Run Python Workflow

```bash
python main.py
```

---

# Model Persistence

The best-performing trained model can be saved and reused using `joblib`.

Example:

```python
from src.persistence import save_model, load_model
```

This supports:

* reproducibility
* deployment-oriented workflows
* prediction reuse
* future model evaluation

---

# Example Prediction

```python
from src.prediction import predict_income
```

Reusable prediction functions are included for generating predictions on unseen observations.

---

# Author

Buhari Nasir Ahmad

Master’s Student — Artificial Intelligence for Sustainable Societies (AISS)

---

# License

This project is intended for academic and educational purposes.

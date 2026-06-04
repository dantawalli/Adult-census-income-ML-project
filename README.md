# Adult Income Classification Using Machine Learning

A comprehensive, reproducible, and interpretable machine learning project for predicting whether an individual's annual income exceeds $50,000 using the Adult Census Income dataset.

This project was developed as part of a Machine Learning and Data Science course and demonstrates the complete machine learning lifecycle, including exploratory data analysis, data preprocessing, feature engineering, model training, hyperparameter optimization, fairness evaluation, model interpretability, and responsible AI considerations.

---

# Project Objectives

The objectives of this project are to:

* develop a complete end-to-end machine learning pipeline for income classification;
* perform exploratory data analysis (EDA) to understand socioeconomic patterns within the dataset;
* preprocess and transform structured tabular data for machine learning;
* engineer meaningful features to improve predictive performance and interpretability;
* compare multiple supervised classification algorithms;
* optimize model performance using hyperparameter tuning;
* evaluate predictive performance using multiple classification metrics;
* interpret model behavior using feature importance and SHAP analysis;
* assess fairness across sensitive demographic groups;
* demonstrate reproducible and responsible machine learning practices.

---

# Dataset

**Dataset:** Adult Census Income Dataset

**Source:**
https://www.kaggle.com/datasets/wenruliu/adult-income-dataset

The dataset contains demographic, educational, occupational, and financial information derived from the 1994 U.S. Census database.

For reproducibility, the dataset used in this project is included directly in the `data/` directory. No external downloads, API keys, or authentication are required to run the notebook.

## Target Variable

**income**

* `<=50K`
* `>50K`

---

# Project Structure

```text
adult-income-ml-project/
│
├── data/
│   └── adult.csv
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

A comprehensive exploratory data analysis was conducted to understand the characteristics, structure, and potential limitations of the dataset.

The analysis included:

* missing-value analysis;
* duplicate-record analysis;
* target-variable distribution analysis;
* numerical feature distribution analysis;
* outlier analysis;
* categorical feature analysis;
* correlation analysis;
* numerical feature vs income analysis;
* categorical feature vs income analysis;
* statistical association testing (Chi-Square and independent sample t-tests);
* demographic representation analysis;
* dataset bias and limitation assessment.

The EDA revealed strong relationships between income and factors such as educational attainment, age, marital status, work hours, and financial activity.

---

# Feature Engineering

Feature engineering techniques were applied to improve model performance and better capture socioeconomic patterns within the dataset.

The engineered features included:

* grouped education categories;
* grouped native-country categories;
* binary capital-gain indicators;
* binary capital-loss indicators;
* investment activity indicators;
* logarithmic transformations for highly skewed financial variables.

Examples include:

* `capital-gain-log`
* `capital-loss-log`
* `has-capital-gain`
* `has-capital-loss`
* `has-capital-activity`

## Outlier Strategy

Exploratory analysis revealed that `capital-gain` and `capital-loss` exhibit extreme positive skewness, with a large concentration of zero values and a small number of very large observations.

These observations were retained because they represent legitimate financial outcomes rather than data-entry errors. Instead of removing outliers, logarithmic transformations (`log1p`) were applied to reduce skewness while preserving the underlying information and relative ordering of observations.

This approach improved the usability of financial variables while maintaining their predictive value.

---

# Data Preprocessing

A reusable Scikit-learn preprocessing pipeline was implemented to ensure consistent and reproducible data transformation.

## Numerical Features

Processed using:

* median imputation;
* standard scaling.

## Categorical Features

Processed using:

* most-frequent imputation;
* one-hot encoding.

The preprocessing pipeline is fitted only on the training data to prevent data leakage and ensure valid model evaluation.

After preprocessing, the feature space expands through one-hot encoding, enabling machine learning models to effectively utilize categorical information.

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
8. Class imbalance handling
9. Model training
10. Hyperparameter tuning
11. Model evaluation
12. Fairness analysis
13. Model interpretability analysis
14. Model comparison
15. Best-model selection
16. Model persistence

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

Model performance was evaluated using multiple complementary metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)
* ROC-AUC

Additional evaluation techniques included:

* Confusion Matrix Analysis
* ROC Curve Analysis
* Model Comparison Tables

These metrics provide a balanced assessment of predictive performance, particularly for imbalanced classification problems.

---

# Fairness and Responsible AI Analysis

The project includes fairness-aware evaluation to assess whether predictive performance varies across demographic groups.

Fairness analysis was conducted across:

* sex groups;
* race groups.

The results demonstrate that predictive performance can vary between demographic populations, highlighting the importance of evaluating machine learning systems beyond aggregate accuracy metrics.

This analysis emphasizes that strong predictive performance does not necessarily imply equitable performance across all groups.

---

# Model Interpretability

Model interpretability was analyzed using multiple complementary approaches.

## Logistic Regression Coefficients

Used to understand the direction and magnitude of feature influence on income predictions.

## Tree-Based Feature Importance

Used to identify the most influential variables within tree-based models.

## SHAP Analysis

SHAP (SHapley Additive exPlanations) was used to provide both global and local explanations of model behavior.

The analysis revealed that variables related to:

* marital status;
* education level;
* capital gain;
* age;
* working hours;

were among the strongest predictors of income level.

---

# Best Performing Model

Among all evaluated models, the tuned Gradient Boosting classifier achieved the strongest overall predictive performance.

## Performance Metrics

* Accuracy: **0.8692**
* Precision: **0.7735**
* Recall: **0.6417**
* F1 Score: **0.7015**
* MCC: **0.6231**
* ROC-AUC: **0.9252**

These results demonstrate the effectiveness of boosting-based ensemble learning methods for capturing complex non-linear relationships within structured socioeconomic datasets.

---

# Reproducibility

This project was designed to be fully reproducible and executable from a clean environment.

Reproducibility measures include:

* pinned package versions in `requirements.txt`;
* fixed random seed (`RANDOM_STATE = 42`);
* local dataset storage in the `data/` directory;
* modular implementation through the `src/` package;
* Scikit-learn preprocessing pipelines;
* prevention of data leakage through pipeline-based transformations;
* model persistence using `joblib`;
* end-to-end notebook execution using **Run All**.

The notebook can be executed without requiring external dataset downloads, API authentication, or manual path modifications.

---

# Technologies and Libraries

The project was implemented using:

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* SciPy
* SHAP
* Joblib
* Jupyter Notebook

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/dantawalli/Adult-census-income-ML-project.git
```

## 2. Navigate to Project Directory

```bash
cd Adult-census-income-ML-project
```

## 3. Create Virtual Environment

```bash
python -m venv .venv
```

## 4. Activate Virtual Environment

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Run the Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/adult_income_classification.ipynb
```

The notebook is designed to run from top to bottom using **Run All** without requiring additional setup beyond dependency installation.

## Run the Python Workflow

```bash
python main.py
```

---

# Model Persistence

The best-performing trained model can be saved and reloaded using Joblib.

Example:

```python
from src.persistence import save_model, load_model
```

This supports:

* reproducibility;
* deployment-oriented workflows;
* prediction reuse;
* future model evaluation.

---

# Example Prediction

```python
from src.prediction import predict_income
```

Reusable prediction functions are included for generating predictions on unseen observations.

---

# Project Highlights

This project demonstrates a complete machine learning workflow for structured tabular data, combining:

* rigorous exploratory data analysis;
* feature engineering and preprocessing;
* model comparison and optimization;
* fairness-aware evaluation;
* SHAP-based interpretability;
* reproducible software engineering practices;
* responsible AI considerations.

The project emphasizes not only predictive performance but also transparency, fairness, interpretability, and reproducibility in applied machine learning.

---

# Author

**Buhari Nasir Ahmad**

Master's Student — Artificial Intelligence for Sustainable Societies (AISS)

---

# License

This project is intended for academic and educational purposes.

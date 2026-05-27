# Adult Income Classification — Applied Programming for Data Science

A modular and reproducible machine learning project for predicting whether an individual's income exceeds $50K per year using the Adult Census Income dataset.

This project was developed for the Applied Programming for Data Science course and demonstrates the integration of software engineering principles with machine learning workflows.

---

# Project Objectives

The project aims to:

* build a complete machine learning workflow in Python
* perform exploratory data analysis (EDA)
* preprocess and transform structured tabular data
* compare multiple classification models
* evaluate models using appropriate metrics
* interpret predictions using SHAP
* demonstrate modular software architecture
* support reproducibility and model persistence

---

# Dataset

Dataset: Adult Census Income Dataset

Source:
[Adult Income Dataset on Kaggle](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset)

Target Variable:

* `income`

  * `<=50K`
  * `>50K`

---

# Project Structure

```text
adult-income-applied-programming/
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
│   ├── modeling.py
│   ├── evaluation.py
│   ├── persistence.py
│   ├── prediction.py
│   └── visualization.py
│
├── requirements.txt
├── main.py
└── README.md
```

---

# Applied Programming Concepts Demonstrated

This project applies several concepts from the Applied Programming course:

* modular Python project structure
* reusable functions
* object-oriented programming using `@dataclass`
* exception handling
* Scikit-learn preprocessing pipelines
* model persistence with `joblib`
* reproducible workflows
* separation of concerns
* structured project organization
* notebook-based exploratory analysis

---

# Machine Learning Workflow

The project follows this workflow:

1. Dataset loading
2. Data cleaning
3. Exploratory Data Analysis (EDA)
4. Feature engineering
5. Preprocessing pipelines
6. Model training
7. Hyperparameter tuning
8. Model evaluation
9. Model interpretation
10. Model persistence
11. Prediction reuse

---

# Models Used

The following classification models were implemented and compared:

* Baseline Classifier
* Logistic Regression
* Decision Tree
* Random Forest

Hyperparameter tuning was performed using `GridSearchCV`.

---

# Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)
* ROC-AUC

---

# Interpretability

Model interpretability was explored using:

* feature importance analysis
* SHAP (SHapley Additive exPlanations)

---

# Installation

## 1. Clone repository

```bash
git clone https://github.com/dantawalli/adult-income-applied-programming.git
```

## 2. Navigate to project

```bash
cd adult-income-applied-programming
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

## Run notebook

```bash
jupyter notebook
```

Then open:

```text
notebooks/adult_income_classification.ipynb
```

---

## Run Python workflow

```bash
python main.py
```

---

# Model Persistence

The best-performing model is saved using `joblib`.

Example:

```python
from src.persistence import save_model, load_model
```

This supports:

* reproducibility
* prediction reuse
* deployment-oriented workflows

---

# Example Prediction

```python
from src.prediction import predict_income
```

The project includes reusable prediction functions for generating predictions on new observations.

---

# Author

Buhari Nasir Ahmad

Master’s Student — Artificial Intelligence for Sustainable Societies (AISS)

---

# License

This project is for academic and educational purposes.

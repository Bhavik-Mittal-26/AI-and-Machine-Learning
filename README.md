 1) 📊 **Mini Project - Data Preprocessing & Regression Model**

---

### 📌 Introduction

This project focuses on end-to-end data preprocessing and regression model building. The main objective is to clean, transform, and prepare raw data for machine learning tasks, followed by implementing an **Ordinary Least Squares (OLS) Regression** model. It demonstrates important concepts of data cleaning, handling missing values, encoding categorical data, feature engineering, and model evaluation.

---

### 🚀 Features

* Handling missing values (numerical & categorical)
* Outlier detection and treatment using **IQR method**
* Encoding categorical features (**One-Hot Encoding**)
* Data transformation and standardization
* Feature engineering for better model performance
* Splitting dataset into training and testing
* Building and evaluating **OLS Regression models**

---

### 🛠️ Technology Used

* **Python**
* **Jupyter Notebook**
* **Libraries:**

  * Pandas
  * NumPy
  * Matplotlib / Seaborn (for visualization)
  * Scikit-learn
  * Statsmodels (for OLS Regression)

---

### 📂 How to Use

1. Clone or download the project notebook.
2. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
   ```
3. Open the notebook in Jupyter:

   ```bash
   jupyter notebook "Mini project 1.ipynb"
   ```
4. Run cells step by step to:

   * Preprocess data
   * Apply transformations
   * Train and evaluate the regression model

---

### 📖 Topics Covered

* **Data Cleaning & Preprocessing**

  * Handling duplicates
  * Missing value treatment (numerical & categorical)
  * Outlier detection (Boxplot, IQR method)
* **Data Transformation**

  * Standardization & Normalization
  * Feature engineering
* **Categorical Data Encoding**
* **Dataset Splitting (Train/Test)**
* **Model Building**

  * OLS Regression
  * Model evaluation (R², comparison between transformations)

---

### ✅ Conclusion

This project helps in understanding the workflow of **data preprocessing to model building**. By exploring multiple transformations and comparing their effects on regression accuracy, it provides **hands-on experience in preparing datasets for predictive modeling**.








 2) # 📊 Liver Disease Prediction - README

## 📌 Project Overview

This project focuses on predicting whether a liver disease case is **curable** or **not curable** based on patient medical data. The model is trained using different classification algorithms and evaluated using standard performance metrics.

## 🚀 Features

* Data preprocessing (cleaning, transformations, handling missing values)
* Implementation of multiple machine learning classification models:

  * Logistic Regression
  * Naive Bayes
  * Support Vector Machine (SVM)
  * K-Nearest Neighbors (KNN)
  * Decision Tree
  * Random Forest
* Model evaluation using:

  * Accuracy
  * Precision
  * Recall
  * F1 Score
* Performance comparison of algorithms

## 🛠️ Tech Stack

* **Language:** Python 🐍
* **Libraries:**

  * `pandas`, `numpy` → Data handling
  * `scikit-learn` → Machine Learning models & metrics
  * `matplotlib`, `seaborn` → Visualization

## 📂 Project Files

* `Predicted modeling on liver data (3).ipynb` → Main Jupyter Notebook with preprocessing, model building, and evaluation
* `README.md` → Project documentation (this file)

## 📊 Evaluation Metrics

The model is evaluated using the following:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_train_class = (y_train_pred > 0.5).astype(int)

print("Accuracy :", accuracy_score(train["Report_curable"], y_train_class))
print("Precision:", precision_score(train["Report_curable"], y_train_class))
print("Recall   :", recall_score(train["Report_curable"], y_train_class))
print("F1 Score :", f1_score(train["Report_curable"], y_train_class))
```

## 📈 Possible Improvements

* Outlier treatment
* Feature engineering
* Hyperparameter tuning
* Using ensemble methods (XGBoost, Gradient Boosting)
* Cross-validation for robust evaluation

## 🎯 Goal

To build an accurate and reliable machine learning model that can help in early prediction and categorization of liver disease outcomes, assisting healthcare professionals in decision-making.

---

✍️ **Author:** Bhavik Mittal


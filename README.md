# 🚢 Titanic Survival Prediction - End-to-End Machine Learning Pipeline

A complete, end-to-end classification machine learning project built from scratch using Python and Scikit-Learn to predict passenger survival on the Titanic dataset. 

---

## 🔍 Project Overview
Instead of just blindly running models, this project focuses on a rigorous, step-by-step data science workflow:
1. **Data Preprocessing & Safety:** Strategic handling of missing values (grouped medians for Age) and preventing data leakage by applying `StandardScaler` strictly on the training set.
2. **Model Training & Hyperparameter Tuning:** Experimenting with multiple classifiers and tuning parameters (like `max_depth` for Decision Trees) to fight overfitting.
3. **Automated Evaluation:** Writing custom Python functions to compare models across multiple metrics programmatically.

---

## 🧹 Data Preprocessing Steps
* **Feature Selection:** Dropped irrelevant or high-cardinality columns (`PassengerId`, `Name`, `Ticket`, `Cabin`, `Embarked`). Kept `Fare`, `Pclass`, `Age`, `Sex`, `SibSp`, `Parch`.
* **Missing Value Imputation:** Imputed missing values in the `Age` column using the median grouped by passenger class (`Pclass`).
* **Categorical Encoding:** Converted the `Sex` column using dictionary mapping (`male` to 0, `female` to 1).
* **Train/Test Split:** Split the dataset locally into 80% training and 20% testing data (`random_state=42`, `stratify=y`).
* **Feature Scaling:** Applied `StandardScaler` exclusively to training features and transformed the test features to avoid data leakage.

---

## 🌲 Models Explored & Tuning
* **K-Nearest Neighbors (KNN):** Trained using scaled features.
* **Decision Tree Classifier:** Tested multiple depths to prevent overfitting:
  * *Unrestricted / Depth 5:* Showed signs of memorization (high training accuracy, dropping test accuracy).
  * *Optimal Depth (`max_depth=3`):* Achieved the best generalization balance (**Train Acc: ~82.7%**, **Test Acc: ~81.0%**).
* **Random Forest Classifier:** Leveraged an ensemble of 100 trees for robust, stable predictions.

---

## 📊 Evaluation & Results
Wrote custom modular functions to evaluate and compare models on the test set:
* **Best Accuracy:** Random Forest 🏆
* **Best Precision:** Decision Tree (`max_depth=3`) 🎯
* **Best Recall:** Random Forest 🛡️
* **Best F1-Score:** Random Forest ⚖️

**The Verdict:** **Random Forest** emerged as the overall top performer due to its superior stability and generalization power, while the tuned Decision Tree proved exceptionally precise for positive survival predictions.

---

## 🛠️ Tech Stack
* **Language:** Python 3.13
* **Libraries:** Scikit-Learn, Pandas, NumPy, Matplotlib
* **Environment:** Jupyter Notebook / VS Code (Windows PowerShell)

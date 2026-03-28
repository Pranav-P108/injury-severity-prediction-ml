# 🚗 Injury Severity Prediction using Machine Learning

## 📌 Overview

This project develops an end-to-end machine learning pipeline to predict **injury severity levels** from road accident data. The goal is to understand how different factors influence injury outcomes and to build models that can accurately classify severity levels.

The project focuses not only on predictive performance but also on handling **class imbalance** and improving **model interpretability**.

---

## 📊 Dataset

The dataset contains accident-related features such as:

* Demographics (e.g., age, sex)
* Crash characteristics
* Vehicle and safety features
* Environmental and contextual factors

The target variable represents **multi-class injury severity levels**, including:

* Fatal Injury
* Incapacitating Injury
* Non-incapacitating Injury
* Possible Injury
* No Injury
* Others (rare classes)

---

## ⚙️ Approach

### 1. Data Preprocessing

* Handling missing values
* One-hot encoding of categorical features
* Feature scaling (for linear models)

### 2. Train-Test Split

* Stratified split to preserve class distribution

### 3. Model Training

The following models were trained and compared:

* Logistic Regression
* Linear Support Vector Classifier (SVC)
* Random Forest
* Gradient Boosting
* Dummy Classifier (baseline)

### 4. Hyperparameter Tuning

* GridSearchCV with cross-validation

### 5. Evaluation Metrics

* Accuracy
* **Macro-F1 Score (primary metric)**

---

## 📈 Results

| Model               | Accuracy | Macro-F1   |
| ------------------- | -------- | ---------- |
| Gradient Boosting   | 0.7974   | 0.5478     |
| Random Forest       | 0.7758   | **0.5486** |
| Logistic Regression | 0.7808   | 0.5057     |
| Linear SVC          | 0.7790   | 0.5025     |
| Dummy Classifier    | 0.4171   | 0.0736     |

📌 **Key Insight:**
Although accuracy is high (~0.78–0.80), it is misleading due to class imbalance.
**Macro-F1 is a better indicator of true performance.**

---

## 🧠 Model Insights

### 🔹 Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

* Majority class (**Fatal Injury**) is predicted very well
* Rare classes are often misclassified
* Strong confusion between neighbouring injury levels

---

### 🔹 Feature Importance (Random Forest)

![Feature Importance](images/feature_importance.png)

* **AGE** is the most influential feature
* Hospital-related and safety-related variables also contribute significantly
* The model relies on multiple interacting factors rather than a single variable

---

## ⚠️ Challenges

* Severe **class imbalance**
* Poor performance on minority classes
* Over-reliance of models on dominant categories

---

## 🚀 Future Improvements

* Apply resampling techniques (e.g., SMOTE)
* Use advanced models (XGBoost, LightGBM, CatBoost)
* Improve minority class detection
* Add explainability using SHAP values
* Perform deeper feature engineering

---

## 🧩 Key Takeaways

* Accuracy alone is not reliable for imbalanced datasets
* Macro-F1 provides a more balanced evaluation
* Tree-based models outperform linear models in this problem
* Interpretability is essential for real-world applications

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* Pandas, NumPy
* Matplotlib

---

## 📂 Project Structure

```
injury-severity-prediction-ml/
│
├── FARS_classification.ipynb
├── README.md
├── requirements.txt
└── images/
```

---

## 📬 Contact

If you’d like to discuss this project or collaborate, feel free to reach out!

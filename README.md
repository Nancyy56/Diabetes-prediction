# 🩺 Diabetes Prediction Using Machine Learning

> A machine learning project to predict whether a patient is diabetic or not using medical data such as glucose level, BMI, blood pressure, and age.

---

## 📘 Project Overview

This project aims to predict the likelihood of diabetes in a patient based on diagnostic health parameters.  
Using the **Pima Indians Diabetes Dataset**, multiple machine learning algorithms were trained, compared, and evaluated.  
The goal is to assist in **early detection** of diabetes, enabling doctors and patients to take timely preventive measures.

---

## 🎯 Objectives

- Analyze medical data and detect diabetes risk.
- Build and compare machine learning models (Logistic Regression, Random Forest, SVM).
- Evaluate models using metrics like Accuracy, Precision, Recall, and F1-score.
- Select and deploy the best-performing model for prediction.
- Demonstrate how AI can support healthcare diagnosis.

---

## 🧠 Tech Stack

| Category | Technologies Used |
|-----------|-------------------|
| **Language** | Python |
| **Libraries** | pandas, NumPy, matplotlib, seaborn, scikit-learn |
| **Algorithms** | Logistic Regression, Random Forest, Support Vector Machine |
| **Dataset** | Pima Indians Diabetes Dataset (Kaggle/UCI Repository) |
| **IDE** | Jupyter Notebook / VS Code |

---

## 📊 Dataset Information

**Dataset Name:** Pima Indians Diabetes Database  
**Total Records:** 768  
**Attributes:**

| Feature | Description |
|----------|--------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body mass index (weight/height²) |
| DiabetesPedigreeFunction | Family history index |
| Age | Patient age |
| Outcome | Class (0 = Non-Diabetic, 1 = Diabetic) |

---

## ⚙️ Workflow

1. **Data Loading & Exploration**  
   - Load data with pandas, check shape, missing values, and correlations.

2. **Data Preprocessing**  
   - Handle missing values using mean/median.
   - Apply feature scaling using `StandardScaler`.
   - Perform train-test split (80–20 stratified).

3. **Feature Engineering**  
   - Feature selection using correlation.
   - Create BMI/Glucose ratio and apply log transformations.
   - Encode categorical variables if necessary.

4. **Model Training**  
   - Train three ML models:
     - Logistic Regression
     - Random Forest Classifier
     - Support Vector Machine (SVM)

5. **Model Evaluation**  
   - Evaluate using Accuracy, Precision, Recall, F1-score, and AUC.
   - Plot Confusion Matrix and ROC Curve.
   - Compare and select the best model.

6. **Prediction**  
   ```python
   new_data = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]
   new_data_scaled = scaler.transform(new_data)
   prediction = rf_model.predict(new_data_scaled)
   print("Predicted Class:", prediction)

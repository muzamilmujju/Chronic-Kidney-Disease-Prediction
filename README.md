# Chronic Kidney Disease (CKD) Prediction using Machine Learning

This project aims to predict the likelihood of Chronic Kidney Disease (CKD) in patients using machine learning models trained on clinical data. Early detection of CKD is critical for effective treatment, and this system provides a reliable ML-based prediction tool to support healthcare professionals.

---

## 🚀 Features

* Multiple machine learning models trained and evaluated
* Best performance achieved with **XGBoost (98% accuracy)**
* Data preprocessing with missing value handling and feature encoding
* Visualizations for EDA and model performance
* Web-based prediction interface using Flask
* Easily extendable and modular codebase

---

## 🧠 Machine Learning Workflow

1. **Data Collection**
   Dataset used: [CKD Dataset](https://github.com/muzamilmujju/Chronic-Kidney-Disease-Prediction/blob/main/Python%20Jupyter%20Notebook/kidney_disease.csv)

2. **Data Preprocessing**

   * Handle missing values
   * Encode categorical variables
   * Normalize/scale features

3. **Model Training**

   * Logistic Regression
   * Random Forest
   * Support Vector Machine
   * K-Nearest Neighbors
   * **XGBoost (Best performer)**

4. **Evaluation Metrics**

   * Accuracy, Precision, Recall, F1-Score
   * ROC-AUC Curve
   * Cross-validation

5. **Deployment**

   * Simple Flask web application
   * User inputs patient data to get real-time prediction

---

## 📷 Executed Screenshots

Here are some screenshots of the application in action:

### 1. Home Page

![Home Page](Screenshot%20(26).png)

### 2. Kidney Disease Input Form

![Input Form](Screenshot%20(27).png)

### 3. Invalid Values:

![Input Form](Screenshot%20(28).png)


### 4. Prediction Result - No Disease

![No Disease](Screenshot%20(29).png)

---


## 📊 Model Performance

| Model               | Accuracy | Precision | Recall   | F1-Score |
| ------------------- | -------- | --------- | -------- | -------- |
| **XGBoost**         | **98%**  | **0.97**  | **0.99** | **0.98** |
| Random Forest       | 96%      | 0.95      | 0.97     | 0.96     |
| Logistic Regression | 94%      | 0.93      | 0.94     | 0.93     |
| SVM                 | 93%      | 0.92      | 0.91     | 0.91     |
| KNN                 | 91%      | 0.89      | 0.90     | 0.89     |

> ✅ **XGBoost achieved the highest accuracy of 98%, making it the best model for CKD prediction.**

---

## 💻 Tech Stack

* **Python 3.8+**
* **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
* **Web Framework**: Flask
* **Frontend**: HTML, CSS

---


## ⚙️ Installation Guide

Follow these steps to set up and run this project on your local machine.

---

### 📁 Step 1: Clone the Repository

Clone the project repository to your local system using Git:

```bash
git clone https://github.com/yourusername/ckd-prediction.git
cd ckd-prediction
```

> Replace `yourusername` with your actual GitHub username.

---

### ⚙️ Steps 2–5: Setup, Dependencies, Dataset & Run

Follow these combined steps to set up the virtual environment, install dependencies, prepare the dataset, and run the project.

#### ✅ On **Windows**:

```bash
:: Step 2: Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

:: Step 3: Install dependencies
pip install -r requirements.txt

:: (If requirements.txt is missing)
:: pip install numpy pandas scikit-learn matplotlib seaborn jupyter

:: Step 4: Make sure ckd.csv is in the project root directory

:: Step 5: Run the notebook or script
jupyter notebook
:: Then open ckd_prediction.ipynb in your browser

:: Or run directly as Python script
python main.py
```

#### ✅ On **macOS/Linux**:

```bash
# Step 2: Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Step 3: Install dependencies
pip install -r requirements.txt

# (If requirements.txt is missing)
# pip install numpy pandas scikit-learn matplotlib seaborn jupyter

# Step 4: Make sure ckd.csv is in the project root directory

# Step 5: Run the notebook or script
jupyter notebook
# Then open ckd_prediction.ipynb in your browser

# Or run directly as Python script
python main.py
```

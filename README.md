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

Follow the steps below to set up and run this project on your local machine.

---

### 📁 Step 1: Clone the Repository

Clone the project to your local system using Git.

git clone https://github.com/yourusername/ckd-prediction.git
cd ckd-prediction
Replace yourusername with your actual GitHub username.

### 🧪 Step 2: Create and Activate a Virtual Environment (Recommended)
Using a virtual environment prevents dependency conflicts.

On Windows:
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate
On macOS/Linux:
bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate
### 📦 Step 3: Install Required Dependencies
Install the required Python packages.

Option 1: Using requirements.txt
If requirements.txt is present:

bash
Copy
Edit
pip install -r requirements.txt
Option 2: Manual Installation
If requirements.txt is missing, install manually:

bash
Copy
Edit
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
### 🧾 Step 4: Prepare the Dataset
Ensure you have the dataset file named ckd.csv in the root folder.

If you don’t have it, download it from a reliable source like Kaggle CKD Dataset and place it inside your project directory.

### ▶️ Step 5: Run the Project
Option 1: If Using a Jupyter Notebook
bash
Copy
Edit
jupyter notebook
Open ckd_prediction.ipynb in the browser and run all cells.

Option 2: If Using a Python Script
bash
Copy
Edit
python main.py

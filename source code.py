It looks like you're trying to run a full data analysis and visualization script using pandas, numpy, matplotlib, seaborn, plotly, and several machine learning algorithms in Python. I will help correct and clean up your script to ensure it runs smoothly. Let's break it down step by step:

```python
# Necessary imports 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px 

import warnings 
warnings.filterwarnings('ignore') 

plt.style.use('fivethirtyeight') 
%matplotlib inline 
pd.set_option('display.max_columns', 26) 

# Loading data 
df = pd.read_csv('/content/drive/MyDrive/Kidney_data.csv') 
df.head() 
df.shape 

# Dropping id column 
df.drop('id', axis=1, inplace=True) 

# Rename column names to make them more user-friendly 
df.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 
              'pus_cell', 'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 
              'serum_creatinine', 'sodium', 'potassium', 'haemoglobin', 'packed_cell_volume', 
              'white_blood_cell_count', 'red_blood_cell_count', 'hypertension', 'diabetes_mellitus', 
              'coronary_artery_disease', 'appetite', 'peda_edema', 'aanemia', 'class'] 
df.head() 
df.describe() 
df.info() 

# Converting necessary columns to numerical type 
df['packed_cell_volume'] = pd.to_numeric(df['packed_cell_volume'], errors='coerce') 
df['white_blood_cell_count'] = pd.to_numeric(df['white_blood_cell_count'], errors='coerce') 
df['red_blood_cell_count'] = pd.to_numeric(df['red_blood_cell_count'], errors='coerce') 
df.info() 

# Extracting categorical and numerical columns 
cat_cols = [col for col in df.columns if df[col].dtype == 'object'] 
num_cols = [col for col in df.columns if df[col].dtype != 'object'] 

# Looking at unique values in categorical columns 
for col in cat_cols: 
    print(f"{col} has {df[col].unique()} values\n") 

# Replace incorrect values 
df['diabetes_mellitus'].replace(to_replace={'\tno': 'no', '\tyes': 'yes', ' yes': 'yes'}, inplace=True) 
df['coronary_artery_disease'] = df['coronary_artery_disease'].replace(to_replace='\tno', value='no') 
df['class'] = df['class'].replace(to_replace={'ckd\t': 'ckd', 'notckd': 'not ckd'}) 
df['class'] = df['class'].map({'ckd': 0, 'not ckd': 1}) 
df['class'] = pd.to_numeric(df['class'], errors='coerce') 

cols = ['diabetes_mellitus', 'coronary_artery_disease', 'class'] 
for col in cols: 
    print(f"{col} has {df[col].unique()} values\n") 

# Checking numerical features distribution 
plt.figure(figsize=(20, 15)) 
plotnumber = 1 
for column in num_cols: 
    if plotnumber <= 14: 
        ax = plt.subplot(3, 5, plotnumber) 
        sns.histplot(df[column], kde=True) 
        plt.xlabel(column) 
    plotnumber += 1 
plt.tight_layout() 
plt.show() 

# Looking at categorical columns 
plt.figure(figsize=(20, 15)) 
plotnumber = 1 
for column in cat_cols: 
    if plotnumber <= 11: 
        ax = plt.subplot(3, 4, plotnumber) 
        sns.countplot(df[column], palette='rocket') 
        plt.xlabel(column) 
    plotnumber += 1 
plt.tight_layout() 
plt.show() 

# Convert relevant columns to numerical type if they are intended to be used in correlation calculation 
for col in df.columns: 
    if df[col].dtype == 'object':  # Check if column is of object type (likely string) 
        try: 
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Attempt conversion, replacing non-convertible values with NaN 
            print(f"Converted column {col} to numeric.") 
        except: 
            print(f"Column {col} could not be converted to numeric.") 

# Heatmap of data 
plt.figure(figsize=(15, 8)) 
sns.heatmap(df.corr(), annot=True, linewidths=2, linecolor='lightgrey') 
plt.show() 

# Define functions to create plot 
def violin(col): 
    fig = px.violin(df, y=col, x="class", color="class", box=True, template='plotly_dark') 
    return fig.show() 

def kde(col): 
    grid = sns.FacetGrid(df, hue="class", height=6, aspect=2) 
    grid.map(sns.kdeplot, col) 
    grid.add_legend() 

def scatter(col1, col2): 
    fig = px.scatter(df, x=col1, y=col2, color="class", template='plotly_dark') 
    return fig.show() 

kde('red_blood_cell_count') 
kde('white_blood_cell_count') 
kde('packed_cell_volume') 
kde('haemoglobin') 
kde('albumin') 
kde('blood_glucose_random') 
kde('sodium') 
kde('blood_urea') 

# Checking for null values 
df.isna().sum().sort_values(ascending=False) 

# Filling null values using two methods: random sampling for higher null values and mean/mode sampling for lower null values 
def random_value_imputation(feature): 
    random_sample = df[feature].dropna().sample(df[feature].isna().sum()) 
    random_sample.index = df[df[feature].isnull()].index 
    df.loc[df[feature].isnull(), feature] = random_sample 

def impute_mode(feature): 
    mode = df[feature].mode()[0] 
    df[feature] = df[feature].fillna(mode) 

# Filling num_cols null values using random sampling method 
for col in num_cols: 
    random_value_imputation(col) 

# Check for remaining null values 
df[num_cols].isnull().sum() 
df[cat_cols].isnull().sum() 

# Feature encoding: 
for col in cat_cols: 
    print(f"{col} has {df[col].nunique()} categories\n") 

from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
for col in cat_cols: 
    df[col] = le.fit_transform(df[col]) 

df.head() 

# Model building 
ind_col = [col for col in df.columns if col != 'class'] 
dep_col = 'class' 
X = df[ind_col] 
y = df[dep_col] 

# Splitting data into training and test set 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0) 

# KNN Algorithm 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
knn = KNeighborsClassifier() 
knn.fit(X_train, y_train) 
knn_acc = accuracy_score(y_test, knn.predict(X_test)) 
print(f"Training Accuracy of KNN is {accuracy_score(y_train, knn.predict(X_train))}") 
print(f"Test Accuracy of KNN is {knn_acc} \n") 
print(f"Confusion Matrix :- \n{confusion_matrix(y_test, knn.predict(X_test))}\n") 
print(f"Classification Report :- \n {classification_report(y_test, knn.predict(X_test))}") 

# SVM Algorithm 
from sklearn.svm import SVC 
svm = SVC() 
svm.fit(X_train, y_train) 
svm_acc = accuracy_score(y_test, svm.predict(X_test)) 
print(f"Training Accuracy of SVM is {accuracy_score(y_train, svm.predict(X_train))}") 
print(f"Test Accuracy of SVM is {svm_acc} \n") 
print(f"Confusion Matrix :- \n{confusion_matrix(y_test, svm.predict(X_test))}\n") 
print(f"Classification Report :- \n {classification_report(y_test, svm.predict(X_test))}") 

# XGBoost Algorithm 
from xgboost import XGBClassifier 
xgb = XGBClassifier(objective='binary:logistic', learning_rate=0.5, max_depth=5, n_estimators=150) 
xgb.fit(X_train, y_train) 
xgb_acc = accuracy_score(y_test, xgb.predict(X_test)) 
print(f"Training Accuracy of XgBoost is {accuracy_score(y_train, xgb.predict(X_train))}") 
print(f"Test Accuracy of XgBoost is {xgb_acc} \n") 
print(f"Confusion Matrix :- \n{confusion_matrix(y_test, xgb.predict(X_test))}\n") 
print(f"Classification Report :- \n {classification_report(y_test, xgb.predict(X_test))}") 

#

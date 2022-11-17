# Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os

from scipy.stats import chi2_contingency

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold

from IPython.display import display
from tqdm.auto import tqdm

print('Starting training the model')

# Parameters
C = 0.1
output_file = f'model_logistic_regression.bin'

df = pd.read_csv('Telco-Customer-Churn.csv')

# Change Churn into 1 (Yes) and 0 (No)
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

# Change data type from int64 to object
df['SeniorCitizen'] = df['SeniorCitizen'].astype('object')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Data splitting
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
y_test = df_test['Churn']
del df_test['Churn']

# Data cleaning
df_full_train.TotalCharges.fillna(df_full_train.TotalCharges.median(), inplace=True)

# Extract only numeric features
num_feat = list(df.select_dtypes(include=['int64', 'float64']).columns)
cat_feat = list(df_full_train.select_dtypes(include=['object']).columns)

# Feature selection
cat_feat.remove('customerID')
cat_feat.remove('gender')
cat_feat.remove('PhoneService')
num_feat.remove('Churn')

final_feat = num_feat + cat_feat

# Data processing
dicts_full_train = df_full_train[final_feat].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)

# Modelling
model = LogisticRegression(solver="liblinear", C=0.1, 
                           max_iter=1000, random_state=42)

model.fit(X_full_train, df_full_train.Churn)


# Saving the Model
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f"the model is saved to {output_file} ")

print('Training Completed')
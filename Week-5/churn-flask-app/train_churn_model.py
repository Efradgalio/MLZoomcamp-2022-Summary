#!/usr/bin/env python
# coding: utf-8

# # Main Code

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# In[2]:


path = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(path)


# In[3]:


df.columns = df.columns.str.lower().str.replace(' ', '_')
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')
    
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


# In[4]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


# In[5]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [ 'gender', 'seniorcitizen', 'partner', 'dependents',
        'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']


# In[6]:


def train(df,y_train, C=1):
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model


# In[7]:


def predict(df,dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:,1]
    return y_pred


# In[8]:


C = 1.0
n_splits = 5


# In[17]:


kfold = KFold(n_splits=n_splits, shuffle = True, random_state=1)
scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
    
    y_train = df_train.churn.values
    y_val = df_val.churn.values
    
    dv, model = train(df_train, y_train)
    y_pred = predict(df_val, dv, model)
    
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('C={0} {1} +- {2}'.format(C, np.mean(scores).round(3), np.std(scores).round(3)))


# In[14]:


dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_predict = predict(df_test, dv, model)

y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_predict)
auc


# # Saving and Loading the Model

# ## Saving the Model

# In[4]:


import pickle


# In[18]:


output_file = f'model_C={C}.bin'
output_file


# In[20]:


f_out = open(output_file, 'wb') # will write a binary (not text but bytes)
pickle.dump((dv, model), f_out)
f_out.close()


# In[21]:


with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)
    # inside this the file is open
    
# outside the file is closed, so it easier for us to not accedentally forget to close the file


# ## Load the Model

# In[2]:


model_file = f'model_C=1.0.bin'


# In[5]:


with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)
    
# we don't need to import scikit-learn, but we need scikit-learn installed in our system,
# so it will know what model and dv means.


# In[6]:


model


# In[7]:


dv


# In[ ]:





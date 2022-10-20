import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

df = pd.read_csv('https://raw.githubusercontent.com/alexeygrigorev/datasets/master/AER_credit_card_data.csv')
df['card'] = np.where(df.card == 'yes', 1, 0)

features = ['reports', 'share', 'expenditure', 'owner']
dicts = df[features].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X = dv.fit_transform(dicts)
y = df['card']

model = LogisticRegression(solver='liblinear').fit(X, y)

# Saving the model
output_file = f'model1.bin'
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)
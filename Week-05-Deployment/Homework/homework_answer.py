# Question 1
# - Install pipenv
# - What's the version of pipenv you installed?
# - Use --version to find out

## pipenv --version --> check it on terminal

# Answer - pipenv, version 2022.10.12

# Question 2
# - Use pipenv to install scikit-learn version 1.0.2
# - What's the first hash for scikit-learn you get in pipfile.lock?

# Answer - sha256:08ef968f6b72033c16c479c966bf37ccd49b06ea91b765e1cc27afefe723920b

# Question 3
# - Use the model that already been saved from train.py
# - Write a script for loading these models with pickle
# - Score this client: {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}
# - What's the probability that this client will get a credit card?

import pickle 

model_file = f'model1.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

customer = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}

X = dv.transform([customer])
y_pred = model.predict_proba(X)[0,1]
print(y_pred)

# Answer - close to 0.162

# Question 4
# - Let's serve this model as a web service
# - Install Flask and gunicorn
# - Now score this client using requests:

"""
url = "YOUR_URL"
client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
requests.post(url, json=client).json()
"""

# - What's the probability that this client will get a credit card?

# Answer - close to 0

# Question 5
# - Build docker image
# - What's the size of this base image?

# Answer - 125 Mb
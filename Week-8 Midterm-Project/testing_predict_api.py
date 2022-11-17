import requests

url = 'http://localhost:9696/predict'
customer_id = 'UO02-AB'

customer = {'tenure': 41,
  'MonthlyCharges': 79.85,
  'TotalCharges': 3320.75,
  'SeniorCitizen': 0,
  'Partner': 'No',
  'Dependents': 'No',
  'MultipleLines': 'No',
  'InternetService': 'DSL',
  'OnlineSecurity': 'Yes',
  'OnlineBackup': 'No',
  'DeviceProtection': 'Yes',
  'TechSupport': 'Yes',
  'StreamingTV': 'Yes',
  'StreamingMovies': 'Yes',
  'Contract': 'One year',
  'PaperlessBilling': 'Yes',
  'PaymentMethod': 'Bank transfer (automatic)'}

response = requests.post(url, json=customer).json()
print(response)

if response['churn'] == True:
    print('sending promo email to {}'.format(customer_id))
else:
    print('not sending promo email to {}'.format(customer_id))
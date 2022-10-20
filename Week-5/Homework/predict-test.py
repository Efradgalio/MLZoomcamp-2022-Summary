import requests

url = "http://localhost:9696/predict"
# client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
client6 = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}

response = requests.post(url, json=client6).json()
print(response)

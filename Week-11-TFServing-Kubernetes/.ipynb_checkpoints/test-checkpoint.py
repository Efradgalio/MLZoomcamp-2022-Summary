import requests

# url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
# url = "https://tjskciesci.execute-api.ap-southeast-1.amazonaws.com/test/predict"
url = 'http://localhost:9696/predict'

data = {'url':  'http://bit.ly/mlbookcamp-pants'}

result = requests.post(url, json=data).json()
print(result)
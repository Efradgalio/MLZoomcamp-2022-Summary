# Telco Churn Prediction
![alt text](https://www.retently.com/wp-content/uploads/2015/11/leading-causes-of-churn-1.png)

“Customer Churn” refers to the loss of customers. That is, if a customer or a client stops taking services from a company, it is said that he/she has churned.

Churn is intimately connected to a company’s performance. The more one learns about customers’ behavior, the more money one can make. Analyzing customer churn also aids in finding and improving the shortcomings of services provided by the company.

This project is a solution for companies who wants to learn which customers are more likely to churn and then the company can take decision for retaining those customers in order to make more money.

## Getting Started
Simply run the following code.

First create the docker first inside your machine.
```
docker build -t <tag_name> .
```

After that, run the following to start the churn web service.
```
docker run –it –rm -p 9696:9696 <tag_name>
```

## Testing the Web Service
Make sure the web service is on, then simply run the following code.
```
python testing_predict_api.py
```

Note: You can change the values inside testing_predict_api.py to play with and get the different result from the web service

# 6. Decision Trees and Ensemble Learning

## 6.1 Session Overview: Credit Risk Scoring Project

In this session we'll learn about decision trees and ensemble learning algorithms. How we can implement and fine-tune these models to make binary classification predictions.

To be specific, we'll use [credit scoring data](https://github.com/gastonstat/CreditScoring) to make model predictions whether the bank should lend the loan to the client or not. The bank makes these decisions based on the historical record.

In the credit scoring classification problem, if the model returns 0 that means the client will payback and if the probability is 1 then its the default client.

## 6.2 Data Cleaning and Preparation

In this section we'll clean and prepare the dataset for the model which involves the following steps:

- Download the data from the given link.
- Reformat categorical columns (`status`, `home`, `marital`, `records`, and `job`) by mapping with appropriate values.
- Replace the maximum value of `income`, `assests`, and `debt` columns with NaNs.
- Extract only those rows in the column `status` who are either ok or default as value.
- Split the data with the distribution of 80% train, 20% validation, and 20% test sets with random seed to `11`.
- Prepare target variable `status` by converting it from categorical to binary, where 0 represents `ok` and 1 represents `default`.
- Finally delete the target variable from the train/val/test dataframe.

## 6.3 Decision Trees


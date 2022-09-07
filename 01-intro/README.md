# Introduction to Machine Learning

## What is Machine Learning

- Machine Learning is extracting information about the data and making predictions.
- The data has two types:
  - Features - they are the characteristics or the attributes of an object. They are the independent variables of the data.
  - Target - it is the actual value of an object that model tries to predict based on the features. It is the dependent variable of the data.
- A machine learning model encapsulates the patterns of the features and makes predictions based on the target values.
- The predictions can't be 100 percent correct but more or less correct.

## Machine Learning vs Rule-Based System

- In a rule-based system, we have the input data with the set of rules to get the output.
- In machine learning, we have the input and output data, and the model defines the rules.
- If possible, it is always good to begin a task with a rule-based system. On the other hand, machine learning is helpful when it comes to complex and tedious tasks.
- To train a machine learning model, we have to encode the data into the numerical format.
- We first train the model by providing features and target variables (known as train data), where the model learns the patterns to map the corresponding target value.
- To make predictions, the trained model only uses the features to find the correct output.

## Supervised Machine Learning

- Supervised Learning is a type of machine learning where we teach the model by providing features and the target variable.
- Two main types of supervised learning are - classification and regression problems.
- Features are also known as feature matrix, denoted by `X`. The target variable is called the target vector, denoted by `y`.
- A matrix is a 2-dimensional array, which is the combination of rows and columns. And, a vector is a 1-dimensional array consisting of numbers.
- In supervised learning, we pass the feature matrix `X` to the function `g` (i.e. the model) during training.
- The goal of the model is to predict values as close as possible to the target variable `y`. Hence, this equation looks something like `g(X) ≈ y`.
- When we make predictions in a classification problem, the model outputs the class probabilities.
- When we make predictions in a regression problem, the model outputs the number.
- Another classification problem is recommendation systems which predict the rating or ranking of something.

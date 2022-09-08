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

## CRISP-DM

- CRISP-DM stands for Cross-Industry Standard Process for Data Mining and it was invented in 1996.
- It is a methodology created to shape data mining projects.
- CRISP-DM consists of six steps:

  1. Business Understanding
  2. Data Understanding
  3. Data Preparation
  4. Modeling
  5. Evaluation
  6. Deployment

- Steps in **Business Understanding**:
  - Identify the business problem and understand how we can solve it.
  - Define the goal that has to be measurable.

- Steps in **Data Understanding**:
  - Analyze available data sources and decide if we need to get more data.
  - Make sure the data is reliable and we track it correctly.
  - Make sure the data is aligned with the problem we are trying to solve.
  - We may have to go back and revise our business goal to make appropriate adjustments if required.

- Steps in **Data Preparation**:
  - Transform data for modeling.
  - It includes data cleaning, building the pipelines, converting the data into a tabular form, etc.

- Steps in **Modelling**:
  - Train the models (this is where the actual machine learning happens).
  - Try different models and select the best one.
  - Sometimes we may have to go back to data preparation for adding new features and/or fix data issues.

- Steps in **Evaluation**:
  - In the evaluation step, we measure how well the model solves the business problem.
  - If the model is not able to achieve the required goal then We may have to go back to the business understanding and readjust the goal.
  - We may even have to stop working on the project if the goal is not achievable.

- ***Evaluation*** + ***Development***:
  - Before deployment, it often happens that we make model evaluations and development together to get quick feedback.
  - Online evaluation means the model gets deployed and evaluated from the live users feedback.
  - Usually in this process we don't evaluate the model on all users feedback rather it's been done on a small percentage of users.

- Steps in **Deployment**
  - In this step, we deploy the model to production to all users.
  - We monitor the performance of the model.
  - We ensure quality and maintainability.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/CRISP-DM_Process_Diagram.png/319px-CRISP-DM_Process_Diagram.png" width=350 height=350/>

*[Figure: Data mining life cycle](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/CRISP-DM_Process_Diagram.png/319px-CRISP-DM_Process_Diagram.png)*

- ML projects require many iterations. The best practice is to start from simple, learn from the feedback, and improve.

## The Model Selection Process

- There are six steps involve in model selection process:

  1. Split the data.
  2. Train the models.
  3. Validate the models.
  4. Select the best model.
  5. Test the best model.
  6. Final check.

  *Note: Steps 2 and 3 are repetitive until we find the best model for our problem*.

- Before using the test data on our final model, we combine the train + validation data then we train the model on this whole dataset.

## Introduction to NumPy

- NumPy stands for Numerical Python. It is a Python library that performs numerical calculations.
- NumPy is written in C language that makes is very fast.
- NumPy is build on linear algebra. It's about metrices and vectors and performing the mathematical calculations on them.
- The key concept in NumPy is the *NumPy array* data type. A NumPy array may have one or more dimension:
  - 1-dimensional arrays represent vectors.
  - 2-dimensional arrays represent matrices.
  - 3-dimensional arrays represent tensors.

*[NumPy tutorial](https://mlbookcamp.com/article/numpy)*


# 3. Machine Learning for Classification

## 3.1 Churn Project

The project aims to identify customers that are likely to churn or stoping to use a service. Each customer has a score associated with the probability of churning. Considering this data, the company would send an email with discounts or other promotions to avoid churning.

The ML strategy applied to approach this problem is binary classification, which for one instance can be expressed as: $g(x_{i}) = y_{i}$

In the formula, $y_{i}$ is the model's prediction and belongs to {0, 1}, being 0 the negative value or no churning, and 1 the positive value or churning. The output corresponds to the likelihood to churning.

In brief, the main idea behind this project is to build a model with historical data from customers and assign a score of the likelihood of churning.

## 3.2 Data Preparation

This part convers the data obtention and procedures of data preparation.

**Commands, functions, and methods** we use for data preparation:

- `!wget` - Linux shell command for downloading data
- `pd.read_csv()` - read csv files
- `df.head()` - take a look of the dataframe
- `df.head().T` - take a look of the transposed dataframe
- `df.columns` - retrieve column names of a dataframe
- `df.columns.str.lower()` - lowercase all the letters
- `df.columns.str.replace(' ', '_')` - replace the space separator
- `df.dtypes` - retreive data types of all series
- `df.index` - retrive indices of a dataframe
- `pd.to_numeric()` - convert a series values to numerical values. The `errors=coerce` argument allows making the transformation despite some encountered errors
- `df.fillna()` - replace NAs with some value
- `(df.x == 'yes').astype(int)` - convert x series of yes-no values to numerical values

## 3.3 Setting Up the Validation Framework

Splitting the dataest with **Scikit-Learn**.

**Classes, functions, and methods**:

- `train_test_split` - Scikit-Learn class for splitting datasets. Linux shell command for downloading data. The `random_state` argument set a random seed for reproducibility purposes.
- `df.reset_index(drop=True)` - reset the indices of a dataframe and delete the previous ones.
- `df.x.values` - extract the values from x series
- `del df['x']` - delete x series from a dataframe

## 3.4 EDA

The EDA for this project consists of:

- Checking missing values
- Looking at the distribution of the target variable (churn)
- Looking at numerical and categorical variables

**Functions and methods**

- `df.isnull().sum()` - returns the number of null values in the dataframe
- `df.x.value_counts()` - returns the number of values for each category in `x` series. The `normalize=True` argument retrieves the percentage of each category. In this project, the mean of churn is equal to the churn rate obtained with the `value_counts()` method.
- `round(x, y)` - round an x number with y decimal places
- `df[x].nunique()` - return the number of unique values in x series

## 3.5 Feature Importance: Churn Rate and Risk Ratio

1. **Churn rate**: Difference between mean of the target variable and mean of categroies for a feature. If this different is greater than `0`, it means that the category is less likely to churn, and if the difference is lower than `0`, the group is more likely to churn. The larger differences are indicators that a variable is more important than others.

2. **Risk ratio**: Ratio between mean of categories for a feature and mean of the target variable. If this ratio is greater than `1`, the category is more likely to churn, and if the ratio is lower than `1`, the category is less likely to churn. It expresses the feature importance in relative terms.

**Functions and methods**:

- `df.groupby('x').y.agg(['mean', 'count'])` - returns a dataframe with mean and count of `y` series grouped by `x` series
- `display(x)` - displays an output in the cell of a jupyter notebook

## 3.6 Feature Importance: Mutual Information

Mutual information is a concept from information theory, which measures how much we can learn about one varaible if we know the value of another. In this project, we can think of this as how much do we learn about churn if we have the information from a particular feature. So, it is a measure of the importance of a categorical variable.

**Classes, functions, and methods**:

- `mutual_info_score(x,y)` - Scikit-Learn class for calculating the mutual information between the x target variable and y feature
- `df[x].apply(y)` - apply a y function to the x series of the dataframe
- `df.sort_values(ascending=False).to_frame(name='x')` - sort values in decending order and called the column as x.

## 3.7 Feature Importance: Correlation

**Correlation coefficient** measures the degree of dependency between two variables. This value is negative if one variable grows while the other decreases, and it is positive if both variables increase. Depending on its size, the dependency between both variables could be low, moderate, or strong. It allows measuring the importance of numerical variables. If the correlation between two variables are $0.0\ to \pm0.2$ their relationship is low, if it is $\pm0.2\ to \pm0.5$ the relationship is moderate, and if the correlation is between $\pm0.6\ to \pm1.0$ then it is a strong relationship.

**Functions and methods**:

- `df[x].corrwith(y)` - return the correlation between x and y series

## 3.8 One-Hot Encoding

One-Hot Encoding allows encoding categorical variables in numerical ones. This method represents each category of a variable as one column, and a 1 is assigned if the value belongs to the category or 0 otherwise.

**Classes, functions, and methods**:

- `df[x].to_dict(oriented='records')` - convert x series to dictionaries, oriented by rows
- `DictVectorizer().fit_transform(x)` - Scikit-Learn class for converting x dictionaries into a spare matrix, and in this way doing the one-hot encoding. It does not affect the numerical variables. Passing `sparse=False` in the vectorizer will return the one-hot encoded matrix.
- `DictVectorizer().get_feature_names()` - returns the names of the columns in the sparse matrix

## 3.9 Logistic Regression

In general, supervised models can be represented with this formula: $g(x_{i}) = y_{i}$

Depending on what is the type of target variable, the supervised task can be regression or classification (binary or multiclass). Binary classification tasks can have negative (0) or positive (1) target values. The output of these models is the probability of $x_i$ belonging to the positive class.

Logistic regression is similar to linear regression because both models take into account the bias term and the weighted sum of features. The difference between these models is that the outputs a value between zero and one, applying the sigmoid function to the linear regression formula.

$$g(x_i) = Sigmoid(w_0 + w_1x_1 + w_2x_2 + ... +w_nx_x)$$
$$Sigmoid=\frac{1}{1 + exp^{(-z)}}$$

In this way, the sigmoid function allows transforming a score into a probability.

## 3.10 Training Logistic Regression with Scikit-Learn

This section is about training a logistic regression model with Scikit-Learn, applying it to the validation dataset, and calculating its accuracy.

**Classes, functions, and methods**:

- `LogisticRegression().fit_transform(x)` - Scikit-Learn class for calculating the logistic regression model
- `LogisticRegression().coef_[0]` - return the coefficients or weights of the LR model
- `LogisticRegression().intercept_[0]` - return the bias or intercept of the LR model
- `LogisticRegression().predict[x]` - make predictions on the x dataset (0 *or* 1) - hard predictions
- `LogisticRegression().predict_proba[x]` - make predictions on the x dataset and return two columns with their probabilities for the two categories (0 *and* 1) - soft predictions

## 3.11 Model Interpretation

Model interpretation is about intercept and coefficients. In the formula of the logistic regression model, only one of the one-hot ecoded categories is multiplied by 1 (means has weight), and the other by 0 (means no weight). In this way, we only consider the appropriate category for each categorical feature.

**Classes, functions, and methods**:

- `list(zip(x, y))` - return a new list of tuples with elements from x jointed with their corresponding elements on y
- `dict(zip(x, y))` - return a dict with x elements as key and the corresponding elements of y as value

## 3.12 Using the Model

For final model, we train the logistic regression model with the full training dataset (training + validation), considering numerical and categorical features. Thus, predictions are made on the test dataset, and we evaluate the model using the accuracy metric.

In the case, when the predictions of validation and test are similar, it means that the model is working well.

## 3.13 Summary

This project is about predicting churn rate of customers from a company. We learned the feature importance of categorical variables, including risk ratio, mutual information, and correlation coefficient of numerical variables. Also, we understood one-hot encoding and implemented logistic regression with Scikit-Learn.

## 3.14 Explore More

More things

- Try to exclude least useful features

Use scikit-learn in project of last week

- Re-implement train/val/test split using scikit-learn in the project from the last week
- Also, instead of our own linear regression, use `LinearRegression` (not regularized) and `RidgeRegression` (regularized). Find the best regularization parameter for Ridge
- There are other ways to implement one-hot encoding. E.g. using the `OneHotEncoding` class. Check how to use it [here](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/03-classification/notebook-scaling-ohe.ipynb).
- Sometimes numerical features require scaling, especially for iterative solver like "lbfgs". Check how to use StandardScaler for that [here](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/03-classification/notebook-scaling-ohe.ipynb).

Other projects

- Lead scoring - https://www.kaggle.com/ashydv/leads-dataset
- Default prediction - https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
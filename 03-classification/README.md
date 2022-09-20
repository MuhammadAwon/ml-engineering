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


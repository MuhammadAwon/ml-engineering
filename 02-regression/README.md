# 2. Machine Learning for Regression

## 2.1 Car Price Prediction Project

This project is about the creation of a model for helping users to predict car prices. The dataset was obtained from [this kaggle competition](https://www.kaggle.com/CooperUnion/cardataset). Alternatively, we can download the data from [mlbookcamp repository](https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv).

## 2.2 Data Preparation

After loading the data, the next step is the data cleaning which includes, checking the inconsistent representation of data in the columns or rows.

**Pandas attributes and methods**:

- `pd.read_csv()` - read csv files
- `df.head()` - take a look of the dataframe
- `df.columns` - retrieve column names of a dataframe
- `df.columns.str.lower()` - lowercase the letters of the column names
- `df.columns.str.replace(' ', '_')` - replace the space separator and replace with the underscore in the column names
- `df.dtypes` - retrieve data types of all features
- `df.index` - retrieve indices of a dataframe

## 2.3 Exploratory Data Analysis

In this step we observe values of the data, handle missing values, categorize the values (categorical, continuous, discrete), find the data distribution (skewness, kurtosis), identify the relationships (correlations, causation), find the outliers in the data.

**Pandas attributes and methods**:

- `df[col].unique()` - returns a list of unique values in the series
- `df[col].nunique()` - returns the number of unique values in the series
- `df.isnull().sum()` - returns the number of null values in the dataframe

**Matplotlib and Seaborn methods**:

- `%matplotlib inline` - assure that plots are displayed in jupyter ntoebook's cells
- `sns.histplot()` - show the histogram of a series

**NumPy methods**:

- `np.log1p` - applies log transformation to a variable and adds one to each result

Long-tail distributions usually confuse the ML models, so the recommendation is to transfrom the target variable distribution to a normal one whenever possible.

## 2.4 Setting Up the Validation Framework

In general, the dataset is split into three parts: training, validation, and test. For each partition, we need to obtain feature matrices X and y vectors of targets. First, the size of partitions is calculated, records are shuffled to guarantee that values of the three partitions contain non-sequential records of the dataset, and the partitions are created with the shuffled indices.

**Pandas attributes and methods**:

- `df.iloc[]` - returns subsets of records of a dataframe, being selected by numerical indices
- `df.reset_index()` - restate the orginal indices
- `del df[col]` - eliminates target variable

**Numpy methods**:

- `np.arange` - returns an array of numbers
- `np.random.shuffle()` - returns a shuffled array
- `np.randon.seed()` - set a seed for reproducibility

## 2.5 Linear Regression Simple

Model for solving regression tasks, in which the objective is to adjust a line for the data and make predictions on new values. The input of this model is the feature matrix `X` and a `y` vector of predictions is obtained, trying to be as close as possible to the actual y values. The linear regression formula is the sum of the bias term \($w_0$\), which refers to the predictions if there is no infromation, and each of the feature values times their corresponding weights \($x_{i1}.w_1 + x_{i2}.w_2 + ... + x_{in}.w_n$\).

So the simple linear regression formula looks like this: $g(x_i) = w_0 + x_{i1}.w_1 + x_{i2}.w_2 + ... + x_{in}.w_n$

Which can be simplify further:
$g(x_i) = w_0 + \sum_{j=1}^{n}w_j.w_{ij}$

We need to assure that the result is shown on the untransformed scale by using the inverse function `exp()`.

## 2.6 Linear Regression Vector

The formula of linear regression can be synthesized with the dot projuct between features and weights. The feature vector includes the *bias* term with an *x* value of one, such as $w_{0}^{x_{i0}},\ where\ x_{i0} = 1\ for\ w_0$.

When all the records are included, the linear regression can be calculated with the dot product between ***feature matrix*** and ***vector of weights***, obtaining the `y` vector of predictions.

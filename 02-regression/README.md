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

Model for solving regression tasks, in which the objective is to adjust a line for the data and make predictions on new values. The input of this model is the **feature matrix** `X` and a `y` **vector of predictions** is obtained, trying to be as close as possible to the **actual** `y` values. The linear regression formula is the sum of the bias term \( $w_0$ \), which refers to the predictions if there is no information, and each of the feature values times their corresponding weights as \( $x_{i1} \cdot w_1 + x_{i2} \cdot w_2 + ... + x_{in} \cdot w_n$ \).

So the simple linear regression formula looks like: 
$g(x_i) = w_0 + x_{i1} \cdot w_1 + x_{i2} \cdot w_2 + ... + x_{in} \cdot w_n$.

And that can be further simplify: $g(x_i) = w_0 + \displaystyle\sum_{j=1}^{n} w_j \cdot x_{ij}$

If we look at the $\displaystyle\sum_{j=1}^{n} w_j \cdot x_{ij}$ part in the above equation, we know that this is nothing else but a vector-vector multiplication. Hence, we can rewrite the equation as $g(x_i) = w_0 + x_i^T \cdot w$

We need to assure that the result is shown on the untransformed scale by using the inverse function `exp()`. 

## 2.6 Linear Regression Vector

The formula of linear regression can be synthesized with the dot projuct between features and weights. The feature vector includes the *bias* term with an *x* value of one, such as $w_{0}^{x_{i0}},\ where\ x_{i0} = 1\ for\ w_0$.

When all the records are included, the linear regression can be calculated with the dot product between ***feature matrix*** and ***vector of weights***, obtaining the `y` vector of predictions.

## 2.7 Training Linear Regression: Normal Equation

Obtaining predictions as close as possible to `y` target values requires the calculation of weights from the general linear regression equation. The feature matrix `X` does not have an inverse because usually it is not square, so it is required to obtain an approximate solution, which can be obtained using the **Gram matrix** (multiplication of feature matrix and its transpose, $X^TX$). The vector of weights or coefficients obtained with this formula is the closest possible solution to the linear regression system.

## 2.8 Baseline Model for Car Price Prediction Project

The linear regression model obtained prevously was used with the dataset of car price prediction. For this model, only the numerical variables were considered. The training data was pre-processed, replacing the `NaN` values with `0`, in such a way that these values were omitted by the model. Then, the model was trained and it allowed to make predictions were compared by plotting their histograms.

## 2.9 Root Mean Squared Error

The RMSE is a measure of the error associated with a model for regression tasks. It is used the interpret the results and to find the accuracy of the model.

## 2.10 Using RMSE on Validation Data

Calculation of the RMSE on validation partition of the dataset of car price prediction. In this way, we have a metric of evaluate the model's performance.

## 2.11 Feature Engineering

Feature engineering is the process of creating new features.

## 2.12 Categorical Variables

Categorical variables are typically strings, and Pandas identify them as `object` types. These variables need to be converted to a numerical form because the machine learning models can interpret only numerical features. It is possible to incorporate certain categories from a feature, not necessarily all of them. This transformation from categorical to numerical variable is known as One-Hot encoding.

## 2.13 Regularization

If the feature matrix has duplicated columns, it does not have an inverse matrix. But, sometimes this error could be passed if certain values are slightly different between duplicated columns.

So, if we apply the normal equation with this feature matrix, the values associated with duplicated columns are very large, which decrease the model performance. To solve this issue, one alternative is adding a small number to the diagonal of the feature matrix, which corresponds to regularization.

This technique works because the addition of small values to the diagonal makes it less likely to have duplicated columns. The regularization value is a parameter of the model. After applying regularization the model performance improved.

## 2.14 
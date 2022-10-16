# 6. Decision Trees and Ensemble Learning

## 6.1 Session Overview: Credit Risk Scoring Project

In this session we'll learn about decision trees and ensemble learning algorithms. How we can implement and fine-tune these models to make binary classification predictions.

To be specific, we'll use [credit scoring data](https://github.com/gastonstat/CreditScoring) to make model predictions whether the bank should lend the loan to the client or not. The bank makes these decisions based on the historical record.

In the credit scoring classification problem, if the model returns 0 that means the status is `ok` and the client will payback and if the probability is 1 then its the `default` client.

## 6.2 Data Cleaning and Preparation

In this section we clean and prepare the dataset for the model which involves the following steps:

- Download the data from the given link.
- Reformat categorical columns (`status`, `home`, `marital`, `records`, and `job`) by mapping with appropriate values.
- Replace the maximum value of `income`, `assests`, and `debt` columns with NaNs.
- Extract only those rows in the column `status` who are either ok or default as value.
- Split the data with the distribution of 80% train, 20% validation, and 20% test sets with random seed to `11`.
- Prepare target variable `status` by converting it from categorical to binary, where 0 represents `ok` and 1 represents `default`.
- Finally delete the target variable from the train/val/test dataframe.

## 6.3 Decision Trees

Decision Trees are powerful algorithms, capable of fitting complex datasets. The decision trees make predictions based on the bunch of *if/else* statements by splitting a node into two or more sub-nodes.

With versatility, the decision tree is also prone to overfitting. One of the reason why this algorithm often overfits because of its depth. It tends to memorize all the patterns in the train data but struggle to performs well on the unseen data (validation or test set).

To overcome with overfitting problem, we can reduce the complexity of the algorithm by reducing the depth size.

The decision tree with only a single depth is called decision stump and it only has one split from the root.

**Classes, functions, and methods**:

- `DecisionTreeClassifier`: classification model from `sklearn.tree` class.
- `max_depth`: hyperparameter to control the depth of decision tree algorithm.
- `export_text`: method from `sklearn.tree` class to display the text report showing the rules of a decision tree.

*Note*: we have already covered `DictVectorizer` in session 3 and `roc_auc_score` in session 4 respectively.

## 6.4 Decision Tree Learning Algorithm

Before we dive how the decision tree learning algorithm works, let's define some key terms of a decision tree:

- **Root node**: The base of the decision tree, where the first split occurs.
- **Splitting**: The process of dividing a node into multiple sub-nodes.
- **Decision node**: When a sub-node is further split into additional sub-nodes.
- **Leaf node**: When a sub-node does not further split into additional sub-nodes; represents possible outcomes.
- **Pruning**: The process of removing sub-nodes of a decision tree.
- **Branch**: A subsection of the decision tree consisting of multiple nodes.

The decision tree finds the **best split** by recursively going over all the features `F` available in the data to make predictions. Then for each feature, it finds all the possible thresholds `Ts` and iterate over them, where each time the data is split using the condition `F > T` and compute the impurity of that split. At the end, the algorithm selects the condition with the lowest impurity (i.e., average misclassification rate).

As we have seen in the previous lesson that if we leave the decision tree for indefinite depth it can cause overfitting and to overcome this we need to have some kind of **stopping criteria**. The stopping criteria is:

1. Group is already pure.
2. Tree reaches the maximum depth limit.
3. Group too small to split.

In summary, the decision tree first finds the best split, then it checks if the max depth stopping criteria is reached. If not, then it checks whether the data on the left is sufficiently large and not pure yet, in that case it keeps finding the best split. Similary, the algorithm also checks the data on the right side of the tree for sufficiently large size and not pure, it keeps performing the best split.

Different types of classification criteria are:

- Gini
- Log Loss or Entropy
- Misclassification

*Note*: Gini and Entropy classfication criteria are commonly used in decision trees.

**Classes, functions, and methods**:

- `df.sort_values()`: sort the values either by column or row.
- `Series.value_counts(normalize=True)`: return the ratio of the unique values of the series.

## 6.5 Decision Trees Parameter Tuning

There are many tunable parameters available in decision tree but the most important ones are `max_depth` and `min_samples_leaf`. First, we want to find the best value for max_depth where we have the highest AUC score and then using this information we try to find the optimal value for `min_samples_leaf`.

*Note*: Typically, if the best AUC score ties at different values of `max_depth`, in that case, we want to choose the smaller value because that is where we have the simpler model.

**Classes, functions, and methods**:

- `df_pivot()`: reshape the dataframe by given index/column values.
- `sns.heatmap()`: plot rectangular data as a color-encoded matrix.

## 6.6 Ensembles and Random Forest

Random Forest is an example of ensemble learning where each model is a decision tree and their predictions are aggregated to identify the most popular result. Random forest only select a random subset of features from the original data to make predictions.

In random forest the decision trees are trained independent to each other.

**Classes, functions, and methods**:

- `from sklearn.ensemble import RandomForestClassifier`: random forest classifier from sklearn ensemble class.
- `plt.plot(x, y)`: draw line plot for the values of y against x values.

## 6.7 Gradient Boosting and XGBoost

Unlike Random Forest where each decision tree trains independently, in the Gradient Boosting Trees, the models are combined sequentially where each model takes the prediction errors made my the previous model and then tries to improve the prediction. This process continues to `n` number of iterations and in the end all the predictions get combined to make final prediction.

XGBoost is one of the libraries which implements the gradient boosting technique. To make use of the library, we need to install with `pip install xgboost`. To train and evaluate the model, we need to wrap our train and validation data into a special data structure from XGBoost which is called `DMatrix`. This data structure is optimized to train xgboost models faster.

**Classes, functions, and methods**:

- `xgb.train()`: method to train xgboost model.
- `xgb_params`: key-value pairs of hyperparameters to train xgboost model.
- `watchlist`: list to store training and validation accuracy to evaluate the performance of the model after each training iteration. The list takes tuple of train and validation set from DMatrix wrapper, for example, `watchlist = [(dtrain, 'train'), (dval, 'val')]`.
- `%%capture output`: IPython magic command which captures the standard output and standard error of a cell.

## 6.8 XGBoost Parameter Tuning

XGBoost has various tunable parameters but the three most important ones are:

- `eta` (default=0.3)
  - It is also called `learning_rate` and is used to prevent overfitting by regularizing the weights of new features in each boosting step. range: [0, 1]
- `max_depth` (default=6)
  - Maximum depth of a tree. Increasing this value will make the model mroe complex and more likely to overfit. range: [0, inf]
- `min_child_weight` (default=1)
  - Minimum number of samples in leaf node. range: [0, inf]

For XGBoost models, there are other ways of finding the best parameters as well but the one we implement in the notebook follows the sequence of:

- First find the best value for `eta`
- Second, find the best value for `max_depth`
- Third, find the best value for `min_child_weight`

Other useful parameter are:

- `subsample` (default=1)
  - Subsample ratio of the training instances. Setting it to 0.5 means that model would randomly sample half of the trianing data prior to growing trees. range: (0, 1]
- `colsample_bytree` (default=1)
  - This is similar to random forest, where each tree is made with the subset of randomly choosen features.
- `lambda` (default=1)
  - Also called `reg_lambda`. L2 regularization term on weights. Increasing this value will make model more conservative.
- `alpha` (default=0)
  - Also called `reg_alpha`. L1 regularization term on weights. Increasing this value will make model more conservative.

## 6.9 Selecting the Final Model

We select the final model from decision tree, random forest, or xgboost based on the best auc scores. After that we prepare the `df_full_train` and `df_test` to train and evaluate the final model. If there is not much difference between model auc scores on the train as well as test data then the model has generalized the patterns well enough.

Generally, XGBoost models perform better on tabular data than other machine learning models but the downside is that these model are easy to overfit cause of the high number of hyperparameter. Therefore, XGBoost models require a lot more attention for parameters tuning to optimize them.

## 6.10 Summary

- Decision trees learn if-then-else rules from data.
- Finding the best split: select the least impure split. This algorithm can overfit, that's why we control it by limiting the max depth and the size of the group.
- Random forest is a way of combininig multiple decision trees. It should have a diverse set of models to make good predictions.
- Gradient boosting trains model sequentially: each model tries to fix errors of the previous model. XGBoost is an implementation of gradient boosting.

## Explore More

- For this dataset we didn't do EDA or feature engineering. You can do it to get more insights into the problem.
- For random forest, there are more parameters that we can tune. Check `max_features` and `bootstrap`.
- There's a variation of random forest caled "extremely randomized trees", or "extra trees". Instead of selecting the best split among all possible thresholds, it selects a few thresholds randomly and picks the best one among them. Because of that extra trees never overfit. In Scikit-Learn, they are implemented in `ExtraTreesClassifier`. Try it for this project.
- XGBoost can deal with NAs - we don't have to do `fillna` for it. Check if not filling NA's help improve performance.
- Experiment with other XGBoost parameters: `subsample` and `colsample_bytree`.
- When selecting the best split, decision trees find the most useful features. This information can be used for understanding which features are more important than otheres. See example here for [random forest](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html) (it's the same for plain decision trees) and for [xgboost](https://stackoverflow.com/questions/37627923/how-to-get-feature-importance-in-xgboost)
- Trees can also be used for solving the regression problems: check `DecisionTreeRegressor`, `RandomForestRegressor` and the `objective=reg:squarederror` parameter for XGBoost.

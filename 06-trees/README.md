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


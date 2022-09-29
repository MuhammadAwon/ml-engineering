# 4. Evaluation Metrics for Classification

## 4.1 Evaluation Metrics: Session Overview

This session is about differenct metrics to evaluate a binary classifier. These measures include accuracy, confusion table, precision, recall, ROC curves (TPR, FRP, random model, and ideal model), AUROC, and cross-validation.

## 4.2 Accuracy and Dummy Model

**Accuracy** measures the fraction of correct predictions. Specifically, it is the number of correct predictions divided by the total number of predictions.

We can change the **decision threshold**, it should not be always 0.5. But, in this particular problem, the best decision cutoff, associated with the highest accuracy (80%), was indeed 0.5.

Note that if we build a **dummy model** in which the decision cutoff (threshold) is 1, the algorithm predicts that no clients will churn, the accuracy would be 73%. Thus, we can see that the improvement of the original model with respect to the dummy model is not as high as we would expect.

Therefore, in this problem accuracy can not tell us how good is the model because the dataset is **unbalanced**, which means that there are more instances from one category (not churn) than the other (churn). This is also known as **class imbalance** and it is the common issue with the accuracy that it can be quite misleading on data with imbalance classes.

**Classes and methods**:

- `np.linspace(x, y, z)` - returns a numpy array starting from x until y with a z step
- `Counter(x)` - collection class that counts the number of instances that satisfy the x condition
- `accuracy_score(x, y)` - sklean.metrics class for calculating the accuracy of a model, given a predicted x dataset and a target y dataset.

## 4.3 Confusion Table

Confusion table is a way to measure different types of errors and correct decisions that binary classifiers can made. Considering this information, it is possible to evaluate the quality of the model by different strategies.

If we predict the probability of churning from a customer, we have the following four scenarios:

- No churn - **Negative class**
  - Customer did not churn - **True Negative (TN)**
  - Customer churned - **False Negative (FN)**
- Churn - **Positive class**
  - Customer churned - **True Positive (TP)**
  - Customer did not churn - **False Positive (FP)**

The confusion table help us to summarize the measures explained above in a tabular format, as is shown below:

|**Actual/Predictions**|**Negative**|**Postive**|
|:-:|---|---|
|**Negative**|TN|FP|
|**Postive**|FN|TP|

The **accuracy** corresponds to the sum of TN and TP divided by the number of total of observations.

## 4.4 Precision and Recall

**Precision** tells us the fraction of positive predictions that are correct. It takes into account only the **positive class** (FP and TP - second column of the confusion matrix), as is stated in the following formula:
j
$$\frac{TP}{FP + TP}$$

**Recall** measures the fraction of correctly identified positive instances. It considers parts of the **positive and negative classes** (FN and TP - second row of confusion table). The formula of this metric is presented below:

$$\frac{TP}{FN + TP}$$

In this problem, the precision and recall values were 67% and 54% respectively. So, these measures reflect some errors of our model that accuracy did not notice due to the **class imbalance**.

## 4.5 ROC Curves

ROC stands for Receiver Operating Characteristic, and this idea was applied during the Second World War for evaluating the strenght of radio detectors for planes. This measure considers **False Positive Rate** (FPR) and **True Positive Rate** (TPR), which are derived from the values of the confusion matrix.

**FPR** is the fraction of false positives (FP) divided by the total number of negatives (TN and FP - the first row of confusion matrix), and we want to minimize it. The formula of FPR is the following:

$$\frac{FP}{TN + FP}$$

On the other hand, **TPR** or **Recall** is the fraction of true positives (TP) divided by the total number of positives (FN and TP - second row of confusion matrix), and we want to maximize this metric. The formula of this measure is presented below:

$$\frac{TP}{FN + TP}$$

ROC curves consider TPR and FPR under all the possible thresholds. If the threshold is 0 or 1, the TPR and FPR scores are the opposite of the threshold (1 and 0 respectively), but they have different meanings, as we explained before.

We need to compare the ROC curves against a point of reference to evaluate its performance, so the corresponding curves of random and ideal models are required. It is possible to plot the ROC curves with FPR and TPR scores vs thresholds, or FPR vs TPR.

**Classes and methods**:

- `np.repeat([x, y], [i, j])` - returns a numpy array with an i number of x values, and a j number of y values
- `roc_curve(x, y)` - sklearn.metrics class for calculating the false positive rates, true positive rates, and thresholds, given a target x dataset and a predicted y dataset

## 4.6 ROC AUC

The Area under the ROC curves can tell us how good is our model with a single value. The AUROC of a random model is 0.5, while for an ideal one is 1.

In the words, AUC can be interpreted as the probability that a randomly selected positive example has a greater score than a randomly selected negative example.

**Classes and methods**:

- `auc(x, y)` - sklearn.metrics class for calculating area under the curve of the x and y. Where x is the false positive rate and y is the true positive rate
- `roc_auc_score(x, y)` - sklearn.metrics class for calculating area under the ROC curves of the x and y. Where x is the target variable and y is the model predictions.

## 4.7 Cross-Validation

**Cross-validation** refers to evaluating the same model on different subsets of a dataset, gettting the average prediction, and spread (standard deviation) within predictions. This method is applied in the **parameter tuning** step, which is the process of selecting the best parameter.

In this algorithm, the full training dataset is divided into **k partitions**, we train the model in k-1 partitions of this dataset and evaluate it on the remaining subset. Then, we end up evaluating the model in all the k folds, and we calculate the average evaluation metric for all the folds.

In general, if the dataset is large, we should use the hold-out validation dataset strategy. On the other hand, if the dataset is small or we want to know the standard deviation of the model (how stable is the model?) across different folds, we can use the cross-validation approach.

**Libraries, classes and methods**:

- `Kfold(k, s, x)` - sklearn.model_selection class for calculating the cross validation with k folds, s boolean attribute for shuffle decisions, and an x random state
- `Kfold.split(x)` - sklearn.Kfold method for splitting the x dataset with the attributes established in the Kfold's object construction
- `for i in tqdm()` - library for showing the progress of each i iteration in a for loop

**Extra resource**: `Kfold()` class returns the iterator, whereas `Kfold.split()` method generate indices to split data into training and test set. The difference between iterator and generatorcan be read [here](https://www.google.com/search?q=python+iterators+and+generators).

## 4.8 Summary

General definitions:

- **Metric**: A single number that describes the performance of a model
- **Accuracy**: Fraction of correct answers; sometimes misleading
- **Precision** and **Recall** are less misleading when we have class imbalance
- **ROC Curve**: A way to evaluate the performance at all thresholds; it is okay to use with imbalance class
- **K-Fold CV**: More reliable estimate for performance (mean + std)

In brief, this session is about different metrics to evaluate a binary classifier. These measures include accuracy, confusion table, precision, recall, ROC curves (TPR, FPR, random model, and ideal model), and AUROC. Also, in this session, we talked about a different way to estimate the performance of the model and make the parameter tuning with cross-validation.

## 4.9 Explore More

- Check the precision and recall of the dummy classifier that always predict "FALSE"
- F1 = $\frac{2*Precision*Recall}{Precision+Recall}$
- Evaluate precision and recall at different thresholds, plot P vs R - this way we'll get the precision/recall curve (similar to ROC curve)
- Area under the PR curve is also a useful metric

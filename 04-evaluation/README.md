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

**Precision** tells us the fraction of positive predictions that are correct. It takes into account only the **positive class** (TP and FP - second column of the confusion matrix), as is stated in the following formula:
j
$$\frac{TP}{TP + FP}$$

**Recall** measures the fraction of correctly identified positive instances. It considers parts of the **positive and negative classes** (TP and FN - second row of confusion table). The formula of this metric is presented below:

$$\frac{TP}{TP + FN}$$

In this problem, the precision and recall values were 67% and 54% respectively. So, these measures reflect some errors of our model that accuracy did not notice due to the **class imbalance**.

## 4.5 ROC Curves


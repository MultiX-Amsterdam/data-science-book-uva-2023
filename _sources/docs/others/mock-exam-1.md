# Mock Exam 1

(Last updated: Feb 21, 2023)

Notice that this mock exam only contains a part of the real exam. In the real exam, there will be more questions.

**Q1:** Which of the following statements about missing data is **FALSE**?
1. Missing Completely At Random (MCAR) means that the missing data is a completely random subset of the entire dataset.
2. Missing data can be treated by replacing them with a constant value.
3. Missing Not At Random (MNAR) cannot be solved simply with imputation.
4. Missing at Random (MAR) means the missing data is related to the variable that has the missing data.

**Q2:** Which of the following statements about overfitting is **FALSE**?
1. Overfitting means fitting the test data extremely well.
2. Overfitting usually happens when using a very complex model.
3. We cannot deal with overfitting by simply removing outliers.
4. Errors of the model that we trained can be decomposed into bias, variance, and noise. Overfitting usually happens when we train a model with high variance.

**Q3:** Which of the following statements about the Decision Tree model is **TRUE**? Notice that we refer to the general Decision Tree, not a specific implementation.
1. Decision Tree can be trained well using entropy as the node-splitting strategy.
2. When splitting nodes, Decision Tree cannot use the same feature multiple times.
3. Decision Tree cannot be used for the regression task.
4. Decision Tree can only handle categorical features.

**Q4:** The research team in a municipality wants to use data science to explain how air pollution affects citizens’ living quality in the city. The municipality requires that the model’s decision-making process is as transparent as possible. This means that a researcher can check the model and explain how the model converts an input to an output intuitively. Which of the following algorithms does **NOT** suit the purpose?
1. Decision Tree
2. Linear Regression
3. Logistic regression
4. Deep Neural Network

**Q5:** Suppose we flipped a coin 12 times, and we got 10 heads and 2 tails. What is the entropy of the result in this coin-flipping experiment? The formula of entropy H is given below:

$$
H = p_{1} * log_{2}(1/p_{1}) + p_{2} * log_{2}(1/p_{2})
$$

where $p_{1}$ is the probability of seeing a head, and $p_{2}$ is the probability of seeing a tail.
1. 0.17
2. 0.20
3. 0.65
4. 0.83

**Q6:** Suppose we have an array A of data with 3 columns, and we want to put the data in a pandas data frame. We want to give the first column with name "C1", the second column with name "C2", and the third column with name "C3". Which of the following code produces the desired output for array A (considering that we already imported the pandas package)? For example, the output should look like the following data frame:
|  | C1 | C2 | C3 |
|---|---|---|---|
| 0 | 5 | 21.6 | 100 |
| 1 | 11 | 48.3 | 213 |
| 2 | 1 | 44.2 | 433 |
1. `pandas.DataFrame(A, rows=["C1", "C2", "C3"])`
2. `pandas.DataFrame(A, columns=["C1", "C2", "C3"])`
3. `pandas.DataFrame(A).set_index(["C1", "C2", "C3"])`
4. `pandas.DataFrame(A, columns=["C1", "C2", "C3"]).set_index(["C1"])`

**Q7:** Suppose we have a pandas data frame D with 100 rows and one column C1. Column C1 has 25% missing data. We want to sum up all valid items in column C1. Which of the following code produces the desired output? For example, if D looks like the data frame below, the code should output 27, which is a sum of 3, 4, and 20.
|  | C1 |
|---|---|
| 0 | NaN |
| 1 | 3 |
| 2 | 4 |
| 3 | 20 |
1. `D.drop("C1", axis=1).sum()`
2. `D.dropna().sum()["C1"]`
3. `D.sum("C1")`
4. `D.groupby("C1").sum()`

## Answers

The block below contains the answers to the questions.

```{toggle}
- Q1 -> 4
- Q2 -> 1
- Q3 -> 1
- Q4 -> 4
- Q5 -> 3
- Q6 -> 2
- Q7 -> 2
```

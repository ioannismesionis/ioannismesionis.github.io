---
layout: post
title: An overview of variable correlations!
subtitle: Hey, variable - are we (co)related?
katex: true
image: /img/correlations/small-correlation-and-causation.png
bigimg: /img/correlations/Correlation_examples_big.png
tags: [machine-learning, mathematics, correlation]
---

Correlations are like two peas in a pod - they just can't be separated. In machine learning, correlations are like the secret ingredient that makes our models stand out. They help us identify patterns and relationships between features, and guide us in selecting the most relevant variables to predict our target variable. Correlations are the spice that brings flavor to our machine learning recipes, and the glue that binds our models together. So, let's embrace correlations, and make some deliciously accurate models!

# Correlations

1. **Pearson Correlation Coefficient:** <br>
This is the most commonly used method for measuring the linear correlation between two variables. It computes the strength and direction of the linear relationship between two continuous variables.

```python
# Compute Pearson correlation coefficient between two columns
df['column1'].corr(df['column2'])
```


2. **Spearman Rank Correlation Coefficient:** <br>
This is a non-parametric method for measuring the correlation between two variables. It computes the strength and direction of the monotonic relationship between two continuous or ordinal variables.

```python
# Compute Spearman rank correlation coefficient between two columns
df['column1'].corr(df['column2'], method='spearman')
```


3. **Kendall's Tau Correlation Coefficient:** <br>
This is another non-parametric method for measuring the correlation between two variables. It computes the strength and direction of the monotonic relationship between two continuous or ordinal variables.

```python
# Compute Kendall's tau correlation coefficient between two columns
df['column1'].corr(df['column2'], method='kendall')
```


4. **Point-Biserial Correlation Coefficient:** <br>
This is a method for measuring the correlation between a continuous variable and a binary variable. It computes the strength and direction of the correlation between a continuous variable and a binary variable (coded as 0 or 1).

```python
# Compute point-biserial correlation coefficient between a column and a binary column
df['column1'].corr(df['binary_column'], method='pearson')
```


5. **Phi Correlation Coefficient:** <br>
This is a method for measuring the correlation between two binary variables. It computes the strength and direction of the correlation between two binary variables (coded as 0 or 1).

```python
# Compute phi correlation coefficient between two binary columns
df['binary_column1'].corr(df['binary_column2'], method='phi')
```

![](/img/correlations/groups.png)



## Summary

![](/img/correlations/correlations_summary.png)



| Correlation name | Formula | Intuition | Assumptions / Limitations |
| --- | --- | --- | --- |
| Pearson Correlation Coefficient

(Continuous variables) | 
$\rho_{X,Y} = \frac{COV(X, Y)}{\sigma_{x}\sigma_{y}} = \frac{\sum_{i=2}^{n}(x_{i} - \bar{x})(y_{i} - \bar{y})}{\sqrt{\sum_{i=2}^{n}(x_{i} - \bar{x})^{2}}\sqrt{\sum_{i=1}^{n}(y_{i} - \bar{y})^{2}}}$ | It is essentially a normalized measurement of the covariance | - Relationship between the two variables is linear. - Variables are normally distributed.
- There are no significant outliers in the data. |
| Spearman Rank Correlation Coefficient

(Continuous and ordinal variables)
 | 
$r_{s} = \rho_{R(X), R(Y)} = \frac{COV(R(X), R(Y))}{\sigma_{X}\sigma_{Y}} \\ \space \space \space \space = 1 - \frac{6\sum_{i=1}^{n}d_{i}}{n(n^{2}-1)}$ 

where $d_{i} = R(X_{i}) - R(Y_{i})$ | It is the Pearson Correlation of the ranked variables. | - The relationship between the variables is monotonic.
(i.e. as one variable increases, the other variable either also increases or decreases)

- Easier to compute compared to Kendall’s Tau correlation.

- May not be appropriate for small sample sizes.

 |
| Kendall's Tau Correlation Coefficient

(Continuous and ordinal variables) | $r = \frac{\text{number of concordant pairs}) - \text{(number of discordant pairs)}}{\text{number of pairs}} = \\ \space\space\space = 1-\frac{2(\text{number of discordant pairs})}{(n \space \text{choices} \space 2)}$

where “concordant” between two variables being:
$\text{sign}(X_{2} - X_{1}) = \text{sign}(Y_{2} - Y_{1}) $ | The intuition behind the formula is that if there are more concordant pairs than discordant pairs, then the variables have a positive monotonic relationship, and the value of tau is close to +1. | - The relationship between the variables is monotonic.
(i.e. as one variable increases, the other variable either also increases or decreases)

- More robust to outliers compared to Spearman’s Correlation.

- Handles tied ranks (i.e. when two or more observations in a dataset have the same value) better than Spearman's Correlation.

- Requires a larger sample size than Spearman's Rank Correlation Coefficient to achieve the same level of power. |
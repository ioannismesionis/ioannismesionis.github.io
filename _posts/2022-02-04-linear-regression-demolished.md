---
layout: post
title: Machine Learning - Demolished!
subtitle: Linear Regression Demolished!
katex: true
image: /img/linear-regression/brain.png
bigimg: /img/linear-regression/yellow-mathematics.jpeg
tags: [machine-learning, mathematics, linear-regression]
---

Having a passion for mathematics, I was delighted to find out that Machine Learning models are using algebra under their hoods. As a result, I decided to create this mini-blog series called "Machine Learning" demolished! The purpose of the blogs is for me to refresh my memory on the optimisation techniques around the Loss function used in the most common Machine Learning models and create some sort of machine learning "mind palace" as Sherlock Holmes says. Hope that you will find the following information as useful as I did.

# Linear Regression - Demolished

---

The goal of *Regression* is to model the relationship between some input data with a continuous-valued target (response variable). Mathematically speaking, regression is a mapping of a D-dimensional vector $x$ with a real continuous target $y$

- **Training data**: $$N$$ training examples of a $D$-dimensional input data.

$$
X = \mathbf{x_1}, \mathbf{x_2}, \dots , \mathbf{x_D}
$$

where, $$\mathbf{x}_{i} $$ =
$$\begin{pmatrix}
x_{1}^{(i)} \\
x_{2}^{(i)} \\
\vdots \\
x_{N}^{(i)}
\end{pmatrix}$$ , $$i=1,2, \dots, D$$.

- **Response**: Continuous-valued target of the corresponding $N$ training examples:

$$
y_{n},\space n = 1, 2, \dots , N
$$

The input data $\mathbf{x}_{i}, \space i=1,2, \dots, D$ are know as *variables* and they can come from different sources:

- quantitative inputs (e.g. Income, Age, etc.)
- basis function, e.g. $\phi_{j}=\mathbf{x}^{j}$ (polynomial regression)
- numeric or "dummy" encodings of qualitative inputs
- interactions between variables, e.g. $\mathbf{x_{1}}=\mathbf{x_{2}} \cdot \mathbf{x_{3}}$

and we assume that these data points are drawn independently from the population distribution.

# Mathematical Formula

---

The *simplest form* of Linear Regression model is linear functions of the input variables:

$$
\hat{y} = f(\mathbf{x}, \mathbf{w}) = w_{0} + w_{1}x_{1}+ \dots + w_{D}x_{D}
$$

with $\mathbf{x}$ in this simple case being $\mathbf{x} = (x_{1}, \dots, x_{D})^{T}$. <br>
(i.e. a single observation per variable)

<aside>
<em>
💡 The formula above has the following limitations: <br>
1. linear function of the parameters:$\space w_{1}, \dots, w_{D}$ <br>
2. linear function of the input variables: $\space x_{1}, \dots ,x_{D}$
</em>
</aside>

In order to remove *limitation number 2*, we introduce the **basis functions** so that the simplest formula is *extended* to:

The *basis form* of Linear Regression model is:

$$
\hat{y} = (\mathbf{x}, \mathbf{w}) =
w_{0} + \sum\limits_{j=1}^{D}w_{j}\phi_{j}(x) =
\sum\limits_{j=0}^{D}w_{j}\phi_{j} = \boldsymbol{w}^{T}
\boldsymbol{\phi(\mathbf{x})}
$$

where $\boldsymbol{\phi_{j}}:$ basis functions and $\boldsymbol{w} = (w_{0}, w_{1}, \dots, w_{D})^{T}$ and  $\boldsymbol{\phi} =(1, \space \phi_{1}, \dots, \phi_{D})^{T}$.

<aside>
<em>
💡 The basis function can be fixed non-linear functions of the input variables $\mathbf{x_{i}}$ so that the basis formula follows the properties: <br>
1. linear function of the parameters, $w_{1}, \dots, w_{D}$
</em>
</aside>

> <span style="color:red"> Assumption 1:</span> <br>
The linear regression formula is a linear function of the parameters $w_{1}, \cdots, w_{D}$
>

In the simplest example where $D=1$, the mapping $f(\mathbf{x}, \mathbf{w})$ can be represented as a single line that change for the different values of $\mathbf{x}$

Increasing the dimension of D can result in a hyperplane, which is inefficient in terms of visualisation.

The ultimate goal of machine learning is to use some input data from which the model can learn in order to predict future data as accurately as possible. The input data is utilised to minimise a *Loss Function* to estimate the values of the *coefficient parameter* $\mathbf{w}$, and then the model may be used for prediction.

Based on the data at hand, the Data Scientist can choose an appropriate Loss Function from a list of functions accessible in the bibliography.

# Loss Functions

---

As a reminder, we made the assumption that our data can be adequately approximated by a linear function. In other words, we hope that $$E(y \mid \mathbf{x}) \approx f(\mathbf{x,w})$$ holds true and is a reasonable approximation.

We can then safely write:

$$
\begin{align*}
y & = f(\mathbf{x, w}) + \boldsymbol{\epsilon}
\\
& = \hat{y} + \boldsymbol{\epsilon}
\end{align*}
$$

where $y \rightarrow$ target variable, <br>
$\hspace{2.5cm} \hat{y} \rightarrow$ estimated value, <br>
$\hspace{2.5cm} f(\mathbf{x, w}) \rightarrow$ deterministic function, <br>
$\hspace{2.5cm} \boldsymbol{\epsilon} \rightarrow$ *residuals (estimation of the error)*

In order to minimise the $\epsilon$ error term, we need to find the $\mathbf{w}$ coefficient weights that will make these $y-\hat{y}$ differences (or their loss functions derivations) as small as possible.

The following section displays the most popular loss functions used in Data Science.

### Mean Squared Error (MSE)

*<span style="color:blue"> Formula:</span>* <br>
$$E_{D}(w) = \frac{1}{N} \sum\limits_{n=1}^{N}(y_{n} - \hat{y})^{2}$$

*<span style="color:blue"> Advantages:</span>*

1. Sum of Squares can be motivated as the Maximum Likelihood Solution under an assumed Gaussian noise model
2. Squared differences have the nice mathematical properties; continuously differentiable which is convenient when trying to minimise it.
3. Sum of Squares is a convex function which mean that the local minimum=global minimum.

*<span style="color:blue"> Disadvantages:</span>*

1. Not robust to outliers as it penalises them to the power of 2
2. Scaled-dependent

### Mean Absolute Error (MAE)

*Formula:* <br>
$$E_{D}(w) = \frac{1}{N} \sum\limits_{n=1}^{N} \mid y_{i} - \hat{y_{i}} \mid$$

*<span style="color:blue"> Advantages:</span>*

1. More robust to outliers compared to MSE

*<span style="color:blue"> Disadvantages:</span>*

1. Not differentiable which needs the application of optimisers such as Gradient Descent to minimise

### Root Mean Squared Error (RMSE)

*<span style="color:blue"> Formula:</span>* <br>
$$E_{D}(w) = \sqrt{MSE}$$

*<span style="color:blue"> Advantages:</span>*

1. Output is at the same unit as the input (interpretation usefulness)

*<span style="color:blue"> Disadvantages:</span>*

1. Not that robust to outliers

### Mean Absolute Percentage Error (MAPE)

*<span style="color:blue"> Formula:</span>* <br> <br>
$$E_{D}(w) = \frac{100\%}{N} \sum\limits_{n=1}^{N} \mid \frac{y_{i} - \hat{y_{i}}}{y_{i}} \mid$$

*<span style="color:blue"> Advantages:</span>*

1. Easy to interpret as a percentage

*<span style="color:blue"> Disadvantages:</span>*

1. The MAPE, as a percentage, only makes sense for values where divisions and ratios make sense. <br>
E.g. not applicable for cases that need to calculate the accuracy of a temperature forecast as it doesn't make sense to calculate percentages of temperatures.
2. Not differentiable everywhere which means that first and second derivatives not always defined

<aside>
<em>
💡 In the context of Machine Learning, selecting a Loss Function to minimise is more than enough considering that the only interest is to "fit" a line into some data. Minimising a Loss Function is a mathematical minimisation problem with no assumptions made for the distribution of the data. In other words, training a linear regression model does not require that the independent or target variables are normally distributed. The normality assumption is only a requirement for certain statistics and hypothesis tests.
</em>
</aside>

To ensure, however, that our $\mathbf{w}$ estimate is unbiased, we need to extend our assumptions about the data.

> <span style="color:red"> Assumption 2:</span> <br>
The residuals are normally distributed with mean = $0$
 *Note: This is called "Normality" of the residuals.*
>

> <span style="color:red"> Assumption 3:</span> <br>
The residuals have constant variance for every input of the data $\mathbf{x_{D}}, n=1, \dots ,D$
*Note: This is known as "homoscedasticity".*
>

> <span style="color:red"> Assumption 4:</span> <br>
The residuals are not correlated with each other, i.e. not auto-correlated.
Auto-correlation takes place when there is a pattern in the rows for the data (e.g. time-series)
>

# Coefficients Estimation

---

Our goal is to estimate the $\boldsymbol{w}$ parameters that minimise the selected **Loss Function**; in our case that can be the Sum of Squares - especially when the $\epsilon$ error is assumed to be $\N(0, \sigma^{2})$ distributed :

$$
E_{D}(w) = RSS = \frac{1}{2} \sum\limits_{n=1}^{N}(\mathbf{y}_{n} - \mathbf{w}^{T}\boldsymbol{\phi} (\mathbf{x}))^{2}\\ \hspace{1.5cm} \hookrightarrow \textcolor{blue}{\text{Residual Sum of Squares}}
$$

In mathematics, to find the minimum of a function, we have to set the derivative of the function to 0. Therefore:

$$
\begin{split}
0 = \bigtriangledown E_{D}(\boldsymbol{w})
& = \frac{1}{2} \cdot 2 \sum\limits_{n=1}^{N}(\boldsymbol{y}_{n} - \boldsymbol{w}^T\boldsymbol{\phi(\mathbf{x})})(\boldsymbol{y}_{n} - \boldsymbol{w}^{T}\boldsymbol{\phi(\mathbf{x})})^{'}
\\ & = \sum\limits_{n=1}^{N}(\boldsymbol{y}_{n} - \boldsymbol{w}^{T}\boldsymbol{\phi(\mathbf{x})})\boldsymbol{\phi(\mathbf{x})^{T}}
\\ & = \sum\limits_{n=1}^{N}(\boldsymbol{y}_{n}\boldsymbol{\phi(\mathbf{x})^{T}} - \boldsymbol{w}^{T}\boldsymbol{\phi(\mathbf{x})^{T}})
\end{split}
$$

Converting the equation above into using Matrix notation, we get:

$$
\begin{split}
0 & = \boldsymbol{\Phi^{T} - \Phi^{T}\Phi w} \Leftrightarrow \\
&\Leftrightarrow \boldsymbol{\Phi^{T}\Phi w} = \boldsymbol{\Phi t}  \Leftrightarrow \\
&\Leftrightarrow \boldsymbol{\hat{w}} = (\boldsymbol{\Phi^{T}\Phi})^{-1}\boldsymbol{\Phi^{T}} \boldsymbol{t} \hspace{0.5cm} \longrightarrow \textcolor{blue}{\textbf{Normal equations}}
\end{split}

$$

where $\boldsymbol{\Phi}$ is called the design matrix

$$
\boldsymbol{\Phi} = \begin{pmatrix}
\phi_{o}(x_{1}) & \phi_{1}(x_{1}) & \cdots & \phi_{D}(x_{1})\\
\phi_{o}(x_{2}) & \phi_{1}(x_{2}) & \cdots & \phi_{D}(x_{2})\\
\vdots & \vdots & \ddots & \vdots \\
\phi_{o}(x_{N}) & \phi_{1}(x_{N}) & \cdots & \phi_{D}(x_{N})\\
\end{pmatrix}
$$

with $\boldsymbol{\phi} = (\phi_{0}, \dots , \phi_{D})^{T}$ and  $\Phi$ being a $N$ x $(D+1)$ matrix.

> <span style="color:red"> Assumption 5:</span> <br>
For the $(\boldsymbol{\Phi}^{T}\boldsymbol{\Phi})^{-1}$ we need to assume that $\boldsymbol{\Phi}$ is of full rank
i.e. *the independent variables are not correlated* (e.g. $\phi_{1} = 3\phi_{3}$).
*Note: This is know as no multicollinearity.*
>

# Model Evaluation

---

## R-Squared

*Description:* <br>
$R^{2}$ measures how much variance can be explained by your model.

$R^{2}$ can also be viewed as how much the regression line is better than the mean line.

*Formula:* <br>
$$R^{2} = 1 - \frac{\text{Unexplained Variance}}{\text{Total Variation}} = \\
\hspace{0.5cm} = 1 - \frac{SS_{reg}}{SS_{mean}} = \\
\hspace{0.5cm} = 1 - \frac{\sum\limits_{i=1}^{N} (y_{i} - \hat{y_{i}})^{2}}{\sum\limits_{i=1}^{N} (y_{i} - \overline{y_{i}})}$$

where $\overline{y_{i}}$ being the mean of target variable.

*Value's Range:* <br>
From 0 (bad model) to 1 (perfect model)

*Comment:* <br>
A problem with the $R^{2}$ metric is that sometimes it increases as we add more variables even if the added variables are irrelevant.

In other words, the model can always map some data to a target variable.

## Adjusted R-Squared

*Description:* <br>
$R^{2}$ - Adjusted overcomes the incorrect increase of the $R^{2}$ by adding extra independent variables.

*Formula:* <br>
$$R_{a}^{2} = 1 - \{ (\frac {n-1}{n-k-1})(1-R^{2})\}$$

where $n=$ number of obserbations <br>
and $k=$ number of features.

*Value's Range:* <br>
From 0 (bad model) to 1 (perfect model)

*Comment:* <br>
As $k$ increases, the denominator decreases which makes the entire value to be subtracted from 1 a large value. As a result, the $R^{2}_{a}$ is decreased which means that the more irrelevant features, the worse the model

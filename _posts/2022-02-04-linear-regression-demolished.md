---
layout: post
title: Machine Learning - Demolished!
subtitle: Linear Regression Demolished!
katex: true
image: /img/linear-regression/brain.png
bigimg: /img/linear-regression/yellow-mathematics.jpeg
tags: [machine-learning, mathematics, linear-regression]
---

Having a passion about mathematics, I was delighted to find out that Machine Learning models are using algebra under their hoods. As a result, I decided to create this mini-blog series called "Machine Learning" demolished! The purpose of the blogs is for me to refresh my memory on the optimisation techniques around the Loss function used in the most common Machine Learning models and create some sort of machine learning "mind palace" as Sherlock Holmes says. Hope that you will find the following information as useful as I did.

### Linear Regression

The goal of Regression is to predict the value of one or more continuous target variables **t** given the value of a D-dimensional vector x of input variables.

* Training Data: X = {$\mathbf{x_1}$, $\mathbf{x_2}, \dots , \mathbf{x_N}$} <br>
$\quad \quad \hookrightarrow$ N training examples.

* Response / Target: {$t_{n}$},  $n = 1, 2, \dots , N$  
$\hspace{2.8cm} \hookrightarrow$ vector $\mathbf{t}$ (of dimension $N$)

The $\textcolor{blue}{\text{D-dimensional vectors}}$ $\textcolor{blue}{\mathbf{x_{i} }}$, $\textcolor{blue}{i=1, \dots, N}$ are know as variables and they can come from different sources:
* quantitative inputs (e.g. Income, Age, etc.)
* basis function, e.g. $\phi_{j}=\mathbf{x}^{j}$ (polynomial regression)
* numeric or "dummy" encodings of qualitative inputs
* interactions between variables, e.g. $\mathbf{x_{1}}=\mathbf{x_{2}} \cdot \mathbf{x_{3}}$

and we assume that these data points are drawn independently from the population distribution.

The simplest form of $\textcolor{blue}{\text{Linear Regression}}$ model is linear functions of the input variables.

$\colorbox{lightgreen}{Simplest Formula:}$ $y(\mathbf{x}, \mathbf{w}) = w_{0} + w_{1}x_{1} + \dots + w_{D}x_{D}$
$\hspace{3.5cm} \hookrightarrow \mathbf{x} = (x_{1}, \dots, x_{D})^{T}$
>⚠️ $\textcolor{blue}{\text{This formual has the following properties:}}$ <br>
> 1. $\textcolor{blue}{\text{linear function of the parameters,} \space w_{1}, \dots, w_{D}}$ 
> 2. $\textcolor{blue}{\text{linear function of the input variables,} \space x_{1}, \dots ,x_{D}}$

In order to remove _limitation number 2_, we introduce the **basis functions** so that the simplest formula is _extended_ to:

$\colorbox{lightgreen}{Basis Formula:}$ $y(\mathbf{x}, \mathbf{w}) = w_{0} +  \sum\limits_{j=1}^{M-1} w_{j}\phi_{j}(x)=$
$\hspace{6.2cm} \hookrightarrow \boldsymbol{\phi_{j}}:$ basis functions
$\hspace{3.81cm} = \sum\limits_{j=0}^{M-1} w_{j}\phi_{j} = \boldsymbol{w}^{T} \boldsymbol{\phi(\mathbf{x})}$ $\rightarrow \boldsymbol{\phi}=(\phi_{0}, \dots, \phi_{M-1})$
$\hspace{6.13cm} \hookrightarrow \boldsymbol{w}=(w_{o}, \dots, w_{M-1})^{T}$


>⚠️ $\textcolor{blue}{\text{The basis function can be fixed non-linear functions of the input variables}}$ $\textcolor{blue}{\mathbf{x_{i}}}$ $\textcolor{blue}{\text{ so that the basis formula follows the properties:}}$ <br>
 >1. $\textcolor{blue}{\text{linear function of the parameters,} \space w_{1}, \dots, w_{D}}$ 

$\textcolor{red}{\underline{\textbf{Assumption 1:}}}$ $\textcolor{red}{\text{The linear regression formula is a linear function of the parameters } w_{1}, \cdots, w_{D} }$

To use a model for prediction, we need to derive the weight parameter $\mathbf{w}$. To do this, we have to define a $\textcolor{blue}{\text{loss function}}$ to minimize.

$$\colorbox{orange}{Typical Loss/Error Functions}$$
1. **Sum of Squares:** $E_{D}(w) = \frac{1}{2} \sum\limits_{n=1}^{N}(t_{n} - \mathbf{w}^{T}\boldsymbol{\phi} (\mathbf{x_{n}}))^{2}$

2. **Absolute Error:** $E_{D}(w) = \frac{1}{2} \sum\limits_{n=1}^{N} \lvert t_{n} - \mathbf{w}^{T}\boldsymbol{\phi} (\mathbf{x_{n}}) \rvert$

>⚠️ $\textcolor{blue}{\text{The most common loss function is the \textbf{Sum of Squares}.}}$
>$\textcolor{blue}{\text{\space\space\space\space\space These are some of the reasons why:}}$ <br>
 >1. $\textcolor{blue}{\textbf{Sum of Squares }\text{can be motivated as the \textbf{Maximum Likelihood Solution} under an assumed \textbf{Gaussian noise model}}. }$ 
 >2. $\textcolor{blue}{\text{Squared differences have the nice mathematical properties; continuously differentiable which is convenient when trying to mimimize it.}}$
 >3. $\textcolor{blue}{\text{Sum of Squares is a \textbf{convex function} which mean that the local minimum=global minimum.}}$

In the context of Machine Learning, selecting a Loss function to minimize is more than enough considering that the only interest is to "fit" a line into some data. In other words, minimizing the Sum of Squares is a mathematical minimization problem with no assumptions made for the distribution of the data. To ensure that our $\mathbf{w}$ estimate is unbiased, we need to extend our assumptions about the data. 

As a reminder, we made the assumption that a linear function can be adequately approximated by a linear function. In other words, we hope that
$$E(t|x) \approx y(\mathbf{x,w})$$ holds true and is a reasonable approximation.
We can then safely write: 
$$ \boldsymbol{t} = y(\mathbf{x, w}) + \boldsymbol{\epsilon} $$ where, 
1. $\mathbf{t} \rightarrow$ target variable
2. $y(\mathbf{x, w}) \rightarrow$ deterministic function
3. $\boldsymbol{\epsilon} \rightarrow$ $\N(0, \sigma^2)$
$\hookrightarrow$ residuals (estimation of the error)

$\textcolor{red}{\underline{\textbf{Assumption 2:}}}$
$\textcolor{red}{\text{The residuals are normally distributed with mean=0}}$
$\textcolor{red}{\text{Note: This is called \underline{"Normality"} of the residuals.}}$ <br>

$\textcolor{red}{\underline{\textbf{Assumption 3:}}}$ 
$\textcolor{red}{\text{The residuals have constant variance for every input of the data} \mathbf{x_{n}}, n=1, \dots ,N}$
$\textcolor{red}{\text{Note: This is known as \underline{"homoscedasticity"}.}}$ <br>

$\textcolor{red}{\underline{\textbf{Assumption 4:}}}$ 
$\textcolor{red}{\text{The residuals are not correlated with each other, i.e. not auto-correlated.}}$
$\textcolor{red}{\text{Auto-correlation takes place when there is a pattern in the rows fo the data (e.g. time-series)}.}$

Our goal is to estimate the $\boldsymbol{w}$ parameters that minimize the selected **Loss Function**; in our case the Sum of Squares:
$$E_{D}(w) = RSS = \frac{1}{2} \sum\limits_{n=1}^{N}(t_{n} - \mathbf{w}^{T}\boldsymbol{\phi} (\mathbf{x_{n}}))^{2}$$ $\hspace{5.5cm} \hookrightarrow \textcolor{blue}{\text{Residual Sum of Squares}}$

In mathematics. to find the minimum of a function, we have to set the derivative of a function to 0. Therefore:

$$
\begin{split}
0 = \bigtriangledown E_{D}(\boldsymbol{w}) & = \frac{1}{2} \cdot 2 \sum\limits_{n=1}^{N}(\boldsymbol{t}_{n} - \boldsymbol{w}^T\boldsymbol{\phi(\mathbf{x_{n}})(\boldsymbol{t}_{n} - \boldsymbol{w}^T\boldsymbol{\phi(\mathbf{x_{n})})^{'}}} \\
  & = \sum\limits_{n=1}^{N}(\boldsymbol{t}_{n} - \boldsymbol{w}^T\boldsymbol{\phi(\mathbf{x_{n}}))\boldsymbol{\phi(\mathbf{x_{n}})^{T}}} \\
  & = \sum\limits_{n=1}^{N}(\boldsymbol{t}_{n}\boldsymbol{\phi(\mathbf{x}_{n})^{T}} - \boldsymbol{w}^{T}\boldsymbol{\phi(\mathbf{x_{n}})^{T}})
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

where $\boldsymbol{\Phi}$ is called the _design matrix_ $\boldsymbol{\Phi} = \begin{pmatrix}  
\phi_{o}(x_{1}) & \phi_{1}(x_{1}) & \cdots & \phi_{d}(x_{1})\\  
\phi_{o}(x_{2}) & \phi_{1}(x_{2}) & \cdots & \phi_{d}(x_{2})\\
\vdots & \vdots & \ddots & \vdots \\  
\phi_{o}(x_{N}) & \phi_{1}(x_{N}) & \cdots & \phi_{d}(x_{N})\\
\end{pmatrix}$

>⚠️ $\textcolor{blue}{\text{Note:}}$
 >1. $\textcolor{blue}{\boldsymbol{\phi} = (\phi_{0}, \dots , \phi_{D})^{T}}$ 
 >2. $\textcolor{blue}{ \textcolor{blue}{\Phi \text{ is a NxD matrix.}}}$

$\textcolor{red}{\underline{\textbf{Assumption 5:}}}$
$\textcolor{red}{\text{For the } (\boldsymbol{\Phi}^{T}\boldsymbol{\Phi})^{-1} \text{ we need to assume that } \boldsymbol{\Phi} \text{ is of full rank, i.e. \underline{the independent variables are not correlated} (e.g. } \phi_{1} = 3\phi_{3}.}$
$\textcolor{red}{\text{Note: This is know as \underline{no multicollinearity.}}}$

$$\colorbox{orange}{Model Evaluation}$$

| Metric                         | Syntax                                                      | Advantages                                                           | Disadvantages                                                                                 |
|--------------------------------|-------------------------------------------------------------|----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| Mean Squared Error (MSE)       | $\frac{1}{N} \sum\limits_{n=1}^{N} (t_{i} - \hat{t_{i}})$   | Differentiable so can be <br> used as a Loss Function                | Not robust to outliers as it penalizes <br> them to the power of 2                            |
| Mean Absolute Error (MAE)      | $\frac{1}{N} \sum\limits_{n=1}^{N} \|t_{i} - \hat{t_{i}}\|$ | More robust to outliers                                              | Not differentiable which needs <br> to the application of optimisers such as Gradient Descent |
| Root Mean Squared Error (RMSE) | $RMSE = \sqrt{MSE}$                                         | Output is at the same unit as the input (interpretation usefullness) | Not that robust to outliers                                                                   |

* $\textbf{R-Squared} (R^{2})$ $\textcolor{red}{\boldsymbol{\rightarrow} \text{Not a performance metric}}$ <br>
$\color{blue}{\hspace{.5cm}  \hookrightarrow \text{Coefficient of Determinition}}$ <br>
$\color{blue}{\hspace{.5cm}  \hookrightarrow \text{Goodness of Fit}}$ <br>

* **Description:** $R^{2}$ measures how much variance can be explained by your model. $R^{2}$ can also be viewed as how much the regression line is better than the mean line.

* **Formula:** $R^{2} = 1 - \frac{\text{Unexplained Variance}}{\text{Total Variation}} = 1 - \frac{SS_{reg}}{SS_{mean}} = 1 - \frac{\sum\limits_{i=1}^{N} (t_{i} - \hat{t_{i}})^{2}}{\sum\limits_{i=1}^{N} (t_{i} - \overline{t_{i}})}$
$\hspace{9cm} \color{blue}{\hookrightarrow \text{\footnotesize{mean of target variable}}}$

* **Value's Range:** From 0 (bad model) to 1 (perfect model)

>⚠️ $\textcolor{red}{\text{Note: A problem with the } R^{2} \text{ metric is that sometimes it increases as we add more variables even if the added variables are irrelevant.}}$ 
>$\textcolor{red}{\text{In other words, the model can always map some data to a target variable.}}$


* $\textbf{Adjusted R-Squared} (R^{2})$

* **Description:** $R^{2}$ - Adjusted overcomes the incorrect increase of the $R^{2}$ by adding extra independent variables. In other words, it penalaizes the excess amount of independent variables.

* **Formula:** $R_{a}^{2} = 1 - \{ (\frac {n-1}{n-k-1})(1-R^{2})\}$ <br>
$\hspace{1.35cm} \color{blue}{\hookrightarrow \text{\footnotesize{Adjusted }} R^{2}}$

where $n=$ Number of observations, <br> $\hspace{0.9cm} k=$ number of features


* **Value's Range:** From 0 (bad model) to 1 (perfect model)

>⚠️ $\textcolor{red}{\text{Note: As } k \text{ increases, the denominator decreases which makes the entire value to be subtracted from 1 a large value.}}$ 
>$\textcolor{red}{\text{As a result, the } R_{a}^{2} \text{ is decreased which means that} \textbf{ the more (irrelevant features, the worse the model.)}}$

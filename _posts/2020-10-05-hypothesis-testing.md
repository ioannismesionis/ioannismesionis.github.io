---
layout: post
title: Hypothesis Testing
subtitle: How I make up my mind!
bigimg: /img/hypothesis-testing/bigimg-choice2.jpg
image: /img/hypothesis-testing/choice.jpg
tags: [statistics, inference, hypothesis-testing, p-value, critical-region, decision-making]
---
Let the decision begin!

# 1. Introduction
As a data science geek, I constantly try to base a decision based on data and facts rather than gut feeling. <br>
I see it more like a game, to be honest, _"What would data tell me?"_

![](/img/hypothesis-testing/Data-has-a-better-idea.jpg)

Sherlock Holmes would go against this game though as he defines gut feeling as the instinctual assumption based on four factors; observations, knowledge, experiences, and learned behaviours. For the sake of the game, I will bypass Sherlock because really, who wants to argue with Sherlock?

Back to our game, it was one of these sunny days that I had decided to go for a stroll with a friend of fine discussing some new technologies that came out on a specific product that I had expressed my interest - no product placement here, ha!

As we were going back and forth with the pros and cons, I decided to let fate decide whether I should proceed with the purchase or not. And by fate, I mean data.

- "I am not sure I follow..." my friend told me.
- "Well my dearest friend, do you know the 'Let the coin decide for you' game?"
- "Yes - you throw a coin _once_ and you call it heads or tails. Then, you take the respective decision based on the outcome" said my friend.
- "Exactly. The only difference is that I will throw the coin _7 times_ because I want to let fate reveal itself by going against the expected distribution of the coin toss"

The game was on!

To decide whether I should buy my beloved product, I would throw a coin 7 times.

If the outcome was extreme: <br>
I would take it as fate telling me to buy the product.
Else: <br>
It would mean that I should follow my normal path without proceeding with the purchase.

# 2. Terminology
Before we crack on with the problem, it would be a good idea to include some useful terminology that will help us make our decision.

## 2.1 Parameter
A **Parameter** is a summary description of a measure of interest that refers to the _population parameter._ The parameter never changes because everyone/everything was included in the survey to derive the parameter. In other words, it is the ground-truth value that we usually want to estimate from a sample.

_Examples:_ Mean (μ), Variance (σ²), Proportion (π)

## 2.2 Statistic
A **Statistic** is, in a sense, opposite from the parameter and that is because a statistic refers to a small part of the population, i.e. _a sample of the population_

In real-world, it is not feasible to get a complete picture of a population. Therefore, we draw a sample out of the population to estimate the population parameter. The summary description of the sample is called (sample) statistics.

_Examples:_ Sample mean (x̄), Sample Variance (S²), Sample proportion(p)

![Parameter and Statistics by reddit](/img/hypothesis-testing/sample_statistics.png)

## 2.3 Sample Distribution
A **sampling distribution** is the probability distribution of a sample statistic that is formed when samples of size _n_ are repeatedly taken from a population. If the sample statistic is the sample mean, then the distribution is called the sampling distribution of sample means.

![Parameter and Statistics by reddit](/img/hypothesis-testing/sampling_distribution.jpg)

In general, we would expect the sampling the mean of the sampling distribution to be approximately equivalent to the population mean i.e. E(x̄) = μ

# 3. Hypothesis Testing
# 3.1 Critical Region
# 3.2 P-Value

# Conclusion

# References
[Everything you need to know about hypothesis testing](https://towardsdatascience.com/everything-you-need-to-know-about-hypothesis-testing-part-i-4de9abebbc8a)

[What is a Parameter in Statistics](https://www.statisticshowto.com/what-is-a-parameter-statisticshowto/)

[Sample distribution](https://mathbitsnotebook.com/Algebra2/Statistics/STsamplingVariability.html)

[What is sampling distribution](https://www.investopedia.com/terms/s/sampling-distribution.asp#:~:text=A%20sampling%20distribution%20is%20a,a%20statistic%20of%20a%20population.)


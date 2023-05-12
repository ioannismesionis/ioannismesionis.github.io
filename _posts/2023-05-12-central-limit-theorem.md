---
layout: post
title: Central Limit Theorem - A Magic Wand for Inference
subtitle: How the CLT turns any data into gold!
katex: true
image: /img/central-limit-theorem/bell_small.png
bigimg: /img/central-limit-theorem/magic.webp
tags: [statistics, central-limit-theorem, normal-distribution]
---

The central limit theorem (CLT) is a powerful tool that allows us to make inferences about populations, even if we don't know the exact distribution of the population. Just wave your wand (i.e., use the CLT) over a sample of data, and the distribution of the sample means will be approximately normally distributed, regardless of the shape of the population distribution.

# Central Limit Theorem
 ![](/img/central-limit-theorem/sampling_distribution.png)

- **Population** <br>
The population of interest that we want to do some inference on (e.g. average height)

- **Sample** <br>
A large sample from the population (e.g. 100 random people) <br>
    - Each of these samples, will have its *own distribution (i.e. sample distribution)*

- **Sample Statistic** <br>
A sample statistic is a *single* value that corresponds to the sample (e.g. mean) <br>
    - For each sample, we have a single 1-2-1 sample statistic

- **Sampling Distribution** <br>
All of the sample statistics (e.g. means), have their own distribution named the **sampling distribution.**

> **Central Limit Theorem:** The sampling distribution of the mean is nearly normally centred at the population mean, with standard error equal to the population standard deviation divided by the square root of the sample size.
> 
> 
> $\hat{x} \approx N(\text{mean} = \mu, \text{SE} = \frac{\sigma}{\sqrt{n}})$
> 

**Conditions for the CLT:**

1. *Independence:* <br> 
*Sampled observations must be independent*
    a. If sampling without replacement, $n <10$%  of the population
        
        <aside>
        💡 We don’t want to sample too large because it is highly likely that we will select an observation that is not independent. <br>
        
        E.g. If a take a sample of myself, if I have a too large sample size, it is likely I will also sample my mother/father etc.
        </aside>
        
2. *Sample size/skew:* <br>
    a. Either the population distribution is normal. <br>
    b. Either the distribution is skewed, the sample size is large (rule of thumb: $n>30$). <br>
[CLT for means - Interactive examples](https://gallery.shinyapps.io/CLT_mean/)

**Layman’s term explanation:**
The *center limit theorem* states that if any random variable, regardless of the distribution, is sampled a large enough times, the sample mean will be approximately normally distributed. This allows for studying the properties of any statistical distribution as long as there is a large enough sample size.
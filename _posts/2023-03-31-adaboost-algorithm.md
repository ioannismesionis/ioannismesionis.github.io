---
layout: post
title: AdaBoost for the win!
subtitle: Adaptive Boosting - The first of its kind!
katex: true
image: /img/adaboost/adaboost-small.png
bigimg: /img/adaboost/adaboost-big.png
tags: [machine-learning, boosting, decision-trees, random-forest]
---

AdaBoost (Adaptive Boosting) is a popular boosting algorithm that combines multiple weak classifiers to create a strong classifier. <br>
The algorithm works by iteratively adjusting the weights of the training instances and focusing more on the misclassified instances in each iteration using a weighted sampled distribution.

# AdaBoost

Assume we have some data $$\{(x_{i}, y_{i} )\}_{i=1}^{n}$$ with $y_{i}$ being a binary column $\in \{-1, 1\}$.

1. Initialize the weights of the training instances:

    > $w_i^{(1)} = \frac{1}{n}$ for $i = 1, \dots, n$
    >
2. For each iteration from 1 to $t$:

    a. Train a decision tree classifier $h_t(x_i)$ on the training data with weights $w_i^{(t)}$.

    <aside>
    💡 The <b>Gini index</b> is being used to decide which feature will produce the best split to create the <i>stump</i> tree. The Gini index formula is $G = \sum_{i=1}^J p_i (1 - p_i)$ where $J$ is the number of classes, and $p_i$ is the proportion of instances with class $i$ in the current node. <br> <br>

    <i>Note:</i> Each weak classifier should be trained on a random subset of the total training set. AdaBoost assigns a “weight” (i.e. <b>weighted sampling method</b>) to each training example, which determines the <b>probability</b> that each example should appear in the training set.

    </aside>

    <aside>
    💡 AdaBoost makes <b>super small decision trees</b> (i.e. <i>stumps</i>).

    </aside>

    b. Calculate the error of the weak classifier:

    > $\epsilon_t = \sum_{i=1}^n w_i^{(t)} I(y_i \neq h_t(x_i))$
    >

    <aside>
    💡 The error term above basically <b>sums up the weights</b> of the <b>misclassified</b> examples which is being used in the <i>formula c.</i> to calculate the weight of the classifier $h_{t}(x_{i})$.

    </aside>

    c. Calculate the **weight** of the **weak classifier** $h_t(x)$:

    > $\alpha_t = \frac{1}{2} \ln(\frac{1 - \epsilon_t}{\epsilon_t})$
    >

    <aside>
    💡 This is just a straightforward calculation replacing $\epsilon_{t}$ from <i>formula b.</i> above.

    Here is the graph of $\alpha$ looks for different error rates $ln(\frac{1 - \epsilon}{\epsilon})$

    </aside>

    ![](/img/adaboost/adaboost.png)

    <aside>
        <p>💡 The intuition of the graph is three-fold:</p>
        <ol>
            <li>The classifier <b>weight grows exponentially</b> as the <b>error approaches 0</b>.
                Better classifiers are given exponentially more weight.</li>
            <li>The classifier <b>weight is zero</b> if the <b>error rate is 0.5</b>.
                A classifier with 50% accuracy is no better than random guessing, so we ignore it.</li>
            <li>The classifier <b>weight grows exponentially <i>negative</i></b> as the <b>error approaches 1</b>.
                We give a negative weight to classifiers with worse than 50% accuracy. “Whatever that classifier says, do the opposite!”.</li>
        </ol>
    </aside>

    d. Update the **weights** of the *training instances*:

    > $w_i^{(t+1)} = \frac{w_i^{(t)} e^{-\alpha_t y_i h_t(x_i)}}{Z_{t}}$ <br>
    , where $Z_t$ is a normalization factor → $Z_t = \sum_{i=1}^n w_i^{(t)} e^{-\alpha_t y_i h_t(x_i)}$
    >

    <aside>
    💡 In the original paper, the weights $w_{i}$ are described as a distribution. To make it a distribution, all of these probabilities should add up to 1. To ensure this, we normalize the weights by dividing each of them by the sum of all the weights, $Z_t$.
    This just means that each weight $w_{i}$ represents the probability that training example $i$ will be selected as part of the training set. <br> <br>

    Also, $y_i$ is the correct output for the training example $i$, and $h_t(x_i)$ is the predicted output by classifier $t$ on this training example.

    <li> If the predicted and actual output <b>agrees</b> -> $y*h(x)$ will always be +1 (either $1*1$ or $(-1)* (-1)$).</li> <br>

    <li>If the predicted and actual output <b>disagrees</b> -> $y * h(x)$ will be negative.</li>

    </aside>

3. Output the final strong classifier:

    > $H(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t h_t(x)\right)$
    >

    <aside>
    💡 The final classifier consists of $T$ weak classifiers:

    <ol>
        <li>$h_t(x)$ is the output of weak classifier $t$.<br>
            <i>Note:</i> In the paper, the outputs are limited to -1 or +1.</li> <br>
        <li>$\alpha_t$ is the weight applied to classifier $t$ as determined by AdaBoost in <i>step c</i>.</li>
    </ol>

    As a result, the final <b><i>output is just a linear combination of all of the weak classifiers</i></b>, and then we make our final decision simply by looking at the sign of this sum.

    </aside>

---
og_image: /assets/img/blog/cs330/9/autoencoder.png
layout: distill
title: "CS-330 Lecture 8: Variational Inference"
description: "This lecture is part of the CS-330 Deep Multi-Task and Meta Learning course, taught by Chelsea Finn in Fall 2023 at Stanford. This post will talk about variational inference, which is a way of approximating complex distributions through Bayesian inference. We will go from talking about latent variable models all the way to amortized variational inference!"
date: 2024-03-30
tags: course
categories: deep-multi-task-and-meta-learning
comments: true

authors:
  - name: Lars C.P.M. Quaedvlieg
    url: "https://lars-quaedvlieg.github.io/"
    affiliations:
      name: EPFL
      
bibliography: blog/cs330/2024-03-30-var-inf.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Probabilistic models
    subsections:
      - name: Latent variable models
  - name: Variational inference
    subsections:
      - name: Kullback–Leibler divergence
      - name: Tightness of the lower bound
  - name: Amortized variational inference
  - name: Practical examples
    subsections:
      - name: Variational autoencoders
      - name: Conditional models

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }

---

This post will talk about **variational inference**, which is a way of approximating complex distributions through 
Bayesian inference. We will go from talking about latent variable models all the way to amortized variational inference!
If you missed the previous post, which was about automatic task construction for unsupervised meta learning, you can 
head over [here](/blog/2024/cs330-stanford-upt-rbm/) to view it.

The link to the lecture slides can be found [here](https://cs330.stanford.edu/materials/cs330_variational_inference_2023.pdf).

This lecture is taught in order to be able to discuss **Bayesian meta learning** in the next part of the series. 
However, it is a bit different from the rest of the content, so feel free to skip it if you’re already comfortable
with this topic!

# Probabilistic models

<figure class="figure col-sm-12">
 <img src="/assets/img/blog/cs330/9/simple_model.png" class="img-fluid" alt="Alt text.">
 <figcaption class="figure-caption text-center">Simple example of a probabilistic model.</figcaption>
</figure>

Machine learning is all about probabilistic models! In supervised learning, we try to learn a distribution $p(y\vert x)$ over a target variable $y$ using data from $p(x)$. This conditional distribution depends on the assumptions that you make about this target variable $y$. For example, in classification, you might treat $y$ as a categorical variable, which means that $p(y\vert x)$ comes from a discrete **categorical distribution**. However, we also often assume that $p(y\vert x)$ comes from a **Gaussian distribution**. Note that instead of outputting a single value for $y$, out model predicts the distribution $p(y\vert x)$.

The previous two examples are *very common*, but simple, distributions. For some problems, more complex distributions are necessary in order to formulate the problem effectively. As we will see later on, variational inference will allow us to find solutions for these complex distributions!

First, let’s also very quickly discuss some terminology. Using Bayes’ rule, we have the following equation for a parameter $\theta$ and some evidence $X$:

$$
p(\theta\vert X) = \frac{p(X\vert \theta)p(\theta)}{p(X)}\;.
$$

In this equation,

1. $p(\theta\vert X)$ is called the *posterior* distribution. It is the probability after the evidence $X$ is considered.
2. $p(\theta)$ is called the *prior* distribution. It is the probability before the evidence $X$ is considered.
3. $p(X\vert \theta)$ is called the *likelihood*. It is the probability of the evidence, given that $\theta$ is true.
4. $p(X)$ is called the *marginal*. It is the probability of the evidence under any circumstance.

The process of training probabilistic models comes from this idea of likelihood. Given that we observe some data $\mathcal{D} = \\{x_1, x_2, \cdots, x_N\\} \sim X$, we want to learn the data distribution $p(x)$. However, we will consider a parameterized form $p(x\vert \theta) = p_\theta(x)$. The goal becomes to maximize the likelihood of observing the samples in $\mathcal{D}$ given $\theta$:

$$
\max_\theta p_\theta(x_1, x_2, \cdots, x_N) = \max_\theta \prod_i p_\theta(x_i)\;.
$$

This assumes independence $x_i \perp x_j \in \mathcal{D}$. One more trick: Since the $\log$-function is a monotonically increasing function, we can rewrite this objective function without changing the optimal parameters $\theta^*$:

$$
\theta^* \leftarrow \arg\max_\theta \frac{1}{N} \sum_i\log p_\theta(x_i)\;.
$$

This will help a lot, since we got rid of the long chain of multiplications, which could be catastrophic for gradient-based optimization methods. This method is fundamental to statistics, and is called **maximum likelihood estimation.**

For simple distributions, such as the categorical and Gaussian distributions that we saw, there are closed-form evaluations of this function. The maximum likelihood estimate of the categorical distribution results in the **cross-entropy** loss, and the one for the Gaussian distributions is the **mean-squared error** loss.

<figure class="figure col-sm-12">
 <img src="/assets/img/blog/cs330/9/diffusion-example.png" class="img-fluid" alt="Alt text.">
 <figcaption class="figure-caption text-center">Example of a text-to-image model.</figcaption>
</figure>

For some problems, assuming the data comes from these distributions is just too simple. For example, **generative models** over images, text, video, or other data may need a more complex distribution. An example of a text-to-video use-case is depicted above <d-cite key="villegas2022phenaki"></d-cite>. For this, a Gaussian distribution might just be too simple. Another example is the class of problems that require a **multimodal** distribution.

<div>
<figure class="figure col-sm-6 float-right">
    <img src="/assets/img/blog/cs330/9/problem-meta.png" class="img-fluid" alt="Alt text.">
   <figcaption class="figure-caption text-center">Ambiguity in meta learning problems.</figcaption>
</figure>

<p>For meta learning, we are so far using a deterministic (i.e. point estimate) of the distribution 
$p(\phi_i\vert \mathcal{D}^\mathrm{tr}_i, \theta)$. This could be a problem when few-shot learning problems are
<b>ambiguous</b>. One example is depicted in the figure on the right. Depending on the representation that the model 
learns, a point estimate might either learn to distinguish samples based on their <i>youth</i> or whether they are
<i>smiling</i>. The goal is ambiguous from the training dataset on the left. Therefore, it would be nice to learn to
<b>generate hypotheses</b> by sampling from $p(\phi_i\vert \mathcal{D}^\mathrm{tr}_i, \theta)$. This can be important for
<i>safety-critical</i> few-shot learning, learning to <i>active learn</i> <d-cite key="woodward2017active"></d-cite>, and <i>learning how to explore</i> in meta reinforcement learning.</p>
</div>

The main question of this lecture: *Can we model and train more complex distributions?* We will use variational inference to answer this question!

## Latent variable models

<div>
<figure class="figure col-sm-5 float-right">
    <img src="/assets/img/blog/cs330/9/gmm.png" class="img-fluid" alt="Alt text.">
   <figcaption class="figure-caption text-center">Example of a fitted Gaussian Mixture Model.</figcaption>
</figure>

<p>Before we get into variational inference, we will talk about what <b>latent variable</b> models are. We will start by 
using a few examples and building out the idea. Let’s say we are given the data from $p(x)$ in the right figure. As 
you can see, fitting a Gaussian distribution would not work very well in this case.</p>

<p>One common method to model these “clustered” points is by using a <b>Gaussian mixture model</b>. The distribution of such
a model follows the following formula:</p>
</div>

$$
p(x) = \sum_zp(x\vert z)p(z)\;.
$$

In this distribution, we introduce latent (hidden) variables $z$. In this example, we let $p(x\vert z)$ be a normal distribution, and $p(z)$ be a discrete categorical distribution. Notice that in this case, the latent variables model the clusters that datapoints belong to, and the conditional distribution $p(x\vert z)$ treats each individual cluster as a normal distribution. Since $z$ is a distribution, a datapoint can be part of a *mixture* of those gaussian distributions, hence the name of the model.

Furthermore, this is also possible for conditional distributions, i.e.:

$$
p(y \vert x) = \sum_z p(y\vert x,z)p(z \vert x)\;.
$$

<div>
<figure class="figure col-sm-4 float-right">
    <img src="/assets/img/blog/cs330/9/mixture-density-network.png" class="img-fluid" alt="Alt text.">
   <figcaption class="figure-caption text-center">Mixture Density Network in practice.</figcaption>
</figure>

<p>This has the name <b>mixture density network</b>. An example of such a network is shown on the right. Notice that the
model outputs the parameters of the distributions instead of a direct value of $y$, which is the length of the paper 
in this case.</p>

<p>Now that we have seen some examples, let’s generalize it to continuous distributions. Let's observe the equation below:</p>

<p>$$
p(x) = \int p(x\vert z)p(z)dz\;.
$$</p>
</div>

<div>
<figure class="figure col-sm-6 float-right">
    <img src="/assets/img/blog/cs330/9/latent-model.png" class="img-fluid" alt="Alt text.">
</figure>

<p>The core idea stays the same: <b>represent a complex distribution by composing two simple distributions</b>. More often 
than not, $p(x\vert z)$ and $p(z)$ will both be represented as normal distributions.</p>

<p>As you can see in the right figure, we need to <b>sample</b> from $p(z)$ in order to get a sample from $p(x\vert z)$.</p>
</div>

However, now, a few questions arise:

1. How can we generate a sample from $p(x)$ after the model is trained?

   <div class="panel-group">
    <div class="panel panel-default">
      <div class="panel-heading">
        <h4 class="panel-title">
          <a data-toggle="collapse" href="#collapse1">Toggle answer.</a>
        </h4>
      </div>
      <div id="collapse1" class="panel-collapse collapse">
        <div class="panel-body"><p><b>Answer</b>: As we said before, you need to sample a $z$ and then use it to compute $p(x \vert z)$, which you then sample from.</p></div>
      </div>
    </div>
   </div>

2. How do we evaluate the likelihood of a given sample $x_i$ (e.g. $p(x_i)$)?

   <div class="panel-group">
    <div class="panel panel-default">
      <div class="panel-heading">
        <h4 class="panel-title">
          <a data-toggle="collapse" href="#collapse2">Toggle answer.</a>
        </h4>
      </div>
      <div id="collapse2" class="panel-collapse collapse">
        <div class="panel-body"><p><b>Answer</b>: To compute $p(x_i)$, we need to sample <b>many</b> $z$ from the distribution $p(z)$ in order to get a good <i>approximation</i> of the integral that defines the distribution $p(x) = \int p(x\vert z)p(z)dz$.</p></div>
      </div>
    </div>
   </div>

Now that we know how to evaluate and sample from latent variable models, let’s look into how we can train these models. Rewriting the maximum likelihood objective with the latent variable model, we obtain the objective function below:

$$
\theta^* \leftarrow \arg\max_\theta \frac{1}{N} \sum_i\log \left( \int p_\theta(x_i\vert z)p(z)dz\right)\;.
$$

In order to optimize this, we need to find the gradient of this objective. However, the integral in the logarithm is *intractable*, since it usually does not have a nice closed-form expression, in contrary to the simple distributions we have seen before. Approximating the integral by sampling from $p(z)$ is incredibly inefficient.

There exist many papers that use latent variable models, and most of them have (slightly) different ways of training them:

- Generative adversarial networks (GANs) <d-cite key="goodfellow2020generative"></d-cite>
- Variation autoencoders (VAEs) <d-cite key="kingma2013auto"></d-cite>
- Normalizing flow models <d-cite key="kobyzev2020normalizing"></d-cite>
- Diffusion models <d-cite key="ho2020denoising"></d-cite>

Note that autoregressive models do not use latent variables, and we model the target as a categorical distribution, which has the closed-form cross-entropy objective as maximum likelihood estimator.

In this lecture, we will focus on methods that use **variational inference**. They have a number of benefits and are probably the most common methods to train latent variable models.

# Variational inference

In this section, we will introduce variational inference, which is a way of formulating a lower bound on the log-likelihood objective. Furthermore, since we will be optimizing this lower bound, we will look into the *tightness* of the bound.

We will look at an alternative formulation of the log-likelihood objective, which is called the *expected* log-likelihood:

$$
\theta^* \leftarrow \arg\max_\theta \frac{1}{N} \sum_i \mathbb{E}_{z \sim p(z \vert x_i)}[\log p_\theta(x_i, z)]\;.
$$

<div>
<figure class="figure col-sm-5 float-right">
    <img src="/assets/img/blog/cs330/9/q-approx.png" class="img-fluid" alt="Alt text.">
   <figcaption class="figure-caption text-center">Approximation of $p(z\vert x_i)$.</figcaption>
</figure>

<p>It is very similar to what we have seen, but now we sample the latent variable with $p(z\vert x_i)$ to evaluate the
logarithm of the joint distribution $p_\theta(x_i, z)$. The intuition behind this formula is that we can make an 
educated guess of $z$ by using $p(z\vert x_i)$ instead of doing random sampling from $p(z)$. In the figure on the right,
this can be seen as mapping $x_i$ back to the latent distribution $p(z)$.</p>

<p>However, there is a problem. Unfortunately, we do not have access to the distribution $p(z \vert x_i)$. Therefore, we 
will try to approximate this distribution with the <b>variational distribution</b> $q_i(z) := \mathcal{N}(\mu_i, \sigma_i)$. 
Note that this is just an estimate, and it will not perfectly model the distribution, but it will help with quickly 
optimizing the objective function, since we can find likely latent variables given the samples!</p>
</div>

Let’s try to now bound $\log p(x_i)$ and introduce $q_i(z)$!

$$
\begin{align*}
\log p(x_i) &= \log \int p(x_i \vert z)p(z) dz \\
&= \log \int p(x_i \vert z)p(z) \frac{q_i(z)}{q_i(z)} dz \\
&= \log \mathbb{E}_{z \sim q_i}\left[\frac{p(x_i \vert z)p(z)}{q_i(z)}\right] \\
&\geq \mathbb{E}_{z \sim q_i}\left[\log\frac{p(x_i \vert z)p(z)}{q_i(z)}\right] \\
&= \mathbb{E}_{z \sim q_i}\left[\log p(x_i \vert z) + \log p(z) - \log q_i(z)\right] \\
&= \mathbb{E}_{z \sim q_i}\left[\log p(x_i \vert z) + \log p(z)\right] + \mathcal{H}(q_i(z))\;.
\end{align*}
$$

In the equation above, we just introduced $q_i(z)$ by adding the fraction $\frac{q_i(z)}{q_i(z)}$, since it equals $1$. Then, we simple rewrote it as an expectation over $q_i(z)$ instead of $p(z)$. This is much nicer than before, since we can actually compute this expectation instead of evaluating an integral. Then, we used Jensen’s inequality to get a lower bound on the objective. We finally did some simple algebra to simplify it. Note that $\mathcal{H}$ is the entropy function. This bound is called the **evidence lower-bound (ELBO)**.

Let’s spend time to talk about the intuition behind this bound. Since it forms a lower-bound on the original objective, maximizing the **ELBO** will also maximize the towards the optimal value of original objective. However, there might be some gap, but we will discuss this later on.

<div>
<figure class="figure col-sm-6 float-right">
    <img src="/assets/img/blog/cs330/9/elbo-part1.png" class="img-fluid" alt="Alt text.">
</figure>

<p>The term $\mathbb{E}_{z \sim q_i}\left[\log p(x_i \vert z) + \log p(z)\right]$ essentially tries to maximize the 
probability $p(x_i, z)$ for a given $z$. This is highlighted on the figure in the right.</p>
</div>

<div>
<figure class="figure col-sm-6 float-right">
    <img src="/assets/img/blog/cs330/9/elbo-part2.png" class="img-fluid" alt="Alt text.">
</figure>

<p>The second term then tries to maximize the entropy $\mathcal{H}(q_i(z))$. Since the entropy is a measure of 
randomness (e.g. a high entropy corresponds to a high randomness), this term will try to make the fit as random as
possible. you can see this as the yellow part in the figure on the right.</p>
</div>

Hopefully this gives some intuition behind the objective!

## Kullback–Leibler divergence

Let’s take a brief detour and talk about a divergence called the **Kullback–Leibler (KL) divergence**. It is a divergence between two distributions, and it can be denoted by the following equation:

$$
\begin{align*}
D_\mathrm{KL}(q \Vert p) &= \mathbb{E}_{x \sim q}\left[\log \frac{q(x)}{p(x)}\right] \\
&= -\mathbb{E}_{x \sim q}\left[\log p(x)\right] - \mathcal{H}(q(x))\;.
\end{align*}
$$

It can be seen as a *difference* between distributions. But, in the last line, you can see that it also measures how *small* the expected log probability of $p$ under distribution $q$, *minus* the entropy of $q$. We will again build some intuition on this.

<div>
<figure class="figure col-sm-6 float-right">
    <img src="/assets/img/blog/cs330/9/kl.png" class="img-fluid" alt="Alt text.">
</figure>

<p>However, note that we are <b>minimizing</b> the KL-divergence, since we want the distributions to be as similar as 
possible. In this case, we are maximizing the log probability under the other distribution, and we are also maximizing 
the entropy, which will lead to a similar intuition as we saw previously on the <b>ELBO</b>.</p>
</div>

## Tightness of the lower bound

Now that you have seen the similarities between the KL divergence and the ELBO objectives, let’s try to put them together. We will try to do this by rewriting the KL divergence and uncovering the ELBO objective function. Recall that the ELBO objective is

$$
\mathcal{L}_i(p, q_i)= \mathbb{E}_{z \sim q_i}\left[\log p(x_i \vert z) + \log p(z)\right] + \mathcal{H}(q_i(z))\;.
$$

Further recall that we approximated $q_i(z) \approx p(z\vert x_i)$ earlier in order to be able to sample $z$, since we do not have access to $p(z\vert x_i)$. Intuitively, it makes sense that we want $q_i(z)$ to be as close as possible to $p(z\vert x_i)$. Let’s now compute the KL divergence between these distributions to see how well $q_i(z)$ approximates.

$$
\begin{align*}
D_\mathrm{KL}(q_i(z) \Vert p(z\vert x_i)) &= \mathbb{E}_{z \sim q_i(z)}\left[\log\frac{q_i(z)}{p(z\vert x_i)}\right] \\
&= \mathbb{E}_{z \sim q_i(z)}\left[\log\frac{q_i(z)p(x_i)}{p(z, x_i)}\right] \\
&= - \mathbb{E}_{z \sim q_i(z)}\left[\log p(z, x_i)\right] - \mathcal{H}(q_i) + \mathbb{E}_{z \sim q(z)}\left[\log p(x_i)\right] \\
&= - \mathcal{L}_i(p, q_i) + \log p(x_i)\;.
\end{align*}
$$

From the first to the second line, we use that $p(z\vert x_i) = \frac{p(x_i, z)}{p(x_i)}$. Then, from the second to third line, we simplify using the rules of the logarithm and already substitute in the entropy. In the final line, we first use that $\log p(z, x_i) = \log p(x_i\vert z)p(z) = \log p(x_i) + \log(z)$, and we use the tower property ($X \perp Y \Rightarrow \mathbb{E}_Y[X] = X$). The two entropies $\mathcal{H}(q_i)$ also cancel out.

Now, we can rewrite the equation to see the following final form:

$$
\log p(x_i) = D_\mathrm{KL}(q_i(z) \Vert p(z\vert x_i)) + \mathcal{L}_i(p, q_i)\;.
$$

We can finally see that when $D_\mathrm{KL}(q_i(z) \Vert p(z\vert x_i)) = 0$, the ELBO bound is tight! Thus, it depends on how well $q_i(z)$ approximates the actual conditional distribution $p(z\vert x_i)$.

We obtain the final optimization objective for variational inference:

$$
\max_{\theta, q_i} \frac{1}{N} \sum_i \mathcal{L}_i(p_\theta, q_i)\;.
$$

The training process is as follows:

1. Sample mini-batch of $x_1, \cdots, x_N$.
2. Compute $\nabla_\theta \mathcal{L}_i$.
   1. Sample $z_1, \cdots, z_m \sim q_i(x_i)$.
   2. Calculate $\nabla_\theta \frac{1}{m}\sum_j\left[\log p(x_i \vert z_j)\right]$.
3. <p>Update $q_i$ with respect to $\mathcal{L}_i$ (For example, if $q_i := \mathcal{N}(\mu_i, \sigma_i)$, then we get $\nabla_{\mu_i} \mathcal{L}_i$ and $\nabla_{\sigma_i} \mathcal{L}_i$).</p>

# Amortized variational inference

Unfortunately, there is another problem. In this method, we have a $q_i$ for **every datapoint** $x_i$. This is not really feasible for problems with large datasets, since there will be $\vert \theta \vert + (\vert \mu_i \vert + \vert \sigma_i \vert) \times N$ parameters.

<figure class="figure col-sm-12">
 <img src="/assets/img/blog/cs330/9/amortized.png" class="img-fluid" alt="Alt text.">
 <figcaption class="figure-caption text-center">The two models for amortized variational inference.</figcaption>
</figure>

Instead of having a single $q_i$ per sample, we can **train a network** $q_\phi(z \vert x_i) \approx p(z \vert x_i)$! We will essentially obtain two networks. This model would output the necessary parameters $\mu_\phi(x_i)$  and $\sigma_\phi(x_i)$. This technique is called **amortized variational inference**.

In this case, we will obtain the new training process as follows:

1. Sample mini-batch of $x_i$.
2. Calculate $\nabla_\theta \mathcal{L}(p_\theta(x_i \vert z), q_\phi(z \vert x_i))$:
   1. Sample $z \sim q_\phi(z \vert x_i)$.
   2. $\nabla_\theta \mathcal{L} \approx \nabla_\theta \log p_\theta(x_i \vert z)$,
3. $\theta \leftarrow \theta + \nabla_\theta \mathcal{L}$.
4. $\phi \leftarrow \phi + \nabla_\phi \mathcal{L}$.

<p>Now, we need to look more at $\nabla_\phi \mathcal{L} = \nabla_\phi \mathbb{E}_{z \sim q_\phi}\left[\log p_\theta(x_i \vert z) + \log p(z)\right] + \mathcal{H}(q_\phi(z))$. Let’s call $r(x_i, z) = \log p_\theta(x_i \vert z) + \log p(z)$. The question now becomes how do we calculate $\nabla_\phi \mathbb{E}_{z \sim q_i}\left[r(x_i, z)\right]$?</p>

Unfortunately, this is non-differentiable, as it depends on samples from $q_i$. Luckily there is a technique called the **reparameterization trick** for the normal distribution, which works as follows:

$$
q_\phi(z\vert x)=\mathcal{N}(\mu(x), \sigma(x)) = \mu(x) + \epsilon \sigma(x)\;.
$$

In the equation above, $\epsilon \sim \mathcal{N}(0, 1)$. We can now rewrite the gradient of the bottleneck as

$$
\nabla_\phi \mathbb{E}_{z \sim q_i}\left[r(x_i, z)\right] = \nabla_\phi \mathbb{E}_{\epsilon \sim \mathcal{N}(0, 1)}\left[r(x_i, \mu(x_i) + \epsilon \sigma(x_i))\right]\;.
$$

Since $\epsilon$ is independent of $\phi$, as you can see in the equations above, we can do backpropagation after applying this trick! However, we still need to sample $\epsilon_1, \cdots \epsilon_m$ in order to approximate the expectation. In practice, it seems that sampling once works well! This is likely the case because the normal distribution is quite centred around its mean, so one sample is often representative enough of an approximation.

The benefits to this methods are that, even though the proofs might be a bit non-trivial, it is **very easy to implement**. Furthermore, is has **low variance**. Unfortunately though, the reparameterization trick only works with continuous (normal) latent variables. However, there are papers that address this, such as vector-quantized variational autoencoders <d-cite key="van2017neural"></d-cite>.

# Practical examples

## Variational autoencoders

We previous saw the following ELBO objective:

$$
\mathcal{L}_i= \mathbb{E}_{z \sim q_\phi}\left[\log p_\theta(x_i \vert z) + \log p(z)\right] + \mathcal{H}(q_\phi(z))\;.
$$

With some simply algebra, this can actually be rewritten into

$$
\mathcal{L}_i = \mathbb{E}_{z \sim q_\phi}\left[\log p_\theta(x_i \vert z)\right] - D_\mathrm{KL}(q_\phi(z \vert x_i) \Vert p(z))\;.
$$

In this case, for normal random variables, $D_\mathrm{KL}(q_\phi(z \vert x_i) \Vert p(z))$ has a convenient analytical form! Using the reparameterization trick and by sampling one $\epsilon$, the final objective can be written as

$$
\max_{\theta, \phi} \frac{1}{N} \sum_i \log p_\theta(x_i \vert \mu_\phi(x_i) + \epsilon \sigma_\phi(x_i)) - D_\mathrm{KL}(q_\phi(z \vert x_i) \Vert p(z))\;.
$$

<figure class="figure col-sm-12">
 <img src="/assets/img/blog/cs330/9/autoencoder.png" class="img-fluid" alt="Alt text.">
 <figcaption class="figure-caption text-center">The variational autoencoder architecture.</figcaption>
</figure>

<div>
<figure class="figure col-sm-4 float-right">
    <img src="/assets/img/blog/cs330/9/autoencoder-example.png" class="img-fluid" alt="Alt text.">
</figure>

<p>This can very conveniently be expressed with the networks in the figure above. There is an encoder model $q_\phi$ which 
takes an input $x_i$ and compresses it into a latent space $z$, where noise is added to the latent variable. The original
input is then “reconstructed” from the latent variable using $p_\theta(x_i\vert z)$. At inference time, you can generate
similar samples to your input simply by sampling multiple $\epsilon$ and reconstructing them! This can also be seen in 
the image on the right. This was introduced in <d-cite key="kingma2013auto"></d-cite>.</p>
</div>

## Conditional models

<figure class="figure col-sm-12">
 <img src="/assets/img/blog/cs330/9/conditional-autoencoder.png" class="img-fluid" alt="Alt text.">
 <figcaption class="figure-caption text-center">Conditional generation with autoencoders..</figcaption>
</figure>

The idea in <d-cite key="razavi2019generating"></d-cite> stays very similar to variational autoencoders. But now, we will try to model the conditional distribution $p(y\vert x)$ instead of just $p(x)$. The loss stays almost identical, but we just condition on $x_i$:

$$
\mathcal{L}_i= \mathbb{E}_{z \sim q_\phi}\left[\log p_\theta(y_i \vert x_i, z) + \log p(z \vert x_i)\right] + \mathcal{H}(q_\phi(z \vert x_i))\;.
$$

Now, $x_i$ can represent image data or whatever is necessary for conditional generation!

***
---
description: Asset Pricing Model that largely following the sprit of ChatGPT
tags:
  - neural-sdf
  - aipt
---

# KellyKuznetsovMalamudXu\_2025

## Artificial Intelligence Asset Pricing Models

**Author(s):** Bryan Kelly, Boris Kuznetsov, Semyon Malamud, Teng Andrea Xu

**Journal / Year:** NBER Working Paper, 2025

***

## Artificial Intelligence Asset Pricing Models

**Author(s):** Bryan Kelly, Boris Kuznetsov, Semyon Malamud, Teng Andrea Xu\
**Journal / Year:** Working paper, December 2024

***

### 1. Summary

* The paper proposes an **Artificial Intelligence Pricing Model (AIPM)** that embeds a **transformer architecture** into the stochastic discount factor.
* The SDF is written as a traded payoff:

$$
M_{t+1}=1-w(X_t)'R_{t+1}.
$$

* Here, (R\_{t+1}) is the vector of stock excess returns and (X\_t) is the matrix of stock characteristics.
* The main idea is to estimate the SDF portfolio weight function:

$$
w_t=w(X_t).
$$

* Standard machine-learning asset-pricing models usually use **own-asset prediction**:

$$
w_{i,t}=f(X_{i,t}).
$$

* This means asset (i)'s portfolio weight depends only on asset (i)'s own characteristics.
* AIPM instead uses **cross-asset information sharing**:

$$
w_{i,t}=f(X_{i,t},X_{1,t},\ldots,X_{N_t,t}).
$$

* The transformer attention mechanism allows the model to decide which other assets are informative for pricing asset (i).
* Empirically, the nonlinear transformer achieves higher out-of-sample Sharpe ratios and lower pricing errors than standard factor models, linear characteristic models, shallow nonlinear SDF models, and deep MLP models without attention.

***

### 2. Key Idea / Framework

* The paper estimates the SDF using the no-arbitrage condition:

$$
E_t[M_{t+1}R_{t+1}]=0.
$$

* Substituting the SDF gives:

$$
E_t\left[\left(1-w(X_t)'R_{t+1}\right)R_{t+1}\right]=0.
$$

* The estimation objective is maximum Sharpe ratio regression:

$$
\min_{\Theta}
E_t\left[
\left(1-w(X_t;\Theta)'R_{t+1}\right)^2
\right]
+
g(\Theta;z).
$$

* This objective can be interpreted as both:

$$
\text{SDF pricing-error minimization}
$$

and

$$
\text{conditional mean-variance efficient portfolio estimation}.
$$

* Therefore, the model is not simply trained to forecast realized returns. It is trained to recover a pricing kernel / SDF portfolio.

***

### 3. Linear Portfolio Transformer

* The paper first explains the idea using an interpretable linear model.
* A standard linear characteristic SDF is:

$$
w_t=X_t\lambda.
$$

* For asset (i):

$$
w_{i,t}=X_{i,t}'\lambda.
$$

* This is an own-asset model.
* The linear portfolio transformer adds an attention matrix:

$$
w_t=A_tX_t\lambda.
$$

* The attention matrix is:

$$
A_t=X_tWX_t'.
$$

* Therefore:

$$
w_t=(X_tWX_t')X_t\lambda.
$$

* For asset (i):

$$
w_{i,t}
=
\sum_{j=1}^{N_t}
\left(X_{i,t}'WX_{j,t}\right)
\left(X_{j,t}'\lambda\right).
$$

* Interpretation:

$$
X_{j,t}'\lambda
=
\text{signal from asset }j,
$$

$$
X_{i,t}'WX_{j,t}
=
\text{attention from asset }i\text{ to asset }j.
$$

* Thus, asset (i)'s weight is a weighted combination of signals from all assets, not only its own signal.
* This is the paper’s core mechanism:

$$
\text{attention}
=
\text{dynamic cross-asset information sharing}.
$$

***

### 4. Nonlinear Transformer SDF

* The full AIPM replaces the simple linear attention structure with a deep nonlinear transformer.
* A transformer block contains:

$$
\text{attention}
+
\text{softmax}
+
\text{feed-forward network}
+
\text{residual connections}.
$$

* The attention layer is:

$$
A(Y)
=
\sum_{h=1}^{H}
\sigma(YW_hY')YV_h.
$$

* The final SDF portfolio weight is:

$$
w_t=T^{(K)}(X_t)\lambda,
$$

where (T^{(K)}) denotes (K) stacked transformer blocks.

* The final SDF is:

$$
M_{t+1}
=
1-\lambda'T^{(K)}(X_t)'R_{t+1}.
$$

* The key difference from an MLP is that the transformer uses other assets’ characteristics through attention, while the MLP only applies nonlinear transformations to each asset’s own characteristics.

***

### 5. Data and Evaluation

* The empirical analysis uses monthly U.S. stock returns from 1963 to 2022.
* The conditioning set is 132 stock-level characteristics from Jensen, Kelly, and Pedersen.
* The out-of-sample period is 1968 to 2022.
* Models are trained using 60-month rolling windows.
* The main evaluation metrics are:

$$
\text{out-of-sample Sharpe ratio}
$$

and

$$
\text{out-of-sample Hansen-Jagannathan distance}.
$$

* The HJ distance measures pricing errors on 132 JKP anomaly factors.

***

### 6. Main Results

| Model            | Sharpe Ratio | Pricing Error |
| ---------------- | -----------: | ------------: |
| FF6              |         1.05 |          0.55 |
| HXZ              |         1.80 |          0.42 |
| BSV              |         3.60 |          0.15 |
| DKKM             |         3.91 |          0.13 |
| Linear Attention |         3.89 |          0.14 |
| MLP              |         4.31 |          0.13 |
| Transformer      |         4.57 |          0.09 |

* The nonlinear transformer performs best:

$$
SR_{\text{Transformer}}=4.57,
\qquad
HJD_{\text{Transformer}}=0.09.
$$

* The main comparison is with MLP:

$$
SR_{\text{MLP}}=4.31,
\qquad
HJD_{\text{MLP}}=0.13.
$$

* Since MLP has deep nonlinearity but no cross-asset attention, the improvement from MLP to Transformer is interpreted as evidence that attention-based information sharing adds value.
* The paper also finds that transformer gains are especially large among large and mega stocks, not only micro stocks.

***

### 7. Contribution & Takeaway

* The paper’s first contribution is to define AIPM as a transformer-based SDF model:

$$
M_{t+1}=1-w(X_t)'R_{t+1}.
$$

* The second contribution is to show that attention has a clear asset-pricing interpretation:

$$
w_{i,t}
=
\sum_j
\left(X_{i,t}'WX_{j,t}\right)
\left(X_{j,t}'\lambda\right).
$$

* This means asset (i)'s SDF weight is constructed using information from other assets.
* The third contribution is empirical: transformer-based SDFs outperform standard factor models and machine-learning SDF benchmarks.

$$
\text{SDF estimation with cross-asset attention}
>
\text{own-asset deep learning without attention}.
$$

* In words, the paper argues that asset pricing models should treat the cross-section of stocks as a contextual system, not as isolated assets.

---
tags:
  - neural-sdf
---

# LiangChenKyriakou\_2025

***

## Nonlinear Pricing Kernel via Explainable Neural Networks

**Author(s):** Jiawen Liang, Cathy Yi-Hsuan Chen, Ioannis Kyriakou\
**Journal / Year:** Working paper, first draft December 2024; this draft October 2025

### 1. Summary

* The paper estimates a nonlinear pricing kernel, or stochastic discount factor (SDF), using a feedforward neural network.
* The pricing kernel is modeled as a positive function of characteristic-based factors:

$$
m(f_t):\mathbb{R}^Q \rightarrow \mathbb{R}_+.
$$

* The central asset-pricing restriction is the unconditional Euler equation:

$$
E[m(f_t)f_t]=0.
$$

* The model estimates the SDF by minimizing a quadratic form of pricing errors:

$$
\min_{\omega\in\Omega}
g(\omega)^\prime \Sigma^{-1} g(\omega),
\qquad
g(\omega)=\frac{1}{T}\sum_{t=1}^T m^{NN}(f_t;\omega)f_t.
$$

* The non-negativity condition,

$$
m^{NN}(f_t;\omega)>0,
$$

is imposed directly through the neural-network architecture by using a Softplus output layer.

* Empirically, the nonlinear neural SDF strongly outperforms linear benchmarks including least squares, Elastic Net, and PCA in out-of-sample pricing errors.

### 2. Backdrops

* Standard SDF models often assume a linear pricing kernel:

$$
m_t = a + b^\prime f_t.
$$

* This linear form can be restrictive in a high-dimensional factor setting because:
  * factor effects may be nonlinear;
  * characteristics may interact with each other;
  * many factors are highly correlated;
  * linear models are exposed to severe estimation error;
  * feature importance in nonlinear models is difficult to test statistically.
* The paper builds on the nonlinear SDF literature, including polynomial approximations, sieve methods, early neural-network SDF models, and recent GAN-based SDF estimation.
* Its main distinction is that it directly parameterizes the entire pricing kernel as a neural network while imposing the no-arbitrage positivity condition by construction.

### 3. Framework / Modeling

* The factor vector is constructed as

$$
f_t = Z_{t-1}^\prime R_t^e,
$$

where $$R_t^e\in\mathbb{R}^N$$ is the vector of excess returns and $$Z_{t-1}\in\mathbb{R}^{N\times Q}$$ contains lagged instruments or characteristics.

* The pricing condition is

$$
E[m(f_t)f_t]=0.
$$

* The neural network starts with

$$
X^{(0)}=f_t.
$$

* For hidden layer $$\ell$$,

$$
X^{(\ell)}
=
\operatorname{Sigmoid}
\left(
W^{(\ell)}X^{(\ell-1)}+\omega_0^{(\ell)}
\right).
$$

* The output layer is

$$
X^{(L+1)}
=
\operatorname{Softplus}
\left(
W^{(L+1)}X^{(L)}+\omega_0^{(L+1)}
\right).
$$

* Therefore, the neural-network SDF is

$$
m^{NN}(f_t;\omega)=X^{(L+1)}.
$$

* Since

$$
\operatorname{Softplus}(z)=\log(1+e^z),
$$

the SDF is positive by construction:

$$
m^{NN}(f_t;\omega)>0.
$$

* The constrained optimization problem is

$$
\min_{\omega\in\Omega}
g(\omega)^\prime \Sigma^{-1}g(\omega),
\qquad
\text{s.t.}
\qquad
m^{NN}(f_t;\omega)>0.
$$

* Because the Softplus output enforces positivity automatically, the constrained SDF problem becomes an unconstrained non-convex neural-network optimization problem.
* The moment function is

$$
g(\omega)
=
\frac{1}{T}
\sum_{t=1}^T
m^{NN}(f_t;\omega)f_t.
$$

* The weighting matrix is

$$
\Sigma=E[f_tf_t^\prime],
$$

where $$f_t$$ denotes demeaned factors.

* The squared pricing error corresponds to an HJ-distance type criterion:

$$
g(\omega)^\prime \Sigma^{-1}g(\omega).
$$

* The paper compares neural-network SDFs with linear SDFs using a model specification test.
* Let $$D_j^2$$ denote the out-of-sample squared HJ distance of neural network $$j$$, and $$D_i^2$$ denote the corresponding value for a linear model $$i$$.
* The null hypothesis is

$$
H_0:
\max_{j\in J^{NN}}
\{D_j^2-D_i^2\}
\le 0.
$$

* This null means that even the worst-performing neural network has no larger pricing error than the linear benchmark.
* The alternative is

$$
H_1:
\max_{j\in J^{NN}}
\{D_j^2-D_i^2\}
>0.
$$

* The test statistic is

$$
\phi_i
=
\max_{j\in J^{NN}}
\sqrt{T_{te}}
\{D_j^2-D_i^2\}.
$$

* The distribution of the test statistic is obtained using a block bootstrap with block length equal to 21 trading days.
* The paper also selects the best neural-network architecture through a similar hypothesis-testing procedure.
* If $$\tilde j$$ is the network with the smallest in-sample squared HJ distance, the null is

$$
H_0:
\max_{j\in J^{NN}\setminus\{\tilde j\}}
\{D_{\tilde j}^2-D_j^2\}
\le 0.
$$

* The corresponding test statistic is

$$
\delta_{\tilde j}
=
\max_{j\in J^{NN}\setminus\{\tilde j\}}
\sqrt{T_{te}}
\{D_{\tilde j}^2-D_j^2\}.
$$

* For factor significance, the paper defines

$$
\xi_q
=
E
\left[
\left(
\frac{\partial m(f_t)}{\partial f_q}
\right)^2
\right].
$$

* The null hypothesis for factor $$q$$ is

$$
H_0:\xi_q=0.
$$

* The alternative is

$$
H_1:\xi_q\ne 0.
$$

* If $$\xi_q$$ is large, then factor $$f_q$$ has a statistically meaningful effect on the neural-network pricing kernel.

### 4. Data / Results

* The empirical application uses U.S. equities.
* The data sources are:
  * CRSP daily stock returns;
  * one-month Treasury rate from the Fama-French data library;
  * 50 firm characteristics from Kozak et al. (2020);
  * 8 ESG characteristics from Refinitiv ASSET4.
* The sample period is January 2002 to December 2019.
* The final stock sample contains 1,375 stocks after requiring both traditional characteristics and ESG data coverage.
* The paper constructs 58 characteristic-based factors:
  * 50 traditional characteristics;
  * 8 ESG-related characteristics.
* The ESG variables are:
  * ESGSCORE;
  * ENVSCORE;
  * SOCIALSCORE;
  * GOVSCORE;
  * EMISSION;
  * ENVINNOVA;
  * RESUSE;
  * GREENNESS.
* Pricing error comparison:

| Model |               In-sample |          Out-of-sample |
| ----- | ----------------------: | ---------------------: |
| LS    | $$3.66\times 10^{-12}$$ | $$9.29\times 10^{-2}$$ |
| EN    |  $$2.54\times 10^{-2}$$ | $$8.67\times 10^{-2}$$ |
| PCA   | $$1.88\times 10^{-17}$$ | $$1.19\times 10^{-1}$$ |
| NN1   |  $$3.62\times 10^{-6}$$ | $$2.17\times 10^{-5}$$ |
| NN2   |  $$1.16\times 10^{-5}$$ | $$8.04\times 10^{-5}$$ |
| NN3   |  $$2.00\times 10^{-5}$$ | $$1.45\times 10^{-4}$$ |
| NN4   |  $$2.40\times 10^{-5}$$ | $$1.55\times 10^{-4}$$ |
| NN5   |  $$2.05\times 10^{-5}$$ | $$1.46\times 10^{-4}$$ |

* LS and PCA achieve extremely small in-sample errors but much larger out-of-sample errors.
* This indicates severe overfitting in the linear models.
* Elastic Net performs poorly both in-sample and out-of-sample, suggesting that linear regularization alone does not capture the nonlinear structure of the pricing kernel.
* All neural-network models outperform all linear benchmarks out-of-sample.
* NN1, the network with one hidden layer, has the smallest out-of-sample pricing error:

$$
2.17\times 10^{-5}.
$$

* The model specification test results are:

| Test         | Test statistic |   q90 |   q95 |   q99 | p-value |
| ------------ | -------------: | ----: | ----: | ----: | ------: |
| LS vs NNs    |          -2.79 | -1.70 | -1.57 | -1.35 |    0.85 |
| EN vs NNs    |          -2.60 | -1.92 | -1.78 | -1.55 |    0.61 |
| PCA vs NNs   |          -3.58 | -1.76 | -1.61 | -1.36 |    0.98 |
| NN selection |          -0.18 | -0.03 | -0.01 |  0.02 |    0.77 |

* In all comparisons between linear models and neural networks, the null is not rejected.
* Therefore, the paper concludes that even the worst neural-network SDF has smaller pricing errors than the corresponding linear benchmark.
* In the NN selection test, NN1 is not rejected as the best neural-network architecture.
* The paper concludes that the shallow one-hidden-layer network best balances approximation power and estimation risk.
* The factor significance test identifies the following top five pricing factors:
  * GLTNOA;
  * INV;
  * BETAARB;
  * SIZE;
  * SGROWTH.
* Most ESG factors are statistically significant in the neural SDF.
* The exception is RESUSE, which is not significant.
* The paper also visualizes the estimated pricing kernel over time.
* The estimated SDF is always nonnegative, as imposed by the Softplus output layer.
* The pricing kernel is countercyclical: its level and volatility rise during NBER recession periods, especially around 2008.
* Robustness checks use additional test assets:
  * 36 Fama-French double-sorted portfolios;
  * 49 industry portfolios;
  * 118 Hou-Xue-Zhang one-way sorted portfolios.
* Pricing errors remain small for these test assets.
* State-dependent tests split the sample by high and low macroeconomic states based on:
  * dividend-price ratio;
  * earnings-price ratio;
  * book-to-market ratio;
  * net equity expansion;
  * Treasury-bill rate;
  * term spread;
  * default spread;
  * stock variance.
* NN1 consistently produces the lowest pricing errors across these states.

### 5. Contribution

* The paper contributes a nonlinear SDF estimation framework based on constrained neural-network optimization.
* The key methodological contribution is the combination of:

$$
\text{Neural SDF}
+
\text{Euler equation loss}
+
\text{SDF positivity constraint}
+
\text{formal specification testing}.
$$

* The SDF is not estimated as a linear combination of factors. Instead, it is estimated as

$$
m^{NN}(f_t;\omega),
$$

a nonlinear function of characteristic-based factor returns.

* The no-arbitrage non-negativity condition is imposed directly through the network architecture:

$$
m^{NN}(f_t;\omega)>0.
$$

* The paper also contributes a formal hypothesis-testing procedure for:
  * comparing nonlinear neural SDFs against linear SDFs;
  * selecting the best neural-network architecture;
  * testing factor significance inside the neural-network pricing kernel.
* Empirically, the main result is that nonlinear neural pricing kernels produce much smaller out-of-sample pricing errors than LS, Elastic Net, and PCA.
* The paper also shows that ESG characteristics can be statistically significant inside a nonlinear SDF, even when their role may be obscured in linear models due to factor correlation and misspecification.
* The most important practical implication is that asset-pricing neural networks should not be treated only as return-forecasting machines. They can be designed as SDF estimators by embedding pricing restrictions directly into the objective function and architecture.

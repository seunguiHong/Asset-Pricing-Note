---
tags:
  - neural-sdf
---

# ChenPelgerZhu\_2024

## Deep Learning in Asset Pricing

**Author(s):** Luyang Chen, Markus Pelger, Jason Zhu\
**Journal / Year:** _Management Science (2024)_

***

### 1. Summary

The paper proposes a deep-learning asset-pricing model that estimates the stochastic discount factor directly from individual stock returns.

The central idea is not to forecast realized returns by minimizing prediction error, but to estimate an SDF that satisfies the fundamental no-arbitrage condition:

$$
E_t\left[M_{t+1}R^e_{i,t+1}\right]=0.
$$

The model represents the SDF as a traded payoff:

$$
M_{t+1}=1-\omega_t^\top R^e_{t+1},
$$

where $$\omega_t$$ is a time-varying portfolio weight vector over individual stocks. This connects the SDF to a conditional mean-variance efficient portfolio.

The paper uses three key neural-network components:

$$
\omega_{i,t}=\omega(h_t,I_{i,t};\theta),
$$

$$
g_{i,t}=g(h^g_t,I_{i,t};\eta),
$$

$$
h_t=LSTM(x_0,\ldots,x_t).
$$

The SDF network estimates portfolio weights, the adversarial network constructs hard-to-price managed test assets, and the LSTM extracts hidden macroeconomic states from many macro time series.

Empirically, the model outperforms linear SDF models, standard deep return-forecasting models, and Fama-French factor models in out-of-sample Sharpe ratio, explained variation, and pricing errors. The paper reports an annual out-of-sample Sharpe ratio of 2.6, compared with 1.7 for the linear special case, 1.5 for the deep-learning forecasting model, and 0.8 for the Fama-French five-factor model. It also explains 8% of individual stock return variation and 23% of expected returns.

***

### 2. Key Idea / Framework

The starting point is the standard asset-pricing equation:

$$
p_t=E_t[M_{t+1}x_{t+1}].
$$

For a gross return,

$$
1=E_t[M_{t+1}R_{i,t+1}].
$$

For an excess return,

$$
E_t[M_{t+1}R^e_{i,t+1}]=0.
$$

This is the fundamental no-arbitrage moment. The paper uses this moment as the criterion function rather than using a return-prediction loss. The abstract explicitly states that the key innovations are to use the fundamental no-arbitrage condition as the criterion function, construct informative test assets adversarially, and extract economic states from macroeconomic time series.

The SDF is written as

$$
M_{t+1}=1-\omega_t^\top R^e_{t+1}.
$$

Here,

$$
R^e_{t+1}=(R^e_{1,t+1},\ldots,R^e_{N,t+1})^\top
$$

and

$$
\omega_t^\top R^e_{t+1}=\sum_{i=1}^{N}\omega_{i,t}R^e_{i,t+1}.
$$

Define

$$
F_{t+1}=\omega_t^\top R^e_{t+1}.
$$

Then

$$
M_{t+1}=1-F_{t+1}.
$$

This means the SDF is an affine transformation of a traded excess-return portfolio.

Plugging this into the excess-return pricing condition gives

$$
E_t[(1-\omega_t^\top R^e_{t+1})R^e_{t+1}]=0.
$$

Expanding,

$$
E_t[R^e_{t+1}]-E_t[R^e_{t+1}R^{e\top}_{t+1}]\omega_t=0.
$$

Therefore,

$$
\omega_t=E_t[R^e_{t+1}R^{e\top}_{t+1}]^{-1}E_t[R^e_{t+1}].
$$

So $$\omega_t^\top R^e_{t+1}$$ can be interpreted as the conditional mean-variance efficient portfolio payoff. This is why the SDF representation is not arbitrary; it is the traded-SDF / asset-span representation.

The associated risk-premium relation is

$$
E_t[R^e_{i,t+1}]=-\frac{\operatorname{Cov}_t(M_{t+1},R^e_{i,t+1})}{E_t[M_{t+1}]}.
$$

Since

$$
M_{t+1}=1-F_{t+1},
$$

we have

$$
\operatorname{Cov}_t(M_{t+1},R^e_{i,t+1})=-\operatorname{Cov}_t(F_{t+1},R^e_{i,t+1}).
$$

Thus,

$$
E_t[R^e_{i,t+1}]=\frac{\operatorname{Cov}_t(F_{t+1},R^e_{i,t+1})}{E_t[M_{t+1}]}.
$$

Equivalently, with a one-factor representation,

$$
R^e_{i,t+1}=\beta_{i,t}F_{t+1}+\varepsilon_{i,t+1},
$$

where

$$
\beta_{i,t}=\frac{\operatorname{Cov}_t(R^e_{i,t+1},F_{t+1})}{\operatorname{Var}_t(F_{t+1})}.
$$

Then,

$$
E_t[R^e_{i,t+1}]=\beta_{i,t}E_t[F_{t+1}].
$$

The paper therefore treats expected returns as compensation for exposure to the learned SDF factor, not as a pure return-forecasting target.

***

### 3. Conditional Moments and Managed Test Assets

The conditional no-arbitrage restriction is

$$
E_t[M_{t+1}R^e_{i,t+1}]=0.
$$

For any time- $$t$$  information function $$g(I_t,I_{i,t})$$,

$$
E_t[M_{t+1}R^e_{i,t+1}g(I_t,I_{i,t})]=g(I_t,I_{i,t})E_t[M_{t+1}R^e_{i,t+1}]=0.
$$

Taking unconditional expectations gives

$$
E[M_{t+1}R^e_{i,t+1}g(I_t,I_{i,t})]=0.
$$

Thus,

$$
E[(1-\omega_t^\top R^e_{t+1})R^e_{i,t+1}g(I_t,I_{i,t})]=0.
$$

The function g is not just a technical device. It creates characteristic-managed test assets. For example,

$$
g(I_{i,t})=\mathbf{1}\{\text{small stock}\}
$$

creates a small-stock managed portfolio moment:

$$
E[M_{t+1}R^e_{i,t+1}\mathbf{1}\{\text{small}\}]=0.
$$

Similarly,

$$
g(I_{i,t})=\mathbf{1}\{\text{high book-to-market}\}
$$

creates a value-stock pricing moment:

$$
E[M_{t+1}R^e_{i,t+1}\mathbf{1}\{\text{value}\}]=0.
$$

The paper generalizes this idea by letting a neural network choose $$g$$. The adversarial network creates characteristic-managed portfolios with the largest pricing errors for a candidate SDF, and those hard-to-price portfolios are then used to estimate a better SDF.

The key adversarial GMM objective is

$$
\min_{\omega}\max_g \frac{1}{N}\sum_{j=1}^{N}\left|E\left[\left(1-\sum_{i=1}^{N}\omega(I_t,I_{i,t})R^e_{i,t+1}\right)R^e_{j,t+1}g(I_t,I_{j,t})\right]\right|^2.
$$

Here,

$$\omega$$ chooses the SDF portfolio weights to reduce pricing errors, while $$g$$ chooses the managed test assets that maximize pricing errors.

So the model solves

$$
\text{best SDF against worst test assets}.
$$

This is why the paper calls the method a generative adversarial method of moments.

***

### 4. Neural-Network Implementation

The SDF portfolio weight for stock $$i$$ at time $$t$$ is modeled as

$$
\omega_{i,t}=\omega(h_t,I_{i,t};\theta).
$$

Here,

$$I_{i,t}$$ denotes firm characteristics, and $$h_t$$ is a hidden macroeconomic state.

The SDF factor is

$$
F_{t+1}(\theta)=\sum_{i=1}^{N_t}\omega(h_t,I_{i,t};\theta)R^e_{i,t+1}.
$$

The SDF is

$$
M_{t+1}(\theta)=1-F_{t+1}(\theta).
$$

The adversarial instrument is

$$
g_{i,t}=g(h^g_t,I_{i,t};\eta).
$$

The pricing moment for test asset j is

$$
m_j(\theta,\eta)=E[M_{t+1}(\theta)R^e_{j,t+1}g(h^g_t,I_{j,t};\eta)].
$$

The loss is

$$
L(\theta,\eta)=\frac{1}{N}\sum_{j=1}^{N}|m_j(\theta,\eta)|^2.
$$

The SDF network minimizes this loss:

$$
\hat\theta=\arg\min_\theta L(\theta,\hat\eta).
$$

The adversarial network maximizes it:

$$
\hat\eta=\arg\max_\eta L(\hat\theta,\eta).
$$

Therefore,

$$
(\hat\theta,\hat\eta)=\arg\min_\theta\arg\max_\eta L(\theta,\eta).
$$

The macro state is extracted using an LSTM:

$$
h_t=LSTM(x_0,\ldots,x_t),
$$

where $$x_t$$ is the macroeconomic data vector. The paper’s empirical design uses 178 macroeconomic time series and extracts states of the economy from them.

***

### 5. Benchmarks

The main benchmark is a standard deep-learning return-forecasting model.

It estimates

$$
\mu_{i,t}=E_t[R^e_{i,t+1}] \approx f_\theta(h_t,I_{i,t})
$$

by minimizing prediction error:

$$
\min_\theta \sum_{t,i}\left(R^e_{i,t+1}-f_\theta(h_t,I_{i,t})\right)^2.
$$

Thus,

$$
\text{FFN benchmark: }\min_\theta (R^e-\hat R^e)^2.
$$

The proposed GAN-SDF model instead solves

$$
\text{GAN-SDF: }\min_\theta\max_\eta |E[M R^e g]|^2.
$$

This distinction is central. The benchmark tries to predict noisy realized returns. The proposed model tries to satisfy no-arbitrage pricing moments.

The paper also considers a linear SDF benchmark:

$$
\omega_{i,t}=\theta^\top I_{i,t}.
$$

Then

$$
F_{t+1}=\sum_i \theta^\top I_{i,t}R^e_{i,t+1}=\theta^\top\left(\sum_i I_{i,t}R^e_{i,t+1}\right).
$$

Define the characteristic-managed factor:

$$
\tilde F_{t+1}=\sum_i I_{i,t}R^e_{i,t+1}.
$$

Then

$$
F_{t+1}=\theta^\top \tilde F_{t+1}.
$$

The linear SDF is

$$
M_{t+1}=1-\theta^\top\tilde F_{t+1}.
$$

So the neural-network model generalizes the linear characteristic-managed SDF:

$$
\omega_{i,t}=\theta^\top I_{i,t}\quad\longrightarrow\quad\omega_{i,t}=\omega(h_t,I_{i,t};\theta).
$$

***

### 6. Data

The empirical analysis uses all available U.S. stocks from CRSP with monthly returns from 1967 to 2016.

The conditioning information consists of 46 time-varying firm-specific characteristics and 178 macroeconomic time series. The paper states that this set includes the relevant pricing anomalies and forecasting variables for the equity risk premium.

The training, validation, and test split is designed to evaluate out-of-sample asset-pricing performance. The model is evaluated on individual stocks as well as standard anomaly-sorted test portfolios.

***

### 7. Contribution & Results

The first contribution is to replace return-prediction loss with a no-arbitrage SDF loss.

Standard deep learning would estimate

$$
E_t[R^e_{i,t+1}]
$$

directly. This paper instead estimates

$$
M_{t+1}
$$

from the restriction

$$
E_t[M_{t+1}R^e_{i,t+1}]=0.
$$

The second contribution is adversarial test-asset construction. Rather than choosing test assets ex ante, the paper learns $$g$$ to construct characteristic-managed portfolios with the largest pricing errors.

The third contribution is macro-state extraction. The model uses LSTM states

$$
h_t=LSTM(x_0,\ldots,x_t)
$$

to summarize high-dimensional macroeconomic information.

The fourth contribution is empirical. The model outperforms out-of-sample benchmark approaches in Sharpe ratio, explained variation, and pricing errors. The reported annual out-of-sample Sharpe ratio is

$$
SR^{GAN}_{annual}=2.6,
$$

compared with

$$
SR^{Linear}=1.7,
$$

$$
SR^{FFN}=1.5,
$$

$$
SR^{FF5}=0.8.
$$

The model explains

$$
8\%
$$

of individual stock return variation and

$$
23\%
$$

of expected returns. On all 46 anomaly-sorted decile portfolios, the model achieves cross-sectional $$R^2$$ above 90%.

The main interpretation is:

$$
\text{No-arbitrage restriction}+\text{flexible neural networks}+\text{adversarial test assets}
$$

works better than

$$
\text{flexible return prediction alone}.
$$

In the authors’ interpretation, economic constraints discipline the learning algorithm and help detect the underlying SDF structure; off-the-shelf prediction approaches can perform worse than even linear no-arbitrage models.

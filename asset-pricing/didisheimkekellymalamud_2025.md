---
description: >-
  AIPT. Contrast to low-dimensional factor structure, the return is more likely
  to be explained by large number of factors.
tags:
  - neural-sdf
  - aipt
---

# DidisheimKeKellyMalamud\_2025

## DidisheimKeKellyMalamud\_2025

### APT or “AIPT”? The Surprising Dominance of Large Factor Models

**Author(s):** Antoine Didisheim, Shikun (Barry) Ke, Bryan T. Kelly, Semyon Malamud\
**Journal / Year:** NBER Working Paper, September 2024; Revised August 2025

***

### 1. Summary

* The paper proposes **Artificial Intelligence Pricing Theory (AIPT)** as an alternative to the traditional **Arbitrage Pricing Theory (APT)**.
* The traditional APT is based on a **few-factor view**:

$$
\text{APT: returns are driven by a small number of common factors.}
$$

* AIPT instead proposes a **many-factor view**:

$$
\text{AIPT: returns are driven by a large number of factors.}
$$

* The paper constructs a large number of nonlinear characteristic-managed factors from stock characteristics.
* The SDF is represented as a traded payoff:

$$
M_{t+1}
=
1-w(Z_t)'R_{t+1}.
$$

* The unknown SDF portfolio weight function is approximated using a large nonlinear basis expansion:

$$
w(Z_t)
\approx
\sum_{p=1}^{P}\lambda_p S_p(Z_t)
=
S_t\lambda.
$$

* This gives the factor-SDF representation:

$$
M_{t+1}
\approx
1-\lambda'S_t'R_{t+1}
=
1-\lambda'F_{t+1},
$$

where

$$
F_{t+1}
=
S_t'R_{t+1}.
$$

* The central empirical finding is:

$$
P \uparrow
\quad
\Rightarrow
\quad
SR_{OS} \uparrow,
\qquad
D_{OS}^{HJ} \downarrow.
$$

* The largest model uses:

$$
P=360{,}000
$$

factors with a 360-month training window:

$$
T=360,
\qquad
c=\frac{P}{T}=1000.
$$

* The large factor SDF outperforms standard low-dimensional factor models such as FF6, HXZ, SY, DHS, and BS.
* The paper argues that the empirical evidence supports the many-factor AIPT view rather than the few-factor APT view. [oai\_citation:0‡DidisheimKeKellyMalamud\_2025.pdf](sediment://file_000000009e747209b78a63d89679291f)

***

### 2. Backdrops

* The APT of Ross (1976) assumes that a small number of common factors explain the joint variation of asset returns.
* Most empirical asset-pricing models follow this low-dimensional structure:

$$
M_{t+1}
=
1-\lambda'f_{t+1},
\qquad
\dim(f_{t+1}) \text{ small}.
$$

* Standard examples include:
  * Fama-French factor models.
  * Hou-Xue-Zhang q-factor models.
  * Stambaugh-Yuan mispricing factors.
  * Daniel-Hirshleifer-Sun behavioral factors.
  * Barillas-Shanken selected factor models.
* The conventional view favors parsimony:

$$
\text{Few factors}
+
\text{simple linear structure}
\Rightarrow
\text{tractable asset-pricing model}.
$$

* The paper challenges this view using recent machine-learning evidence that overparameterized models can perform well out-of-sample.
* The relevant machine-learning idea is the **virtue of complexity**:

$$
\text{More parameters}
\not\Rightarrow
\text{worse out-of-sample performance}.
$$

* In asset pricing, the paper asks whether the SDF is better approximated by many nonlinear factors rather than a few linear factors.
* The paper separates two effects:

$$
\text{More raw data}
\neq
\text{more model complexity}.
$$

* The empirical design holds the raw information set fixed and varies only the number of generated factors:

$$
Z_t \text{ fixed},
\qquad
P \uparrow.
$$

* This makes the main test:

$$
\text{Does increasing the number of nonlinear factors improve SDF performance?}
$$

***

### 3. Framework / Modeling

#### 3.1 Conditional SDF as a Traded Portfolio

* The starting point is the conditional SDF representation:

$$
M_{t+1}
=
1-w(Z_t)'R_{t+1}.
$$

* (R\_{t+1}) is the vector of excess returns on risky assets:

$$
R_{t+1}
\in
\mathbb{R}^{N_t}.
$$

* (Z\_t) is the stock-characteristic matrix:

$$
Z_t
\in
\mathbb{R}^{N_t \times D}.
$$

* (w(Z\_t)) is the conditional SDF portfolio weight:

$$
w(Z_t)
\in
\mathbb{R}^{N_t}.
$$

* A good SDF should satisfy the pricing condition:

$$
E_t[M_{t+1}R_{t+1}]
=
0.
$$

* Substituting the traded-payoff SDF:

$$
E_t[(1-w(Z_t)'R_{t+1})R_{t+1}]
=
0.
$$

* Thus, estimating the SDF is equivalent to estimating the conditional portfolio weight function (w(Z\_t)).

***

#### 3.2 From Low-Dimensional Factor Models to AIPT

* A standard low-dimensional factor model restricts the SDF to a small number of factors:

$$
M_{t+1}^{\text{small}}
=
1-\lambda'f_{t+1},
\qquad
f_{t+1}
\in
\mathbb{R}^{K},
\qquad
K \ll T.
$$

* In characteristic-based form, this corresponds to a restricted SDF weight function:

$$
w(Z_t)
=
Z_t^{\text{small}}\lambda.
$$

* For example, a Fama-French-type model uses only a small set of characteristics:

$$
Z_t^{\text{small}}
=
[
\text{market},
\text{size},
\text{value},
\ldots
].
$$

* The AIPT approach replaces the low-dimensional restriction with a large nonlinear basis:

$$
w(Z_t)
\approx
\sum_{p=1}^{P}\lambda_p S_p(Z_t).
$$

* In matrix form:

$$
w(Z_t)
\approx
S_t\lambda,
$$

where

$$
S_t
=
[
S_{1,t},
S_{2,t},
\ldots,
S_{P,t}
]
\in
\mathbb{R}^{N_t \times P}.
$$

* The SDF becomes:

$$
M_{t+1}
\approx
1-\lambda'S_t'R_{t+1}.
$$

* Define generated factor returns:

$$
F_{t+1}
=
S_t'R_{t+1}
\in
\mathbb{R}^{P}.
$$

* Then:

$$
M_{t+1}
\approx
1-\lambda'F_{t+1}.
$$

* Therefore, the model is:

$$
\text{nonlinear in characteristics}
\quad
\text{but linear in generated factors}.
$$

***

#### 3.3 Characteristic-Managed Factors

* Each generated signal (S\_p(Z\_t)) is used as a portfolio weight over individual stocks.
* The (p)-th factor return is:

$$
F_{p,t+1}
=
S_p(Z_t)'R_{t+1}.
$$

* The full factor vector is:

$$
F_{t+1}
=
\begin{bmatrix}
F_{1,t+1} \\
F_{2,t+1} \\
\vdots \\
F_{P,t+1}
\end{bmatrix}
=
S_t'R_{t+1}.
$$

* The SDF portfolio return is the linear combination of these factors:

$$
R_{t+1}^{M}
=
\lambda'F_{t+1}.
$$

* The SDF is:

$$
M_{t+1}
=
1-R_{t+1}^{M}.
$$

* The modeling flow is:

$$
Z_t
\longrightarrow
S_t
\longrightarrow
F_{t+1}=S_t'R_{t+1}
\longrightarrow
R_{t+1}^{M}=\lambda'F_{t+1}
\longrightarrow
M_{t+1}=1-R_{t+1}^{M}.
$$

***

#### 3.4 Random Fourier Factors

* The paper operationalizes the nonlinear basis functions using **Random Fourier Features**.
* For each random draw (\omega\_p):

$$
\omega_p
\sim
N(0,I).
$$

* The nonlinear features are:

$$
[S_{2p-1,t},S_{2p,t}]
=
[
\sin(\gamma Z_t\omega_p),
\cos(\gamma Z_t\omega_p)
].
$$

* Each (\omega\_p) creates a random linear combination of stock characteristics:

$$
Z_t\omega_p.
$$

* The sine and cosine transformations generate nonlinear basis functions:

$$
\sin(\gamma Z_t\omega_p),
\qquad
\cos(\gamma Z_t\omega_p).
$$

* Therefore:

$$
Z_t
\in
\mathbb{R}^{N_t \times D}
\quad
\longrightarrow
\quad
S_t
\in
\mathbb{R}^{N_t \times P}.
$$

* Each column of (S\_t) becomes a characteristic-managed portfolio:

$$
F_{p,t+1}
=
S_{p,t}'R_{t+1}.
$$

* This is a random-feature neural representation:

$$
Z_t
\longrightarrow
\{\sin(\gamma Z_t\omega_p),\cos(\gamma Z_t\omega_p)\}_{p=1}^{P/2}
\longrightarrow
S_t\lambda.
$$

* The hidden-layer weights (\omega\_p) are random and fixed.
* The output-layer weights (\lambda) are estimated.
* The model can therefore be interpreted as a **random-feature neural SDF**:

$$
w_\lambda(Z_t)
=
S_t\lambda.
$$

***

#### 3.5 Ridge SDF Estimation

* The SDF coefficient vector (\lambda) is estimated as the mean-variance efficient combination of the generated factors.
* Without regularization:

$$
\hat{\lambda}
=
\widehat{E}[F_tF_t']^{-1}
\widehat{E}[F_t].
$$

* In large factor models, (P) may exceed (T):

$$
P>T.
$$

* Then:

$$
\widehat{E}[F_tF_t']
\quad
\text{is rank-deficient}.
$$

* The paper uses ridge regularization:

$$
\hat{\lambda}(z)
=
\left(
zI+\widehat{E}[F_tF_t']
\right)^{-1}
\widehat{E}[F_t].
$$

* This solves the ridge Markowitz problem:

$$
\hat{\lambda}(z)
=
\arg\max_{\lambda}
\left\{
\widehat{E}[\lambda'F_t]
-
\frac{1}{2}
\widehat{E}[(\lambda'F_t)^2]
-
z\lambda'\lambda
\right\}.
$$

* The estimated SDF portfolio return is:

$$
\widehat{R}_t^M
=
\hat{\lambda}(z)'F_t.
$$

* The estimated SDF is:

$$
\widehat{M}_t
=
1-\widehat{R}_t^M.
$$

***

#### 3.6 Pricing-Error Interpretation

* A valid SDF should price the factor returns:

$$
E[M_tF_t]
=
0.
$$

* For a candidate SDF:

$$
\widetilde{M}_t
=
1-\lambda'F_t.
$$

* The factor pricing error is:

$$
E[\widetilde{M}_tF_t]
=
E[(1-\lambda'F_t)F_t].
$$

* The Hansen-Jagannathan distance is:

$$
D^{HJ}
=
E[\widetilde{M}_tF_t]'
E[F_tF_t']^{-1}
E[\widetilde{M}_tF_t].
$$

* With ridge regularization, the estimator can be interpreted as minimizing regularized pricing errors:

$$
\hat{\lambda}(z)
=
\arg\min_{\lambda}
\left\{
E[(1-\lambda'F_t)F_t]'
E[F_tF_t']^{-1}
E[(1-\lambda'F_t)F_t]
+
z\lambda'\lambda
\right\}.
$$

* Thus, the estimation target is not return forecasting MSE:

$$
\min E[(R_{t+1}-\hat{R}_{t+1})^2].
$$

* The target is SDF recovery:

$$
\min
\text{pricing error}
\quad
\text{or}
\quad
\max
\text{SDF Sharpe ratio}.
$$

***

#### 3.7 Complexity Ratio

* Model complexity is measured by:

$$
c
=
\frac{P}{T}.
$$

* Classical low-dimensional econometrics corresponds to:

$$
c \approx 0.
$$

* AIPT focuses on the high-complexity regime:

$$
c>0,
\qquad
\text{especially}
\qquad
c\gg 1.
$$

* The empirical model reaches:

$$
P=360{,}000,
\qquad
T=360,
\qquad
c=1000.
$$

* The central empirical prediction is:

$$
c \uparrow
\quad
\Rightarrow
\quad
SR_{OS} \uparrow,
\qquad
D^{HJ}_{OS} \downarrow.
$$

***

#### 3.8 Theoretical Trade-Off

* Increasing (P) improves approximation:

$$
P \uparrow
\quad
\Rightarrow
\quad
\text{specification bias} \downarrow.
$$

* Increasing (P) also makes estimation harder:

$$
\frac{P}{T} \uparrow
\quad
\Rightarrow
\quad
\text{limits to learning} \uparrow.
$$

* The key trade-off is:

$$
\text{approximation gain}
\quad
\text{vs.}
\quad
\text{limits to learning}.
$$

* In high-complexity settings, the estimator does not converge to the true SDF even when the model is correctly specified.
* The paper calls this:

$$
\text{limits to learning}.
$$

* Complexity creates implicit shrinkage:

$$
z
\longrightarrow
Z^*(z;q;c),
$$

with

$$
Z^*(z;q;c)>z.
$$

* Complexity also creates complexity risk:

$$
\text{complexity risk}
=
\text{sampling variation that remains when } P/T>0.
$$

* Complexity is beneficial when approximation gains dominate learning costs:

$$
\text{approximation gain}
>
\text{limits to learning}.
$$

***

#### 3.9 Eigenvalue Condition for the Virtue of Complexity

* Whether complexity helps depends on the eigenvalue distribution of the factor covariance matrix.
* If eigenvalues are concentrated:

$$
\eta_1,\eta_2,\ldots
\text{ decay quickly},
$$

then only a few factors matter:

$$
\text{few dominant factors}.
$$

* This corresponds to the APT view:

$$
\text{APT}
\Rightarrow
\text{low effective rank}.
$$

* In this case:

$$
P \uparrow
\not\Rightarrow
SR_{OS} \uparrow.
$$

* If eigenvalues are dispersed:

$$
\eta_1,\eta_2,\ldots
\text{ decay slowly},
$$

then many factors matter:

$$
\text{many relevant factors}.
$$

* This corresponds to the AIPT view:

$$
\text{AIPT}
\Rightarrow
\text{high effective rank}.
$$

* In this case:

$$
P \uparrow
\Rightarrow
SR_{OS} \uparrow,
\qquad
D_{OS}^{HJ} \downarrow.
$$

***

#### 3.10 APT vs AIPT Calibration

* The paper illustrates the theory with two eigenvalue calibrations.
* AIPT calibration:

$$
\eta_p
=
\frac{1}{p^{0.5}}.
$$

* This implies slowly decaying eigenvalues:

$$
\text{many factors remain relevant}.
$$

* Effective rank is approximately:

$$
\text{Effective Rank}
\approx
500.
$$

* Predicted behavior:

$$
c \uparrow
\Rightarrow
SR_{OS} \uparrow,
\qquad
D_{OS}^{HJ} \downarrow.
$$

* APT calibration:

$$
\eta_p
=
\frac{1}{p^2}.
$$

* This implies rapidly decaying eigenvalues:

$$
\text{only a few factors dominate}.
$$

* Effective rank is approximately:

$$
\text{Effective Rank}
\approx
2.5.
$$

* Predicted behavior:

$$
c \uparrow
\not\Rightarrow
SR_{OS} \uparrow.
$$

* The empirical results match the AIPT calibration more closely than the APT calibration.

***

### 4. Data / Results

#### 4.1 Data

* Monthly U.S. stock returns.
* Stock-level characteristics from Jensen, Kelly, and Pedersen.
* Sample period:

$$
1963
\text{ to }
2023.
$$

* Original number of characteristics:

$$
153.
$$

* Number of retained characteristics:

$$
D=130.
$$

* Stock universe:
  * NYSE.
  * AMEX.
  * NASDAQ.
  * CRSP share codes 10, 11, or 12.
* Filters:
  * Exclude nano stocks.
  * Drop stock-month observations with more than 30% missing characteristic values.
  * Rank-standardize each characteristic cross-sectionally to:

$$
[-0.5,0.5].
$$

* Conditioning matrix:

$$
Z_t
\in
\mathbb{R}^{N_t \times 130}.
$$

* Rolling training window:

$$
T=360
\text{ months}.
$$

* Out-of-sample evaluation starts in:

$$
1993.
$$

***

#### 4.2 Evaluation Metrics

* Out-of-sample SDF Sharpe ratio:

$$
SR_{OS}
=
\sqrt{12}
\frac{
\widehat{E}_{OS}[\widetilde{R}_t^M]
}{
\widehat{\sigma}_{OS}(\widetilde{R}_t^M)
}.
$$

* Out-of-sample Hansen-Jagannathan distance:

$$
\widehat{D}^{HJ}
=
\widehat{E}_{OS}
[
\widetilde{M}_tR_t^T
]'
\widehat{E}_{OS}
[
R_t^TR_t^{T'}
]^+
\widehat{E}_{OS}
[
\widetilde{M}_tR_t^T
].
$$

* Main test assets:
  * JKP anomaly factors.
  * JKP theme factors.
  * Fama-French 25 size-value portfolios.
  * Random Fourier factors themselves.

***

#### 4.3 Main Complexity Result

* The paper varies the factor dimension:

$$
P=36,\ldots,360{,}000.
$$

* The complexity ratio is:

$$
c=\frac{P}{T}.
$$

* The largest model has:

$$
c=1000.
$$

* Main empirical pattern:

$$
P \uparrow
\Rightarrow
SR_{OS} \uparrow.
$$

* Main empirical pattern for pricing errors:

$$
P \uparrow
\Rightarrow
D_{OS}^{HJ} \downarrow.
$$

* The largest complex model achieves:

$$
SR_{OS}^{\text{Complex}}
\approx
3.7.
$$

* Low-complexity models have much lower performance.

***

#### 4.4 Benchmark Comparison

* Benchmark models:

| Model | Description                                       |
| ----- | ------------------------------------------------- |
| FF6   | Fama-French five factors plus momentum            |
| SY    | Stambaugh-Yuan mispricing factors                 |
| HXZ   | Hou-Xue-Zhang q-factor model plus expected growth |
| DHS   | Daniel-Hirshleifer-Sun behavioral factors         |
| BS    | Barillas-Shanken selected factor model            |

* Benchmark Sharpe ratios:

$$
SR_{OS}^{FF6}
\approx
0.8.
$$

$$
SR_{OS}^{HXZ}
\approx
1.2.
$$

* Complex model Sharpe ratio:

$$
SR_{OS}^{\text{Complex}}
\approx
3.7.
$$

* Pricing error reduction relative to FF6:

$$
54.8\%.
$$

* Pricing error reduction relative to HXZ:

$$
50.4\%.
$$

* Pairwise alpha of complex SDF against benchmarks:

$$
\alpha^{\text{Complex vs Benchmark}}
\in
[39.4\%,44.4\%]
\quad
\text{per year}.
$$

* Corresponding t-statistics:

$$
t
\approx
13.
$$

* Reverse benchmark alphas are small and statistically insignificant:

$$
\alpha^{\text{Benchmark vs Complex}}
\approx
0.
$$

* Interpretation:

$$
\text{Complex SDF subsumes benchmark SDFs}.
$$

***

#### 4.5 Nonlinear Benchmark Versions

* The paper checks whether the result is driven only by using 130 characteristics.
* It constructs nonlinear versions of benchmark models using only the characteristics underlying each benchmark.
* For FF6, the raw characteristics are:

$$
\text{size, value, investment, profitability, momentum}.
$$

* Original FF6 performance:

$$
SR_{OS}^{FF6}
\approx
0.74.
$$

* Nonlinear high-complexity FF performance:

$$
SR_{OS}^{\text{Nonlinear FF}}
\in
[1.93,2.50].
$$

* Interpretation:

$$
\text{same small information set}
+
\text{large nonlinear expansion}
\Rightarrow
\text{large performance gain}.
$$

***

#### 4.6 Sparsity Test

* The paper tests whether the large factor model can be compressed using principal components.
* Full complex model:

$$
SR_{OS}^{\text{Full}}
\approx
3.7.
$$

* With (K=5) principal components:

$$
SR_{OS}^{K=5}
\approx
1.3.
$$

* With (K=25) principal components:

$$
SR_{OS}^{K=25}
\approx
2.9.
$$

* Pricing errors increase after dimension reduction:

$$
D_{K=5}^{HJ}
>
D_{\text{Full}}^{HJ}.
$$

$$
D_{K=25}^{HJ}
>
D_{\text{Full}}^{HJ}.
$$

* Interpretation:

$$
\text{Large factor SDF}
\neq
\text{sparse SDF}.
$$

***

#### 4.7 Macro Results

* The paper studies whether the complex SDF predicts future macroeconomic conditions.
* Local projection:

$$
y_{t+h}-y_t
=
a_h
+
b_{SDF,h}
\left(
\sum_{j=0}^{11}R_{t-j}^{M}
\right)
/\sigma_M
+
\sum_{j=0}^{2}b_{j,h}y_{t-j}
+
e_{t+h}.
$$

* Macro variables include:
  * industrial production,
  * macroeconomic uncertainty,
  * federal funds rate,
  * employment,
  * consumer sentiment,
  * CPI,
  * housing starts,
  * Baa-Aaa spread,
  * oil prices,
  * dollar index.
* A positive complex SDF return predicts:

$$
\text{industrial production} \uparrow,
$$

$$
\text{employment} \uparrow,
$$

$$
\text{consumer sentiment} \uparrow,
$$

$$
\text{housing starts} \uparrow,
$$

$$
\text{macroeconomic uncertainty} \downarrow,
$$

$$
\text{credit spreads} \downarrow,
$$

$$
\text{inflation} \downarrow.
$$

* Quantitative results:

$$
1\sigma
\text{ positive complex SDF return}
\Rightarrow
\text{industrial production rises by about }3\%
\text{ over three years}.
$$

$$
1\sigma
\text{ positive complex SDF return}
\Rightarrow
\text{macroeconomic uncertainty falls by about }7.5\%
\text{ over two years}.
$$

***

### 5. Contribution

* The paper proposes **AIPT** as a many-factor alternative to the traditional APT.
* It provides direct empirical evidence that large factor models dominate standard low-dimensional factor models out-of-sample.
* It shows that the benefit of complexity is not merely due to adding more raw characteristics:

$$
Z_t \text{ fixed},
\qquad
P \uparrow,
\qquad
SR_{OS} \uparrow.
$$

* It develops a random-feature SDF construction:

$$
Z_t
\longrightarrow
S_t
\longrightarrow
F_{t+1}=S_t'R_{t+1}
\longrightarrow
M_{t+1}=1-\lambda'F_{t+1}.
$$

* It connects large factor models to the neural SDF literature through a random-feature neural representation.
* It shows that SDF performance improves even in highly overparameterized regimes:

$$
P \gg T.
$$

* It provides random matrix theory to explain when complexity improves asset-pricing performance.
* The key theoretical condition is that the factor covariance eigenvalue distribution must be sufficiently dispersed:

$$
\text{dispersed eigenvalues}
\Rightarrow
\text{many relevant factors}
\Rightarrow
\text{virtue of complexity}.
$$

* The paper argues that the empirical evidence is more consistent with:

$$
\text{AIPT: many factors}
$$

than with:

$$
\text{APT: few factors}.
$$

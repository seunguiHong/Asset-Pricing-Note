# ClarkWest\_2007

## Forecasts and nested model comparisons

**Author(s):** Todd E. Clark, Kenneth D. West\
**Journal / Year:** _Journal of Econometrics_ (2007)

***

### 1. Summary

* The paper proposes a forecast-comparison test for **nested predictive models**.
* In nested comparisons, the larger model contains the benchmark as a special case.
* Under the null, the extra predictors in the larger model have zero population coefficients.
* Even under that null, the larger model still estimates those coefficients in finite samples.
* That estimation adds noise to forecasts and pushes the larger model's raw MSPE upward.
* Raw MSPE comparisons therefore bias results against the larger nested model.
* Clark and West fix this by adding back the squared forecast gap between the benchmark and the larger model.

***

### 2. Background

The paper studies forecast comparisons of the form

$$
\text{parsimonious benchmark model}
\quad \text{vs.} \quad
\text{larger model that nests the benchmark}.
$$

This setup is common in asset pricing and macro forecasting. In asset pricing, the benchmark is often a constant expected return model, while the larger model adds predictors. In macro forecasting, the benchmark may be an autoregression, while the larger model adds lags or indicators.

The issue is simple. Under the null, the larger model does not improve the population forecast. But it still pays a finite-sample cost from estimating unnecessary parameters.

***

### 3. Framework

Let model 1 be the benchmark model:

$$
y_t = X_{1t}'\beta_1^* + e_{1t}.
$$

Let model 2 be the larger model:

$$
y_t = X_{1t}'\delta^* + Z_t'\gamma^* + e_{2t}
= X_{2t}'\beta_2^* + e_{2t}.
$$

Model 2 nests model 1 when

$$
\gamma^* = 0.
$$

Under the null,

$$
\gamma^* = 0,
$$

so

$$
X_{1t}'\beta_1^* = X_{2t}'\beta_2^*,
\qquad
e_{1t} = e_{2t}.
$$

The population null is equal MSPE:

$$
E(e_{1t}^2) - E(e_{2t}^2) = 0.
$$

Let the out-of-sample forecasts be

$$
\hat{y}_{1,t+\tau},
\qquad
\hat{y}_{2,t+\tau},
$$

with forecast errors

$$
\hat{e}_{1,t+\tau} = y_{t+\tau} - \hat{y}_{1,t+\tau},
\qquad
\hat{e}_{2,t+\tau} = y_{t+\tau} - \hat{y}_{2,t+\tau}.
$$

The raw MSPE difference is

$$
\widehat{\sigma}_1^2 - \widehat{\sigma}_2^2
= P^{-1}\sum \hat{e}_{1,t+\tau}^2
- P^{-1}\sum \hat{e}_{2,t+\tau}^2.
$$

The key identity is

$$
\hat{e}_{1,t+1}^2 - \hat{e}_{2,t+1}^2
= 2\hat{e}_{1,t+1}\left(\hat{y}_{1,t+1} - \hat{y}_{2,t+1}\right)
- \left(\hat{y}_{1,t+1} - \hat{y}_{2,t+1}\right)^2.
$$

The last term is negative in the raw MSPE difference:

$$
\left(\hat{y}_{1,t+1} - \hat{y}_{2,t+1}\right)^2.
$$

That term captures the extra estimation noise from the larger model. Clark and West therefore define the adjusted MSPE for model 2 as

$$
\widehat{\sigma}_{2,\mathrm{adj}}^2
= P^{-1}\sum \hat{e}_{2,t+\tau}^2
- P^{-1}\sum \left(\hat{y}_{1,t+\tau} - \hat{y}_{2,t+\tau}\right)^2.
$$

The adjusted MSPE difference is

$$
\widehat{\sigma}_1^2 - \widehat{\sigma}_{2,\mathrm{adj}}^2.
$$

Equivalently, define the Clark-West adjusted loss differential:

$$
\hat{f}_{t+\tau}
= \hat{e}_{1,t+\tau}^2
- \left[\hat{e}_{2,t+\tau}^2 - \left(\hat{y}_{1,t+\tau} - \hat{y}_{2,t+\tau}\right)^2\right].
$$

So

$$
\hat{f}_{t+\tau}
= \hat{e}_{1,t+\tau}^2
- \hat{e}_{2,t+\tau}^2
+ \left(\hat{y}_{1,t+\tau} - \hat{y}_{2,t+\tau}\right)^2.
$$

The sample statistic is the average adjusted loss differential:

$$
\bar{f} = P^{-1}\sum \hat{f}_{t+\tau}.
$$

The test is implemented by regressing $\hat{f}\_{t+\tau}$ on a constant:

$$
\hat{f}_{t+\tau} = \alpha + u_{t+\tau}.
$$

The Clark-West statistic is the t-statistic for $\alpha$:

$$
CW = \frac{\hat{\alpha}}{\widehat{se}(\hat{\alpha})}.
$$

The alternative is one-sided:

$$
H_A: E(\hat{f}_{t+\tau}) > 0.
$$

A positive and significant Clark-West statistic implies that the larger model improves forecast accuracy after correcting for finite-sample estimation noise.

***

### 4. Data and results

The paper evaluates the method with Monte Carlo experiments and empirical applications.

The simulations compare:

* raw MSPE-normal test,
* MSPE-adjusted test,
* CCS test,
* MSPE-adjusted test with simulated critical values,
* MSPE-adjusted test with bootstrap critical values.

The raw MSPE-normal test is severely undersized because it is not centered correctly under the nested null. The MSPE-adjusted statistic performs much better in size.

For nominal 10% one-sided tests, the paper reports the following median empirical sizes:

| Test                                     | Median empirical size |
| ---------------------------------------- | --------------------: |
| MSPE-adjusted                            |                 0.080 |
| MSPE-normal                              |                 0.008 |
| CCS                                      |                 0.107 |
| MSPE-adjusted, simulated critical values |                 0.115 |
| MSPE-adjusted, bootstrap                 |                 0.096 |

The paper also shows that the adjusted test has better size and power than the raw MSPE-normal comparison. The empirical examples cover stock return forecasting and GDP growth forecasting. In both cases, the adjustment changes the interpretation by removing the artificial penalty on the larger model.

***

### 5. Contribution

The main contribution is a practical test for forecast comparisons with nested models.

The paper shows that raw MSPE differences are miscentered when the benchmark is nested inside the larger model. It then proposes the correction

$$
\hat{e}_{1,t+\tau}^2 - \hat{e}_{2,t+\tau}^2
\quad \longrightarrow \quad
\hat{e}_{1,t+\tau}^2 - \hat{e}_{2,t+\tau}^2
+ \left(\hat{y}_{1,t+\tau} - \hat{y}_{2,t+\tau}\right)^2.
$$

This adjustment isolates the larger model's forecast value from its extra estimation noise.

The resulting test is easy to implement with standard forecast errors and forecast paths. For multi-step horizons or serially correlated forecast errors, the standard error should be autocorrelation-consistent. The method became a standard tool for testing whether a larger predictive model improves out-of-sample performance relative to a nested benchmark.

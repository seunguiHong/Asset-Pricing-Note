# DieboldMarion\_1995

## Comparing Predictive Accuracy

**Author(s):** Francis X. Diebold, Robert S. Mariano\
**Journal / Year:** _Journal of Business & Economic Statistics_ (1995)

***

### 1. Summary

* The paper proposes a formal test for comparing the predictive accuracy of two competing forecasts.
* The central object is the loss differential:

$$
d_t = g(e_{it}) - g(e_{jt}).
$$

Here, $$e_{it}$$ and $$e_{jt}$$ are forecast errors from models $$i$$ and $$j$$, and $$g(\cdot)$$ is a loss function. (It is of importance loss function is not limited with sqaured error)

* Equal predictive accuracy is:

$$
H_0: E[d_t] = 0.
$$

* The Diebold-Mariano test asks whether the sample mean loss differential is statistically different from zero, using a long-run variance correction for serial correlation.

### 2. Background

* Forecast comparisons often report average loss measures such as MSPE or MAE, but average losses alone do not provide statistical inference.
* In time-series forecasting, forecast loss differentials are often serially correlated.
* This matters especially for multi-step-ahead forecasts because overlapping horizons induce serial correlation in forecast errors and loss differentials.
* A simple paired $$t$$-test is generally invalid because it ignores autocovariances in the loss-differential process.
* Diebold and Mariano provide a general framework that allows:
  * general loss functions,
  * serial correlation,
  * non-Gaussian forecast errors,
  * multi-step-ahead forecast errors,
  * contemporaneous dependence between forecast errors.

### 3. Framework

Let the forecast errors from two competing forecasts be:

$$
e_{it}, \qquad e_{jt}.
$$

Let $$g(\cdot)$$ be the forecast loss function. Define the loss differential:

$$
d_t = g(e_{it}) - g(e_{jt}).
$$

Equal predictive accuracy means:

$$
H_0: E[g(e_{it})] = E[g(e_{jt})].
$$

Equivalently,

$$
H_0: E[d_t] = 0.
$$

Let

$$
\mu = E[d_t].
$$

The sample mean loss differential is:

$$
\bar d
=
\frac{1}{T}\sum_{t=1}^{T} d_t
=
\frac{1}{T}\sum_{t=1}^{T}\left[g(e_{it}) - g(e_{jt})\right].
$$

Under covariance stationarity and weak dependence of $$d_t$$:

$$
\sqrt{T}(\bar d - \mu) \Rightarrow N(0, 2\pi f_d(0)).
$$

The spectral density at frequency zero is:

$$
f_d(0)
=
\frac{1}{2\pi}\sum_{\tau=-\infty}^{\infty} \gamma_d(\tau).
$$

where

$$
\gamma_d(\tau) = E[(d_t - \mu)(d_{t-\tau} - \mu)].
$$

Therefore:

$$
2\pi f_d(0)
=
\sum_{\tau=-\infty}^{\infty}\gamma_d(\tau)
=
\gamma_d(0)
+
2\sum_{\tau=1}^{\infty}\gamma_d(\tau).
$$

This is the long-run variance of the loss differential.

Under the null $$H_0: \mu = 0$$, the Diebold-Mariano statistic is:

$$
S_1
=
\frac{\bar d}{\sqrt{2\pi \hat f_d(0)/T}}.
$$

Equivalently,

$$
S_1
=
\frac{\bar d}{\sqrt{\hat\Omega/T}}.
$$

where

$$
\hat\Omega = 2\pi \hat f_d(0).
$$

The long-run variance is estimated by a weighted sum of sample autocovariances:

$$
2\pi \hat f_d(0)
=
\sum_{\tau=-(T-1)}^{T-1}
l\left(\frac{\tau}{S(T)}\right)\hat\gamma_d(\tau).
$$

where $$l(\cdot)$$ is a lag window and $$S(T)$$ is the truncation lag.

The sample autocovariance is:

$$
\hat\gamma_d(\tau)
=
\frac{1}{T}\sum_{t=|\tau|+1}^{T}
(d_t - \bar d)(d_{t-|\tau|} - \bar d).
$$

For $$k$$-step-ahead forecasts, the paper discusses the benchmark case of $$(k-1)$$-dependence. Then the long-run variance can be estimated using autocovariances up to lag $$k-1$$:

$$
\hat\Omega
=
\hat\gamma_d(0)
+
2\sum_{\tau=1}^{k-1}\hat\gamma_d(\tau).
$$

With squared-error loss:

$$
g(e_{it}) = e_{it}^2.
$$

So

$$
d_t = e_{it}^2 - e_{jt}^2.
$$

and the null becomes:

$$
H_0: E[e_{it}^2 - e_{jt}^2] = 0.
$$

### 4. Data and results

* The paper is methodological, but it evaluates the proposed test through Monte Carlo simulations and an empirical exchange-rate forecasting application.
* The simulations examine forecast-comparison tests under different conditions:
  * Gaussian forecast errors,
  * fat-tailed forecast errors,
  * different sample sizes,
  * serially correlated loss differentials,
  * quadratic loss,
  * alternative finite-sample tests.
* The empirical application compares exchange-rate forecasts, including forward-rate-based forecasts and random-walk forecasts.
* The results show that inference about forecast accuracy can depend materially on correcting for serial correlation in the loss differential.
* The paper emphasizes that ignoring serial correlation may produce misleading forecast-comparison results.

### 5. Contribution

* The paper establishes the canonical framework for testing equal predictive accuracy between two forecasts.
* It reduces forecast comparison to inference on the mean loss differential:

$$
H_0: E[d_t] = 0.
$$

* It allows general loss functions:

$$
d_t = g(e_{it}) - g(e_{jt}).
$$

* It shows that the correct asymptotic variance of the average loss differential is the long-run variance:

$$
\Omega
=
2\pi f_d(0)
=
\gamma_d(0)
+
2\sum_{\tau=1}^{\infty}\gamma_d(\tau).
$$

* It gives the standard Diebold-Mariano statistic:

$$
DM = \frac{\bar d}{\sqrt{\hat\Omega/T}}.
$$

* It explains why multi-step forecast comparisons require serial-correlation correction.
* The paper becomes the standard baseline for non-nested forecast-accuracy comparison, while later tests such as Clark-West modify this logic for nested model comparisons.

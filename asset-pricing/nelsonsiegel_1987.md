# NelsonSiegel\_1987

## Parsimonious Modeling of Yield Curves

**Author(s):** Charles R. Nelson, Andrew F. Siegel\
**Journal / Year:** _Journal of Business_ (1987)

***

### 1. Summary

* The paper proposes a parsimonious yield-curve model that can generate the empirical shapes commonly observed in the term structure: monotonic, humped, and S-shaped curves.
* The central construction is to model the instantaneous forward rate with a low-dimensional exponential structure and recover the yield curve by averaging forward rates over maturity.
* The model fits U.S. Treasury bill yields well and extrapolates meaningfully enough to predict long-bond prices with high correlation.

***

### 2. Key Idea / Framework

The paper motivates the forward-rate specification from the solution to a second-order differential equation.

Let (r(m)) denote the instantaneous forward rate at maturity (m).\
The authors first consider the case in which (r(m)) is given by the solution to a second-order differential equation with real and unequal roots.\
In that case, the forward rate takes the form

$$
r(m)=\beta_0+\beta_1 \exp(-m/\tau_1)+\beta_2 \exp(-m/\tau_2).
$$

This specification can generate monotonic, humped, and S-shaped forward-rate curves depending on the values of ( \beta\_1 ) and ( \beta\_2 ), and its long-run asymptote is ( \beta\_0 ).

The associated yield to maturity is defined as the average of forward rates:

$$
R(m)=\frac{1}{m}\int_0^m r(x)\,dx.
$$

So the implied yield curve inherits the same broad range of shapes.

The paper then argues that this unequal-roots specification is empirically overparameterized.\
As ( \tau\_1 ) and ( \tau\_2 ) vary, it is possible to find different coefficient values that produce nearly the same fit.\
This motivates moving to a more parsimonious model.

The parsimonious specification comes from the **equal-roots case**.\
In that case, the solution becomes

$$
r(m)=\beta_0+\beta_1 \exp(-m/\tau)+\beta_2 \left(\frac{m}{\tau}\right)\exp(-m/\tau). \tag{1}
$$

This is the Nelson-Siegel forward-rate equation.

The important point is that the extra term

$$
\left(\frac{m}{\tau}\right)\exp(-m/\tau)
$$

comes from the equal-roots solution.\
So the hump-shaped component is not inserted arbitrarily; it is the repeated-root counterpart of the unequal-roots exponential solution.

The three components of the forward curve are therefore

$$
1,\qquad \exp(-m/\tau),\qquad \left(\frac{m}{\tau}\right)\exp(-m/\tau).
$$

These correspond naturally to:

* long-term component,
* short-term component,
* medium-term hump component.

To obtain the yield curve, integrate equation (1) from (0) to (m) and divide by (m).\
This gives

$$
R(m)=\beta_0+(\beta_1+\beta_2)\frac{1-\exp(-m/\tau)}{m/\tau}-\beta_2 \exp(-m/\tau). \tag{2}
$$

This is the yield-curve representation used in the paper.

The endpoint behavior is straightforward.

As (m\to\infty),

$$
r(m)\to\beta_0,\qquad R(m)\to\beta_0.
$$

So ( \beta\_0 ) controls the long-run level.

As (m\to 0),

$$
R(m)\to\beta_0+\beta_1.
$$

So the short end is determined by ( \beta\_0+\beta\_1 ), while ( \beta\_2 ) controls the medium-maturity curvature.

For empirical fitting, the paper rewrites the yield curve as

$$
R(m)=a+b\frac{1-\exp(-m/\tau)}{m/\tau}+c\exp(-m/\tau). \tag{3}
$$

Conditional on ( \tau ), this is linear in coefficients, so (a), (b), and (c) can be estimated by linear least squares.

***

### 3. Data

* The empirical application uses U.S. Treasury bill quote-sheet data sampled every fourth Thursday from January 22, 1981, through October 27, 1983.
* The cross section typically contains around 32 traded maturities, ranging roughly from a few days to about one year, though the first two shortest maturities are excluded because they exhibit abnormally high yields likely related to transaction costs.
* The observed bill quotes are converted into continuously compounded annualized yields, and the model is fitted cross-section by cross-section to the yield-maturity pairs.

***

### 4. Contribution & results

* The paper provides a low-dimensional alternative to spline and polynomial yield-curve fitting and shows that a second-order exponential structure is flexible enough to capture the standard empirical shapes of the term structure.
* Across the Treasury bill samples, the model fits very well: the median (R^2) is about 0.959 and the median residual standard deviation is about 7.25 basis points.
* The hump-generating component materially improves fit relative to a restricted monotonic model, and the fitted curves also extrapolate well enough to produce long-bond price predictions that are highly correlated with actual bond prices.

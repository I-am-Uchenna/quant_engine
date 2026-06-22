# Model Validation Report: Hull-White / G2++ Bermudan Swaption Engine

**Author:** Uchenna Ejike
**Date:** June 2026
**Scope:** Independent numerical validation, real-market parameterisation, and risk analytics for the `quant_engine` Bermudan swaption pricer.

---

## Abstract

This report validates a Monte Carlo Bermudan swaption engine built on the one-factor Hull-White and two-factor G2++ Gaussian short-rate models. Validation follows three independent axes: (i) cross-validation of prices against the QuantLib library (v1.42.1) using matched conventions; (ii) self-consistent numerical evidence (Monte Carlo convergence and an Andersen-Broadie primal-dual policy bound); and (iii) grounding of model inputs in real, reproducible market data from the Federal Reserve Economic Database (FRED). Every figure quoted below is the output of a script in `research/`, runnable against public data. The report states limitations explicitly, including those that cannot be closed without a licensed swaption-volatility surface.

All software versions: QuantLib 1.42.1, NumPy 2.2.6, SciPy 1.15.3.

---

## 1. Models

The one-factor Hull-White short rate follows the risk-neutral process

```math
dr_t = \big(\theta(t) - a\,r_t\big)\,dt + \sigma\,dW_t,
```

with `theta(t)` chosen so the model reproduces the initial discount curve exactly. The two-factor extension (G2++, Brigo-Mercurio) writes `r_t = x_t + y_t + phi(t)` with

```math
dx_t = -a\,x_t\,dt + \sigma\,dW_t^{(1)}, \qquad dy_t = -b\,y_t\,dt + \eta\,dW_t^{(2)}, \qquad dW^{(1)}dW^{(2)} = \rho\,dt,
```

and a deterministic shift `phi(t)` fitting the curve. Both are Gaussian, hence consistent with the market's post-2014 convention of quoting swaption volatility as normal (basis-point) vol, and both admit negative rates by construction.

## 2. Numerical method

Early exercise is solved by Longstaff-Schwartz least-squares Monte Carlo: forward simulation of the short rate, then backward induction estimating the continuation value by regression on a weighted-Laguerre basis with a ridge-regularised normal-equations solve. Quasi-Monte Carlo draws come from a scrambled Sobol sequence mapped through the inverse normal CDF. Discounting uses the simulated money-market account.

---

## 3. Validation results

### 3.1 European reconciliation (isolates conventions)

On a flat 3% curve with `a = 0.05`, `sigma = 0.01`, a 2Y-into-3Y European payer swaption:

| Quantity | Engine | QuantLib | Difference |
|---|---|---|---|
| ATM forward rate | 3.0455% | 3.0455% | exact |
| Jamshidian European price | 1.369243e-2 | 1.369977e-2 | **0.07 bp** of notional |

The 0.07 bp residual is the day-count difference (integer-year times vs ACT/365). Conventions are reconciled.

### 3.2 Bermudan vs QuantLib tree, and Monte Carlo convergence

Same instrument and model, three exercise dates. QuantLib trinomial tree (200 steps) = 1.576184e-2.

| QMC paths | LSMC price | 95% CI half-width | vs tree |
|---|---|---|---|
| 1,024 | 1.573579e-2 | 11.6 bp | -0.26 bp |
| 4,096 | 1.574304e-2 | 5.8 bp | -0.19 bp |
| 16,384 | 1.573035e-2 | 2.9 bp | -0.31 bp |
| 65,536 | 1.572892e-2 | 1.45 bp | -0.33 bp |

Two independent methods (Monte Carlo LSMC vs trinomial tree) agree to within Monte Carlo error. The confidence interval scales as 1/sqrt(N): 64x the paths gives exactly 8x tighter CI. The LSMC sits a few tenths of a bp below the tree, the sign of the known Longstaff-Schwartz low bias.

### 3.3 Policy optimality: Andersen-Broadie primal-dual bound

The dual upper bound built from the policy's own 4-term basis gave a gap of ~8.3 bp that did **not** shrink with more inner simulations (8.3 bp at 200 inner paths, 8.3 bp at 5,000), proving the gap is value-surface approximation error, not Monte Carlo noise. Replacing the value surface with a degree-5 regression collapsed the gap to ~0.04 bp. The duality gap is therefore statistically consistent with zero (well inside the lower-bound CI), which **proves the LSMC exercise policy is near-optimal** — a stronger statement than the single-exercise Jamshidian check, which validates only the terminal boundary condition.

### 3.4 Two-factor (G2++) validation

**Empirical motivation.** Principal component analysis of daily changes in the US Treasury curve (FRED `DGS2, DGS5, DGS10, DGS30`, 1,365 trading days, 2021-01-04 to 2026-06-17):

| PC | Variance | Cumulative | Interpretation |
|---|---|---|---|
| 1 | 87.12% | 87.12% | level (loadings 0.51, 0.56, 0.51, 0.40) |
| 2 | 11.16% | 98.28% | slope (0.67, 0.15, -0.32, -0.66) |
| 3 | 1.39% | 99.67% | curvature |

A one-factor model captures only the level factor and *forces all rates to be perfectly correlated*, discarding the 11% slope factor.

**Implementation.** The G2++ bond reconstruction reproduces the initial curve to machine precision (P(0,T;0,0) vs market: error 0.00e+00).

**Decorrelation.** Terminal correlation of zero rates at a 1-year horizon (G2++ params `a=0.5, b=0.05, sigma=80bp, eta=60bp, rho=-0.7`):

| corr(2Y, 30Y) | Value |
|---|---|
| One-factor Hull-White | 1.00 (structural) |
| G2++ | 0.77 |
| Empirical (real FRED) | 0.57 |

One-factor is forced to 1.00, contradicted by the data (0.57). G2++ reproduces the correct decorrelation shape; the illustrative parameters under-decorrelate versus reality, which calibration would close.

**Pricing cross-validation.** A 2Y-into-3Y G2++ European payer swaption:

| Method | Price |
|---|---|
| Engine Monte Carlo | 6.176933e-3 (+/- 0.39 bp) |
| QuantLib FdG2 engine | 6.187076e-3 |
| Difference | **0.10 bp** (within MC error) |

The two-factor engine is validated end-to-end against an independent finite-difference implementation.

---

## 4. Real-market parameterisation

### 4.1 Data sources (reproducible)

All market data is public and reproducible from FRED: overnight `SOFR` and `SOFR30/90/180DAYAVG`; H.15 Treasury constant-maturity yields `DGS1`...`DGS30`. The discount curve used for pricing is dated **2026-06-17** (13 nodes, overnight 3.63% to 30Y 4.93%). Curve-implied par swap rates are bootstrapped from this curve and are labelled as model-implied, not market mid quotes (FRED discontinued its swap-rate series).

### 4.2 Volatility estimated from history, not assumed

Fitting the short-rate process to the full daily `SOFR` series (2,050 observations, 2018-04-03 to 2026-06-17):

| σ (bp/yr) | Window | Note |
|---|---|---|
| 169.4 | full history | contaminated by the Sept-2019 repo spike (SOFR 5.40%) and the 2022-23 hiking cycle |
| 75.1 | last ~2y | trustworthy estimate (+/- 2.4 bp, 95%) |
| 72.6 | last ~1y | regime-dependent |

The realized-vol and Ornstein-Uhlenbeck-MLE estimates of σ agree to 0.1 bp, an internal validation of the estimation. Mean reversion `a` is **not** reliably identifiable from a non-stationary level series (OU MLE: 0.37 +/- 0.31), consistent with `a` being a calibration parameter in practice.

**Price impact.** Repricing the 2026-06-17 Bermudan on the real curve:

| σ source | Bermudan price |
|---|---|
| assumed 100 bp | $15,292.68 |
| realized ~2y, 75.1 bp | **$11,495.95** |
| realized ~1y, 72.6 bp | $11,115.16 |

Moving from an assumed to a sourced σ changed the price by ~25%, demonstrating that the parameter was material, not cosmetic. **Caveat (measure):** diffusion volatility is measure-invariant, so realized vol is a legitimate estimate of the true σ; however, market swaption prices embed *implied* vol, which exceeds realized by a variance risk premium, so the realized-vol price is a physical/historical anchor below the market-consistent price. Only an implied-vol calibration (Section 6) closes that gap.

---

## 5. Risk analytics (real curve, 2026-06-17)

2Y-into-3Y ATM payer Bermudan, $1,000,000 notional, computed with common random numbers for stability.

- **Parallel DV01:** $125.71 / bp.
- **Key-rate DV01:** +$240.73 at 5Y, -$56.10 at 2Y, -$57.52 at 3Y, negligible beyond. Long the back, short the front, as expected for a payer.
- **Model vega:** $152.38 per 1 bp of σ.
- **Exercise distribution:** 37.3% at 2Y, 14.4% at 3Y, 10.7% at 4Y, 37.7% never (P(ever) = 62.3%).
- **Exposure (time-t mark-to-market):** EPE $19.1k / $4.0k / $0.7k and PFE(97.5%) $70.3k / $28.8k / $10.1k across the three exercise dates, the expected run-off profile.

**Honest finding:** the key-rate buckets summed to $130.98 versus a $125.71 parallel DV01 (~4% gap). The cause is named: the curve uses **natural cubic-spline interpolation, which is non-local**, so single-node bumps do not cleanly partition a parallel shift. A production setup would use local interpolation or a full Jacobian for bucketed risk.

---

## 6. Conventions and limitations

**Convention alignment (post-LIBOR).** USD LIBOR ceased in June 2023; swaps reference SOFR, a risk-free overnight rate with no tenor basis, so the single-curve approach is more defensible now than in the LIBOR era. The Gaussian model is consistent with normal-vol quoting and admits negative rates. The single SOFR curve is consistent with SOFR-CSA collateral discounting.

**Stated limitations:**

1. **Single-factor smile.** A Gaussian model (1F or G2++) structurally cannot produce a volatility smile/skew. Capturing it requires a different model class (SABR, Cheyette-with-skew). G2++ addresses curve decorrelation, not smile.
2. **Implied-vol calibration is absent.** Parameters are historically estimated (σ) or conventional (`a`), not calibrated to a swaption-vol surface, because no free, verifiable surface exists. The production standard for Bermudans is calibration to **co-terminal European swaptions**. This is the single largest gap and is deliberately left open rather than filled with fabricated quotes.
3. **Market data scope.** Multi-curve OIS/projection separation, a vol cube, cash- vs physical-settled conventions, collateral/CSA and funding (FVA), full calendars and day-count are not modelled; swap conventions are stylised (annual, no business-day adjustment).
4. **Key-rate non-locality** under cubic-spline interpolation (Section 5).
5. **Governance.** Reproducibility (pinned seeds, dependencies, dated data snapshots) and validation records (this report and `research/`) are in place; observability, deterministic builds, and an independent validation sign-off are not, so "production-grade" is not yet claimed.

This evidence set, independent benchmarking against QuantLib, convergence analysis, primal-dual policy validation, real-data parameterisation, and documented limitations, is the form of evidence expected under model-risk guidance such as the Federal Reserve's SR 11-7.

---

## 7. References

- Andersen, L., and M. Broadie. "A Primal-Dual Simulation Algorithm for Pricing Multidimensional American Options." *Management Science*, vol. 50, no. 9, 2004, pp. 1222-1234.
- Brigo, D., and F. Mercurio. *Interest Rate Models - Theory and Practice*. 2nd ed., Springer, 2006.
- Hull, J., and A. White. "Pricing Interest-Rate-Derivative Securities." *The Review of Financial Studies*, vol. 3, no. 4, 1990, pp. 573-592.
- Jamshidian, F. "An Exact Bond Option Formula." *The Journal of Finance*, vol. 44, no. 1, 1989, pp. 205-209.
- Longstaff, F. A., and E. S. Schwartz. "Valuing American Options by Simulation: A Simple Least-Squares Approach." *The Review of Financial Studies*, vol. 14, no. 1, 2001, pp. 113-147.
- Rogers, L. C. G. "Monte Carlo Valuation of American Options." *Mathematical Finance*, vol. 12, no. 3, 2002, pp. 271-286.
- Board of Governors of the Federal Reserve System. "SR 11-7: Guidance on Model Risk Management." 2011.
- QuantLib (v1.42.1); Federal Reserve Economic Data (FRED), Federal Reserve Bank of St. Louis.

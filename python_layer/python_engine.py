"""Pure-Python Hull-White Bermudan swaption reference engine.

This module mirrors the public surface of the compiled ``quant_engine_cpp``
extension so the rest of the stack (and the Streamlit dashboard in particular)
can price swaptions on hosts where the C++ core has not been built -- for
example Streamlit Community Cloud, which installs Python dependencies but does
not compile native extensions.

The numerics follow the same model as ``cpp_core``:

* a natural cubic-spline zero curve with flat extrapolation,
* the exact Gaussian Hull-White one-factor transition,
* analytic Hull-White discount-bond reconstruction,
* Longstaff-Schwartz backward induction with a weighted-Laguerre basis and a
  ridge-regularised normal-equations solve,
* a Jamshidian decomposition for the single-exercise European reduction.

Quasi-Monte Carlo draws come from ``scipy.stats.qmc.Sobol`` rather than a custom
direction-number table. Prices therefore match the C++ engine in expectation
(within Monte Carlo error); they are not draw-for-draw identical. When the
compiled extension is available it remains the faster, authoritative path.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.stats import norm, qmc

FloatArray = NDArray[np.float64]

_BASIS_COUNT = 4
_EPSILON = 1.0e-12

#: Identifies this module as the reference (non-compiled) engine. The loader and
#: the dashboard read it to report which backend is active.
IS_REFERENCE_ENGINE = True


class SwaptionType(Enum):
    Payer = "payer"
    Receiver = "receiver"


class SobolScrambling(Enum):
    None_ = "none"
    DigitalShift = "digital_shift"


class YieldCurve:
    """Natural cubic-spline zero curve with flat extrapolation outside the nodes."""

    def __init__(
        self,
        times: Sequence[float] | FloatArray,
        zero_rates: Sequence[float] | FloatArray,
        day_count: object | None = None,
        interpolation: object | None = None,
    ) -> None:
        t = np.ascontiguousarray(np.asarray(times, dtype=np.float64))
        z = np.ascontiguousarray(np.asarray(zero_rates, dtype=np.float64))
        if t.size != z.size:
            raise ValueError("times and zero_rates must have the same length")
        if t.size < 2:
            raise ValueError("at least two curve nodes are required")
        if np.any(np.diff(t) <= 0.0):
            raise ValueError("curve times must be strictly increasing")
        self._times = t
        self._zero_rates = z
        self._spline = CubicSpline(t, z, bc_type="natural")
        self._spline_derivative = self._spline.derivative()

    def zero_rate(self, t: float) -> float:
        if t <= self._times[0]:
            return float(self._zero_rates[0])
        if t >= self._times[-1]:
            return float(self._zero_rates[-1])
        return float(self._spline(t))

    def zero_rate_derivative(self, t: float) -> float:
        if t <= self._times[0] or t >= self._times[-1]:
            return 0.0
        return float(self._spline_derivative(t))

    def instantaneous_forward(self, t: float) -> float:
        clamped = max(0.0, t)
        return self.zero_rate(clamped) + clamped * self.zero_rate_derivative(clamped)

    def discount_factor(self, t: float) -> float:
        if t < 0.0:
            raise ValueError("discount time must be non-negative")
        return math.exp(-self.zero_rate(t) * t)

    @property
    def times(self) -> FloatArray:
        return self._times

    @property
    def zero_rates(self) -> FloatArray:
        return self._zero_rates


class HullWhiteProcess:
    """One-factor Hull-White short-rate process under the risk-neutral measure."""

    def __init__(self, curve: YieldCurve, mean_reversion: float, volatility: float) -> None:
        if mean_reversion <= 0.0:
            raise ValueError("mean reversion must be strictly positive")
        if volatility < 0.0:
            raise ValueError("volatility must be non-negative")
        self.curve = curve
        self.mean_reversion = float(mean_reversion)
        self.volatility = float(volatility)
        self.initial_short_rate = curve.instantaneous_forward(0.0)

    def alpha(self, t: float) -> float:
        a = self.mean_reversion
        one_minus_exp = 1.0 - math.exp(-a * t)
        return self.curve.instantaneous_forward(t) + (
            self.volatility * self.volatility / (2.0 * a * a)
        ) * one_minus_exp * one_minus_exp

    def bond_volatility(self, option_expiry: float, bond_maturity: float) -> float:
        a = self.mean_reversion
        if option_expiry == 0.0 or bond_maturity == option_expiry or self.volatility == 0.0:
            return 0.0
        b = (1.0 - math.exp(-a * (bond_maturity - option_expiry))) / a
        variance_scale = (1.0 - math.exp(-2.0 * a * option_expiry)) / (2.0 * a)
        return self.volatility * b * math.sqrt(max(0.0, variance_scale))

    def discount_bond(self, t: float, maturity: float, short_rate):
        """Analytic Hull-White zero-coupon bond P(t, maturity) given r_t.

        ``short_rate`` may be a scalar or a NumPy array; the return type follows.
        """
        a = self.mean_reversion
        if maturity == t:
            return np.ones_like(short_rate) if isinstance(short_rate, np.ndarray) else 1.0
        b = (1.0 - math.exp(-a * (maturity - t))) / a
        p0_t = self.curve.discount_factor(t)
        p0_maturity = self.curve.discount_factor(maturity)
        convexity = (
            self.volatility * self.volatility / (4.0 * a)
        ) * (1.0 - math.exp(-2.0 * a * t)) * b * b
        scale = (p0_maturity / p0_t) * math.exp(
            b * self.curve.instantaneous_forward(t) - convexity
        )
        return scale * np.exp(-b * np.asarray(short_rate, dtype=np.float64))

    def simulate_short_rate_paths(self, normals: FloatArray, time_grid: FloatArray) -> FloatArray:
        a = self.mean_reversion
        path_count = normals.shape[0]
        time_count = time_grid.size
        rates = np.empty((path_count, time_count), dtype=np.float64)
        rates[:, 0] = self.initial_short_rate
        for step in range(1, time_count):
            dt = float(time_grid[step] - time_grid[step - 1])
            decay = math.exp(-a * dt)
            alpha_s = self.alpha(float(time_grid[step - 1]))
            alpha_t = self.alpha(float(time_grid[step]))
            variance = (self.volatility * self.volatility / (2.0 * a)) * (
                1.0 - math.exp(-2.0 * a * dt)
            )
            stddev = math.sqrt(max(0.0, variance))
            rates[:, step] = (
                rates[:, step - 1] * decay
                + alpha_t
                - alpha_s * decay
                + stddev * normals[:, step - 1]
            )
        return rates

    def integrated_short_rates(self, rates: FloatArray, time_grid: FloatArray) -> FloatArray:
        integrals = np.zeros_like(rates)
        for step in range(1, time_grid.size):
            dt = float(time_grid[step] - time_grid[step - 1])
            integrals[:, step] = integrals[:, step - 1] + 0.5 * (
                rates[:, step - 1] + rates[:, step]
            ) * dt
        return integrals


@dataclass
class BermudanSwaptionSpec:
    notional: float
    fixed_rate: float
    type: SwaptionType
    exercise_times: list[float]
    payment_times: list[float]


@dataclass
class LsmcSimulationConfig:
    path_count: int = 32768
    seed: int = 42
    ridge_lambda: float = 1.0e-10
    sobol_bits: int = 32
    scrambling: SobolScrambling = SobolScrambling.DigitalShift


@dataclass
class PricingResult:
    price: float
    standard_error: float
    path_count: int
    exercise_times: FloatArray
    exercise_boundary: FloatArray
    rate_time_grid: FloatArray
    sample_short_rate_paths: FloatArray


def _weighted_laguerre_basis(short_rate: FloatArray) -> FloatArray:
    """Weighted Laguerre design matrix, shape (n, 4), matching the C++ basis."""
    x = np.maximum(0.0, 20.0 * (short_rate + 0.05))
    weight = np.exp(-0.5 * x)
    x2 = x * x
    x3 = x2 * x
    return np.column_stack(
        (
            weight,
            weight * (1.0 - x),
            weight * (1.0 - 2.0 * x + 0.5 * x2),
            weight * (1.0 - 3.0 * x + 1.5 * x2 - x3 / 6.0),
        )
    )


def _ridge_beta(basis: FloatArray, target: FloatArray, ridge_lambda: float) -> FloatArray:
    gram = basis.T @ basis
    rhs = basis.T @ target
    lam = max(ridge_lambda, 1.0e-14)
    return np.linalg.solve(gram + lam * np.eye(_BASIS_COUNT), rhs)


class LsmcEngine:
    """Longstaff-Schwartz Bermudan swaption engine over the Hull-White process."""

    def __init__(self, process: HullWhiteProcess) -> None:
        self.process = process

    def _validate(self, spec: BermudanSwaptionSpec, config: LsmcSimulationConfig) -> None:
        if spec.notional <= 0.0:
            raise ValueError("swaption notional must be strictly positive")
        if spec.fixed_rate < 0.0:
            raise ValueError("fixed rate must be non-negative")
        if not spec.exercise_times:
            raise ValueError("at least one exercise time is required")
        if not spec.payment_times:
            raise ValueError("at least one payment time is required")
        if spec.payment_times[-1] <= spec.exercise_times[0]:
            raise ValueError("payment schedule must extend beyond first exercise")
        if spec.exercise_times[-1] >= spec.payment_times[-1]:
            raise ValueError("last exercise must be before final payment")
        if config.path_count < 2:
            raise ValueError("at least two Monte Carlo paths are required")
        if config.ridge_lambda < 0.0:
            raise ValueError("ridge lambda must be non-negative")

    def swap_present_value(self, exercise_time: float, short_rate, spec: BermudanSwaptionSpec):
        short_rate = np.asarray(short_rate, dtype=np.float64)
        fixed_leg = np.zeros_like(short_rate)
        last_discount = np.ones_like(short_rate)
        has_future_payment = False
        previous = exercise_time
        for payment_time in spec.payment_times:
            if payment_time <= exercise_time + _EPSILON:
                continue
            accrual = payment_time - previous
            discount = self.process.discount_bond(exercise_time, payment_time, short_rate)
            fixed_leg = fixed_leg + spec.fixed_rate * accrual * discount
            last_discount = discount
            previous = payment_time
            has_future_payment = True
        if not has_future_payment:
            return np.zeros_like(short_rate)
        floating_leg = 1.0 - last_discount
        if spec.type == SwaptionType.Payer:
            unit_value = floating_leg - fixed_leg
        else:
            unit_value = fixed_leg - floating_leg
        return spec.notional * unit_value

    def price(self, spec: BermudanSwaptionSpec, config: LsmcSimulationConfig) -> PricingResult:
        self._validate(spec, config)

        path_count = int(config.path_count)
        exercise_times = [float(t) for t in spec.exercise_times]
        exercise_count = len(exercise_times)
        time_count = exercise_count + 1

        time_grid = np.empty(time_count, dtype=np.float64)
        time_grid[0] = 0.0
        time_grid[1:] = exercise_times

        normals = _sobol_standard_normals(exercise_count, path_count, int(config.seed))
        rates = self.process.simulate_short_rate_paths(normals, time_grid)
        integrals = self.process.integrated_short_rates(rates, time_grid)

        sample_count = min(path_count, 64)
        sample_paths = np.ascontiguousarray(rates[:sample_count].copy())

        exercise_boundary = np.full(exercise_count, np.nan, dtype=np.float64)

        final_col = exercise_count
        final_rates = rates[:, final_col]
        cashflows = np.maximum(
            0.0, self.swap_present_value(float(time_grid[final_col]), final_rates, spec)
        )
        final_itm = cashflows > 0.0
        if np.any(final_itm):
            exercise_boundary[exercise_count - 1] = float(np.mean(final_rates[final_itm]))

        current_col = final_col
        for exercise_index in range(exercise_count - 2, -1, -1):
            target_col = exercise_index + 1
            discount = np.exp(-(integrals[:, current_col] - integrals[:, target_col]))
            cashflows = cashflows * discount

            target_rates = rates[:, target_col]
            intrinsic = np.maximum(
                0.0, self.swap_present_value(float(time_grid[target_col]), target_rates, spec)
            )
            itm = intrinsic > 0.0
            itm_count = int(np.count_nonzero(itm))

            continuation = cashflows.copy()
            if itm_count > _BASIS_COUNT:
                basis = _weighted_laguerre_basis(target_rates[itm])
                beta = _ridge_beta(basis, cashflows[itm], config.ridge_lambda)
                continuation[itm] = np.maximum(0.0, basis @ beta)

            exercise_now = itm & (intrinsic > continuation)
            cashflows = np.where(exercise_now, intrinsic, cashflows)
            if np.any(exercise_now):
                exercise_boundary[exercise_index] = float(np.mean(target_rates[exercise_now]))
            current_col = target_col

        discount_to_zero = np.exp(-integrals[:, current_col])
        path_values = cashflows * discount_to_zero
        price = float(np.mean(path_values))
        if path_count > 1:
            mean_square = float(np.mean(path_values * path_values))
            variance = max(0.0, mean_square - price * price) * path_count / (path_count - 1)
            standard_error = math.sqrt(variance / path_count)
        else:
            standard_error = 0.0

        return PricingResult(
            price=price,
            standard_error=standard_error,
            path_count=path_count,
            exercise_times=np.asarray(exercise_times, dtype=np.float64),
            exercise_boundary=exercise_boundary,
            rate_time_grid=time_grid,
            sample_short_rate_paths=sample_paths,
        )

    def european_swaption_jamshidian(
        self, spec: BermudanSwaptionSpec, option_expiry: float
    ) -> float:
        if option_expiry <= 0.0:
            raise ValueError("option expiry must be positive")

        maturities: list[float] = []
        coupons: list[float] = []
        previous = option_expiry
        for payment_time in spec.payment_times:
            if payment_time <= option_expiry + _EPSILON:
                continue
            accrual = payment_time - previous
            maturities.append(payment_time)
            coupons.append(spec.fixed_rate * accrual)
            previous = payment_time
        if not maturities:
            return 0.0
        coupons[-1] += 1.0

        def portfolio(short_rate: float) -> float:
            return sum(
                coupon * float(self.process.discount_bond(option_expiry, maturity, short_rate))
                for coupon, maturity in zip(coupons, maturities)
            )

        low, high = -0.25, 0.25
        f_low = portfolio(low) - 1.0
        f_high = portfolio(high) - 1.0
        expansion = 0
        while f_low * f_high > 0.0 and expansion < 64:
            low -= 0.25 * (expansion + 1)
            high += 0.25 * (expansion + 1)
            f_low = portfolio(low) - 1.0
            f_high = portfolio(high) - 1.0
            expansion += 1
        if f_low * f_high > 0.0:
            raise RuntimeError("failed to bracket Jamshidian root")
        for _ in range(100):
            mid = 0.5 * (low + high)
            f_mid = portfolio(mid) - 1.0
            if abs(f_mid) < 1.0e-14:
                low = high = mid
                break
            if f_low * f_mid <= 0.0:
                high = mid
                f_high = f_mid
            else:
                low = mid
                f_low = f_mid
        root_rate = 0.5 * (low + high)

        p0_expiry = self.process.curve.discount_factor(option_expiry)
        option_value = 0.0
        for coupon, maturity in zip(coupons, maturities):
            strike = float(self.process.discount_bond(option_expiry, maturity, root_rate))
            p0_maturity = self.process.curve.discount_factor(maturity)
            sigma_p = self.process.bond_volatility(option_expiry, maturity)
            if sigma_p < 1.0e-14:
                forward_bond = p0_maturity / p0_expiry
                if spec.type == SwaptionType.Receiver:
                    bond_option = p0_expiry * max(0.0, forward_bond - strike)
                else:
                    bond_option = p0_expiry * max(0.0, strike - forward_bond)
            else:
                h = math.log(p0_maturity / (strike * p0_expiry)) / sigma_p + 0.5 * sigma_p
                if spec.type == SwaptionType.Receiver:
                    bond_option = p0_maturity * norm.cdf(h) - strike * p0_expiry * norm.cdf(
                        h - sigma_p
                    )
                else:
                    bond_option = strike * p0_expiry * norm.cdf(-h + sigma_p) - p0_maturity * norm.cdf(
                        -h
                    )
            option_value += coupon * bond_option
        return spec.notional * option_value


def _sobol_standard_normals(dimension: int, path_count: int, seed: int) -> FloatArray:
    """Sobol QMC draws mapped to standard normals via the inverse CDF."""
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=seed)
    with warnings.catch_warnings():
        # A non power-of-two sample size only costs balance, not correctness.
        warnings.simplefilter("ignore")
        uniforms = sampler.random(path_count)
    uniforms = np.clip(uniforms, 1.0e-12, 1.0 - 1.0e-12)
    return norm.ppf(uniforms)


def build_exercise_schedule(maturity_years: float, frequency_per_year: int) -> tuple[float, ...]:
    steps = max(1, int(round(maturity_years * float(frequency_per_year))))
    dt = maturity_years / float(steps)
    return tuple(round(dt * float(i), 12) for i in range(1, steps + 1))


def build_payment_schedule(
    maturity_years: float, tenor_years: float, frequency_per_year: int
) -> tuple[float, ...]:
    payments = max(1, int(round(tenor_years * float(frequency_per_year))))
    dt = tenor_years / float(payments)
    return tuple(round(maturity_years + dt * float(i), 12) for i in range(1, payments + 1))

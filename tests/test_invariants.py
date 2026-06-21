from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest


def _import_cpp_module() -> object:
    try:
        import quant_engine_cpp  # type: ignore[import-not-found]

        return quant_engine_cpp
    except ModuleNotFoundError:
        root = Path(__file__).resolve().parents[1]
        for build_dir in (
            root / "build-zig-python-clean",
            root / "build-zig-python",
            root / "build",
        ):
            if any(build_dir.glob("quant_engine_cpp*")):
                sys.path.insert(0, str(build_dir))
                import quant_engine_cpp  # type: ignore[import-not-found, no-redef]

                return quant_engine_cpp
        raise


qe = _import_cpp_module()


def _flat_curve(rate: float = 0.03) -> object:
    times = np.ascontiguousarray(np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=np.float64))
    rates = np.ascontiguousarray(np.full(times.shape, rate, dtype=np.float64))
    return qe.YieldCurve(times, rates)


def _deterministic_process(rate: float = 0.03) -> object:
    return qe.HullWhiteProcess(_flat_curve(rate), 0.05, 0.0)


def _payer_spec(
    exercise_times: list[float],
    fixed_rate: float = 0.025,
    notional: float = 1.0,
) -> object:
    return qe.BermudanSwaptionSpec(
        notional,
        fixed_rate,
        qe.SwaptionType.Payer,
        exercise_times,
        [2.0, 3.0, 4.0],
    )


def test_discounted_zero_coupon_bond_is_risk_neutral_martingale() -> None:
    curve = _flat_curve(0.035)
    process = qe.HullWhiteProcess(curve, 0.08, 0.006)

    horizon = 1.0
    maturity = 3.0
    path_count = 8192
    time_grid = np.ascontiguousarray(np.linspace(0.0, horizon, 25, dtype=np.float64))
    normals = np.empty((path_count, time_grid.size - 1), dtype=np.float64)
    sobol = qe.SobolSequence(
        qe.SobolConfig(
            time_grid.size - 1,
            32,
            12345,
            qe.SobolScrambling.DigitalShift,
        )
    )
    bridge = qe.BrownianBridge(time_grid)
    bridge_values = np.empty((path_count, time_grid.size), dtype=np.float64)
    sobol.fill_standard_normal(1, normals)
    bridge.transform_to_brownian_values(normals, bridge_values)

    transition_normals = np.empty_like(normals)
    for step in range(1, time_grid.size):
        dt = float(time_grid[step] - time_grid[step - 1])
        transition_normals[:, step - 1] = (
            bridge_values[:, step] - bridge_values[:, step - 1]
        ) / math.sqrt(dt)

    rates = np.empty((path_count, time_grid.size), dtype=np.float64)
    integrals = np.empty((path_count, time_grid.size), dtype=np.float64)
    process.simulate_short_rate_paths(transition_normals, time_grid, rates)
    process.integrated_short_rates(rates, time_grid, integrals)

    discounted_bonds = np.empty(path_count, dtype=np.float64)
    final_col = time_grid.size - 1
    for path in range(path_count):
        bond_price = process.discount_bond(horizon, maturity, float(rates[path, final_col]))
        money_market_discount = math.exp(-float(integrals[path, final_col]))
        discounted_bonds[path] = money_market_discount * bond_price

    estimate = float(np.mean(discounted_bonds))
    target = float(curve.discount_factor(maturity))
    assert estimate == pytest.approx(target, abs=2.5e-3)


def test_single_exercise_bermudan_matches_jamshidian_european_price() -> None:
    process = _deterministic_process(0.03)
    engine = qe.LsmcEngine(process)
    spec = _payer_spec([1.0], fixed_rate=0.025, notional=1.0)
    config = qe.LsmcSimulationConfig(
        path_count=64,
        seed=77,
        ridge_lambda=1.0e-12,
        sobol_bits=24,
        scrambling=qe.SobolScrambling.DigitalShift,
    )

    lsmc_price = float(engine.price(spec, config).price)
    jamshidian_price = float(engine.european_swaption_jamshidian(spec, 1.0))

    assert lsmc_price == pytest.approx(jamshidian_price, abs=1.0e-4)


def test_exercise_frequency_monotonicity() -> None:
    process = _deterministic_process(0.03)
    engine = qe.LsmcEngine(process)
    annual = _payer_spec([1.0, 2.0], fixed_rate=0.025, notional=1.0)
    quarterly = _payer_spec(
        [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        fixed_rate=0.025,
        notional=1.0,
    )
    config = qe.LsmcSimulationConfig(
        path_count=64,
        seed=91,
        ridge_lambda=1.0e-12,
        sobol_bits=24,
        scrambling=qe.SobolScrambling.DigitalShift,
    )

    annual_price = float(engine.price(annual, config).price)
    quarterly_price = float(engine.price(quarterly, config).price)

    assert quarterly_price + 1.0e-12 >= annual_price

from __future__ import annotations

import xlwings as xw  # type: ignore[import-untyped]

from quant_engine.python_layer.ray_orchestrator import PricingInput, price_bermudan_swaption


@xw.func
def BermudanSwaption(
    notional: float,
    maturity_years: float,
    tenor_years: float,
    strike: float,
    volatility: float,
    mean_reversion: float = 0.05,
    total_paths: int = 100_000,
    exercise_frequency_per_year: int = 1,
    fixed_leg_frequency_per_year: int = 1,
    payer: bool = True,
) -> float:
    request = PricingInput(
        notional=float(notional),
        maturity_years=float(maturity_years),
        tenor_years=float(tenor_years),
        strike=float(strike),
        volatility=float(volatility),
        mean_reversion=float(mean_reversion),
        total_paths=int(total_paths),
        exercise_frequency_per_year=int(exercise_frequency_per_year),
        fixed_leg_frequency_per_year=int(fixed_leg_frequency_per_year),
        payer=bool(payer),
    )
    result = price_bermudan_swaption(request)
    return float(result.price)

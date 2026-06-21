from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, Sequence, cast

import numpy as np
from numpy.typing import NDArray
import ray  # type: ignore[import-untyped]

from quant_engine.python_layer.data_io import (
    CurveArrays,
    MarketDataManager,
    ensure_cpp_module_path,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PricingInput:
    notional: float = 1_000_000.0
    maturity_years: float = 2.0
    tenor_years: float = 5.0
    strike: float = 0.035
    volatility: float = 0.01
    mean_reversion: float = 0.05
    total_paths: int = 100_000
    exercise_frequency_per_year: int = 1
    fixed_leg_frequency_per_year: int = 1
    payer: bool = True
    seed: int = 42
    ridge_lambda: float = 1.0e-10
    sobol_bits: int = 32


@dataclass(frozen=True)
class ChunkResult:
    price: float
    standard_error: float
    path_count: int
    exercise_times: tuple[float, ...]
    exercise_boundary: tuple[float, ...]


@dataclass(frozen=True)
class DistributedPricingResult:
    price: float
    standard_error: float
    path_count: int
    chunk_count: int
    exercise_times: tuple[float, ...]
    exercise_boundary: tuple[float, ...]


def _positive_int(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_request(request: PricingInput) -> None:
    if request.notional <= 0.0:
        raise ValueError("notional must be positive")
    if request.maturity_years <= 0.0:
        raise ValueError("maturity_years must be positive")
    if request.tenor_years <= 0.0:
        raise ValueError("tenor_years must be positive")
    if request.strike < 0.0:
        raise ValueError("strike must be non-negative")
    if request.volatility < 0.0:
        raise ValueError("volatility must be non-negative")
    if request.mean_reversion <= 0.0:
        raise ValueError("mean_reversion must be positive")
    _positive_int(request.total_paths, "total_paths")
    _positive_int(request.exercise_frequency_per_year, "exercise_frequency_per_year")
    _positive_int(request.fixed_leg_frequency_per_year, "fixed_leg_frequency_per_year")
    if request.sobol_bits <= 0 or request.sobol_bits > 32:
        raise ValueError("sobol_bits must be in [1, 32]")


def build_exercise_schedule(maturity_years: float, frequency_per_year: int) -> tuple[float, ...]:
    steps = max(1, int(round(maturity_years * float(frequency_per_year))))
    dt = maturity_years / float(steps)
    return tuple(round(dt * float(i), 12) for i in range(1, steps + 1))


def build_payment_schedule(
    maturity_years: float,
    tenor_years: float,
    frequency_per_year: int,
) -> tuple[float, ...]:
    payments = max(1, int(round(tenor_years * float(frequency_per_year))))
    dt = tenor_years / float(payments)
    return tuple(round(maturity_years + dt * float(i), 12) for i in range(1, payments + 1))


def chunk_path_count(total_paths: int, cpu_count: int | None = None) -> tuple[int, ...]:
    _positive_int(total_paths, "total_paths")
    available_cpus = max(1, cpu_count if cpu_count is not None else (os.cpu_count() or 1))
    if total_paths < 2:
        raise ValueError("total_paths must be at least 2")
    chunk_count = min(available_cpus, max(1, total_paths // 2))
    base = total_paths // chunk_count
    remainder = total_paths % chunk_count
    return tuple(base + (1 if index < remainder else 0) for index in range(chunk_count))


def _ray_local_mode_from_environment() -> bool:
    value = os.environ.get("QUANT_ENGINE_RAY_LOCAL_MODE", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def ensure_ray(
    num_cpus: int | None = None,
    local_mode: bool | None = None,
) -> bool:
    del local_mode
    if ray.is_initialized():
        return False
    ray.init(
        num_cpus=num_cpus,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    return True


def _weighted_boundary(results: Sequence[ChunkResult]) -> tuple[float, ...]:
    if not results:
        return tuple()
    boundary_count = len(results[0].exercise_boundary)
    weighted: list[float] = []
    for index in range(boundary_count):
        numerator = 0.0
        denominator = 0
        for result in results:
            value = result.exercise_boundary[index]
            if math.isfinite(value):
                numerator += value * float(result.path_count)
                denominator += result.path_count
        weighted.append(float("nan") if denominator == 0 else numerator / float(denominator))
    return tuple(weighted)


def aggregate_chunk_results(results: Sequence[ChunkResult]) -> DistributedPricingResult:
    if not results:
        raise ValueError("at least one chunk result is required")
    total_paths = sum(result.path_count for result in results)
    mean = sum(result.price * float(result.path_count) for result in results) / float(total_paths)
    if total_paths > 1:
        sum_squares = 0.0
        for result in results:
            sample_variance = result.standard_error * result.standard_error * float(result.path_count)
            sum_squares += float(result.path_count - 1) * sample_variance
            sum_squares += float(result.path_count) * (result.price - mean) * (result.price - mean)
        variance = max(0.0, sum_squares / float(total_paths - 1))
        standard_error = math.sqrt(variance / float(total_paths))
    else:
        standard_error = 0.0
    return DistributedPricingResult(
        price=mean,
        standard_error=standard_error,
        path_count=total_paths,
        chunk_count=len(results),
        exercise_times=results[0].exercise_times,
        exercise_boundary=_weighted_boundary(results),
    )


def _array_to_tuple(values: Iterable[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _run_pricing_chunk(
    request: PricingInput,
    curve_times: FloatArray,
    zero_rates: FloatArray,
    path_count: int,
    seed: int,
) -> ChunkResult:
    ensure_cpp_module_path()
    import quant_engine_cpp  # type: ignore[import-not-found]

    curve = quant_engine_cpp.YieldCurve(
        np.ascontiguousarray(curve_times, dtype=np.float64),
        np.ascontiguousarray(zero_rates, dtype=np.float64),
    )
    process = quant_engine_cpp.HullWhiteProcess(
        curve,
        float(request.mean_reversion),
        float(request.volatility),
    )
    engine = quant_engine_cpp.LsmcEngine(process)
    swaption_type = (
        quant_engine_cpp.SwaptionType.Payer
        if request.payer
        else quant_engine_cpp.SwaptionType.Receiver
    )
    exercise_times = build_exercise_schedule(
        request.maturity_years,
        request.exercise_frequency_per_year,
    )
    payment_times = build_payment_schedule(
        request.maturity_years,
        request.tenor_years,
        request.fixed_leg_frequency_per_year,
    )
    spec = quant_engine_cpp.BermudanSwaptionSpec(
        float(request.notional),
        float(request.strike),
        swaption_type,
        list(exercise_times),
        list(payment_times),
    )
    config = quant_engine_cpp.LsmcSimulationConfig(
        path_count=int(path_count),
        seed=int(seed),
        ridge_lambda=float(request.ridge_lambda),
        sobol_bits=int(request.sobol_bits),
        scrambling=quant_engine_cpp.SobolScrambling.DigitalShift,
    )
    result = engine.price(spec, config)
    return ChunkResult(
        price=float(result.price),
        standard_error=float(result.standard_error),
        path_count=int(result.path_count),
        exercise_times=_array_to_tuple(result.exercise_times),
        exercise_boundary=_array_to_tuple(result.exercise_boundary),
    )


@ray.remote  # type: ignore[misc]
def _ray_price_chunk(
    request: PricingInput,
    curve_times: FloatArray,
    zero_rates: FloatArray,
    path_count: int,
    seed: int,
) -> ChunkResult:
    return _run_pricing_chunk(request, curve_times, zero_rates, path_count, seed)


def price_bermudan_swaption(
    request: PricingInput,
    curve_arrays: CurveArrays | None = None,
    cpu_count: int | None = None,
    local_mode: bool | None = None,
) -> DistributedPricingResult:
    _validate_request(request)
    arrays = curve_arrays
    if arrays is None:
        manager = MarketDataManager()
        try:
            manager.ensure_market_data()
            arrays = manager.get_curve_arrays()
        finally:
            manager.close()

    chunks = chunk_path_count(request.total_paths, cpu_count)
    ensure_ray(num_cpus=len(chunks), local_mode=local_mode)
    refs = [
        _ray_price_chunk.remote(
            request,
            arrays.times,
            arrays.zero_rates,
            int(path_count),
            int(request.seed + 104_729 * index),
        )
        for index, path_count in enumerate(chunks)
    ]
    chunk_results = cast(list[ChunkResult], ray.get(refs))
    return aggregate_chunk_results(chunk_results)


def price_locally_without_ray(
    request: PricingInput,
    curve_arrays: CurveArrays | None = None,
) -> DistributedPricingResult:
    _validate_request(request)
    arrays = curve_arrays
    if arrays is None:
        manager = MarketDataManager()
        try:
            manager.ensure_market_data()
            arrays = manager.get_curve_arrays()
        finally:
            manager.close()
    result = _run_pricing_chunk(
        request,
        arrays.times,
        arrays.zero_rates,
        request.total_paths,
        request.seed,
    )
    return aggregate_chunk_results([result])

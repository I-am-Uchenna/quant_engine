from __future__ import annotations

import math
import sys
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import duckdb  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CurveArrays:
    times: FloatArray
    zero_rates: FloatArray


@dataclass(frozen=True)
class MarketDataSummary:
    as_of: date
    curve_nodes: int
    swap_quotes: int
    parquet_directory: Path | None
    source: str


@dataclass(frozen=True)
class FredSeriesSpec:
    series_id: str
    tenor_years: float
    field_name: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_env_file(env_path: str | Path | None = None) -> None:
    path = project_root() / ".env" if env_path is None else Path(env_path)
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def fred_api_key() -> str:
    load_env_file()
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "FRED_API_KEY is required. Put it in quant_engine/.env or set the environment variable."
        )
    if len(key) != 32 or not key.isalnum() or key.lower() != key:
        raise RuntimeError(
            "FRED_API_KEY must be the 32-character lower-case key value only. "
            "Do not paste a FRED account URL, Markdown link, placeholder, or TOML syntax into the value."
        )
    return key


def ensure_cpp_module_path(root: Path | None = None) -> Path:
    base = project_root() if root is None else root
    direct_candidates = [
        base / "build-zig-python-clean",
        base / "build-zig-python",
        base / "build",
    ]
    for candidate in direct_candidates:
        if any(candidate.glob("quant_engine_cpp*.pyd")):
            candidate_text = str(candidate)
            if candidate_text not in sys.path:
                sys.path.insert(0, candidate_text)
            return candidate

    matches = sorted(base.glob("build*/quant_engine_cpp*.pyd"))
    if not matches:
        raise ModuleNotFoundError(
            "quant_engine_cpp was not found. Build Phase 2 before loading Python infrastructure."
        )
    module_dir = matches[0].parent
    module_dir_text = str(module_dir)
    if module_dir_text not in sys.path:
        sys.path.insert(0, module_dir_text)
    return module_dir


def load_cpp_engine() -> ModuleType:
    ensure_cpp_module_path()
    import quant_engine_cpp  # type: ignore[import-not-found]

    return quant_engine_cpp


def _as_float_array(values: list[float]) -> FloatArray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64))


SOFR_SERIES: tuple[FredSeriesSpec, ...] = (
    FredSeriesSpec("SOFR", 1.0 / 360.0, "overnight_sofr"),
    FredSeriesSpec("SOFR30DAYAVG", 30.0 / 365.0, "sofr_30_day_average"),
    FredSeriesSpec("SOFR90DAYAVG", 90.0 / 365.0, "sofr_90_day_average"),
    FredSeriesSpec("SOFR180DAYAVG", 180.0 / 365.0, "sofr_180_day_average"),
)


TREASURY_SERIES: tuple[FredSeriesSpec, ...] = (
    FredSeriesSpec("DGS1", 1.0, "treasury_1y"),
    FredSeriesSpec("DGS2", 2.0, "treasury_2y"),
    FredSeriesSpec("DGS3", 3.0, "treasury_3y"),
    FredSeriesSpec("DGS5", 5.0, "treasury_5y"),
    FredSeriesSpec("DGS7", 7.0, "treasury_7y"),
    FredSeriesSpec("DGS10", 10.0, "treasury_10y"),
    FredSeriesSpec("DGS20", 20.0, "treasury_20y"),
    FredSeriesSpec("DGS30", 30.0, "treasury_30y"),
)


def _fred_csv_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def _fred_api_url(series_id: str, api_key: str, limit: int = 250) -> str:
    query = urlencode(
        {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": str(limit),
        }
    )
    return f"https://api.stlouisfed.org/fred/series/observations?{query}"


def _parse_fred_value(raw_value: str) -> float | None:
    value = raw_value.strip()
    if value in {"", "."}:
        return None
    return float(value) / 100.0


def _download_fred_series_from_api(series_id: str, api_key: str) -> dict[date, float]:
    request = Request(
        _fred_api_url(series_id, api_key),
        headers={"User-Agent": "quant-engine/0.1"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        message = body
        try:
            parsed_error = json.loads(body)
            message = str(parsed_error.get("error_message", body))
        except json.JSONDecodeError:
            pass
        raise RuntimeError(f"FRED API rejected {series_id}: {message}") from exc
    parsed = json.loads(payload)
    observations: dict[date, float] = {}
    for row in parsed.get("observations", []):
        raw_date = str(row.get("date", "")).strip()
        raw_value = str(row.get("value", "")).strip()
        if not raw_date:
            continue
        value = _parse_fred_value(raw_value)
        if value is not None:
            observations[date.fromisoformat(raw_date)] = value
    if not observations:
        raise ValueError(f"FRED API series {series_id} has no usable observations")
    return observations


def _download_fred_series(series_id: str, api_key: str | None = None) -> dict[date, float]:
    credential = api_key or fred_api_key()
    return _download_fred_series_from_api(series_id, credential)


def _latest_aligned_observation(
    specs: tuple[FredSeriesSpec, ...],
    api_key: str | None = None,
) -> tuple[date, dict[str, float]]:
    downloaded = {
        spec.series_id: _download_fred_series(spec.series_id, api_key) for spec in specs
    }
    anchor_date = min(max(observations) for observations in downloaded.values())
    aligned: dict[str, float] = {}
    for series_id, observations in downloaded.items():
        eligible_dates = [obs_date for obs_date in observations if obs_date <= anchor_date]
        if not eligible_dates:
            raise ValueError(f"FRED series {series_id} has no observation by {anchor_date}")
        aligned[series_id] = observations[max(eligible_dates)]
    return anchor_date, aligned


def _interpolated_zero_rate(nodes: list[tuple[float, float]], tenor: float) -> float:
    if tenor <= nodes[0][0]:
        return nodes[0][1]
    if tenor >= nodes[-1][0]:
        return nodes[-1][1]
    for left, right in zip(nodes[:-1], nodes[1:]):
        if left[0] <= tenor <= right[0]:
            weight = (tenor - left[0]) / (right[0] - left[0])
            return left[1] + weight * (right[1] - left[1])
    return nodes[-1][1]


def _discount_from_nodes(nodes: list[tuple[float, float]], tenor: float) -> float:
    zero_rate = _interpolated_zero_rate(nodes, tenor)
    return math.exp(-zero_rate * tenor)


def _curve_implied_par_swap_rate(
    nodes: list[tuple[float, float]],
    maturity_years: float,
    tenor_years: float,
    fixed_frequency_per_year: int = 1,
) -> float:
    payments = max(1, int(round(tenor_years * float(fixed_frequency_per_year))))
    accrual = tenor_years / float(payments)
    start_discount = _discount_from_nodes(nodes, maturity_years)
    final_discount = _discount_from_nodes(nodes, maturity_years + tenor_years)
    annuity = 0.0
    for payment_index in range(1, payments + 1):
        payment_time = maturity_years + accrual * float(payment_index)
        annuity += accrual * _discount_from_nodes(nodes, payment_time)
    if annuity <= 0.0:
        raise ValueError("invalid swap annuity from downloaded curve")
    return (start_discount - final_discount) / annuity


class MarketDataManager:
    def __init__(self, database_path: str | Path | None = None) -> None:
        self._database_path = ":memory:" if database_path is None else str(database_path)
        self._connection: Any = duckdb.connect(self._database_path)

    def close(self) -> None:
        self._connection.close()

    def download_public_market_data(
        self,
        parquet_directory: str | Path | None = None,
        fred_api_key: str | None = None,
    ) -> MarketDataSummary:
        specs = SOFR_SERIES + TREASURY_SERIES
        valuation_date, observations = _latest_aligned_observation(specs, fred_api_key)
        curve_nodes = [(0.0, observations["SOFR"])]
        curve_nodes.extend(
            (spec.tenor_years, observations[spec.series_id]) for spec in SOFR_SERIES
        )
        curve_nodes.extend(
            (spec.tenor_years, observations[spec.series_id]) for spec in TREASURY_SERIES
        )
        curve_nodes = sorted({round(tenor, 12): rate for tenor, rate in curve_nodes}.items())

        curve_rows = [
            (
                valuation_date.isoformat(),
                float(tenor),
                float(rate),
                "FRED:FRBNY_SOFR_AVERAGES_AND_H15_TREASURY",
            )
            for tenor, rate in curve_nodes
        ]

        swap_rows: list[tuple[str, float, float, float, str]] = []
        for maturity in (1.0, 2.0, 3.0, 5.0, 7.0, 10.0):
            for tenor in (1.0, 2.0, 5.0, 10.0):
                par_rate = _curve_implied_par_swap_rate(curve_nodes, maturity, tenor)
                swap_rows.append(
                    (
                        valuation_date.isoformat(),
                        maturity,
                        tenor,
                        par_rate,
                        "CURVE_IMPLIED_FROM_FRED_PUBLIC_RATES",
                    )
                )

        self._connection.execute("DROP TABLE IF EXISTS sofr_curve")
        self._connection.execute("DROP TABLE IF EXISTS swap_rates")
        self._connection.execute("DROP TABLE IF EXISTS market_sources")
        self._connection.execute(
            """
            CREATE TABLE sofr_curve (
                as_of DATE,
                tenor_years DOUBLE,
                zero_rate DOUBLE,
                source VARCHAR
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE swap_rates (
                as_of DATE,
                maturity_years DOUBLE,
                tenor_years DOUBLE,
                par_rate DOUBLE,
                source VARCHAR
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE market_sources (
                source_name VARCHAR,
                url VARCHAR,
                note VARCHAR
            )
            """
        )
        self._connection.executemany("INSERT INTO sofr_curve VALUES (?, ?, ?, ?)", curve_rows)
        self._connection.executemany("INSERT INTO swap_rates VALUES (?, ?, ?, ?, ?)", swap_rows)
        self._connection.executemany(
            "INSERT INTO market_sources VALUES (?, ?, ?)",
            [
                (
                    "FRED:SOFR",
                    _fred_csv_url("SOFR"),
                    "FRBNY Secured Overnight Financing Rate via FRED.",
                ),
                (
                    "FRED:SOFR_AVERAGES",
                    _fred_csv_url("SOFR30DAYAVG"),
                    "FRBNY 30/90/180 day SOFR averages via FRED.",
                ),
                (
                    "FRED:H15_TREASURY",
                    _fred_csv_url("DGS10"),
                    "Federal Reserve H.15 Treasury constant maturity rates via FRED.",
                ),
                (
                    "CURVE_IMPLIED_SWAP_RATES",
                    "internal",
                    "Forward-start par swap rates derived from the downloaded public curve.",
                ),
            ],
        )

        parquet_path: Path | None = None
        if parquet_directory is not None:
            parquet_path = Path(parquet_directory)
            parquet_path.mkdir(parents=True, exist_ok=True)
            self._connection.execute(
                "COPY sofr_curve TO ? (FORMAT PARQUET)",
                [str(parquet_path / "sofr_curve.parquet")],
            )
            self._connection.execute(
                "COPY swap_rates TO ? (FORMAT PARQUET)",
                [str(parquet_path / "swap_rates.parquet")],
            )

        return MarketDataSummary(
            as_of=valuation_date,
            curve_nodes=len(curve_rows),
            swap_quotes=len(swap_rows),
            parquet_directory=parquet_path,
            source="FRED public rates; swap rates curve-implied",
        )

    def load_market_data(
        self,
        sofr_curve_path: str | Path,
        swap_rates_path: str | Path | None = None,
    ) -> None:
        curve_source = str(sofr_curve_path)
        self._connection.execute("DROP TABLE IF EXISTS sofr_curve")
        self._connection.execute(
            """
            CREATE TABLE sofr_curve AS
            SELECT CAST(as_of AS DATE) AS as_of,
                   CAST(tenor_years AS DOUBLE) AS tenor_years,
                   CAST(zero_rate AS DOUBLE) AS zero_rate
            FROM read_parquet(?)
            """,
            [curve_source],
        )

        if swap_rates_path is not None:
            swap_source = str(swap_rates_path)
            self._connection.execute("DROP TABLE IF EXISTS swap_rates")
            self._connection.execute(
                """
                CREATE TABLE swap_rates AS
                SELECT CAST(as_of AS DATE) AS as_of,
                       CAST(maturity_years AS DOUBLE) AS maturity_years,
                       CAST(tenor_years AS DOUBLE) AS tenor_years,
                       CAST(par_rate AS DOUBLE) AS par_rate
                FROM read_parquet(?)
                """,
                [swap_source],
            )

    def ensure_market_data(self) -> MarketDataSummary:
        tables = {
            row[0]
            for row in self._connection.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
        }
        if "sofr_curve" not in tables or "swap_rates" not in tables:
            return self.download_public_market_data()

        curve_count = int(
            self._connection.execute("SELECT COUNT(*) FROM sofr_curve").fetchone()[0]
        )
        swap_count = int(
            self._connection.execute("SELECT COUNT(*) FROM swap_rates").fetchone()[0]
        )
        if curve_count == 0 or swap_count == 0:
            return self.download_public_market_data()

        latest = self._connection.execute("SELECT MAX(as_of) FROM sofr_curve").fetchone()[0]
        return MarketDataSummary(
            as_of=latest,
            curve_nodes=curve_count,
            swap_quotes=swap_count,
            parquet_directory=None,
            source="existing_duckdb_tables",
        )

    def get_curve_arrays(self, as_of: date | None = None) -> CurveArrays:
        self.ensure_market_data()
        if as_of is None:
            rows = self._connection.execute(
                """
                SELECT tenor_years, zero_rate
                FROM sofr_curve
                WHERE as_of = (SELECT MAX(as_of) FROM sofr_curve)
                ORDER BY tenor_years
                """
            ).fetchall()
        else:
            rows = self._connection.execute(
                """
                SELECT tenor_years, zero_rate
                FROM sofr_curve
                WHERE as_of = ?
                ORDER BY tenor_years
                """,
                [as_of.isoformat()],
            ).fetchall()

        if len(rows) < 2:
            raise ValueError("at least two SOFR curve rows are required")

        times = [float(row[0]) for row in rows]
        rates = [float(row[1]) for row in rows]
        if times[0] > 0.0:
            times.insert(0, 0.0)
            rates.insert(0, rates[0])
        return CurveArrays(times=_as_float_array(times), zero_rates=_as_float_array(rates))

    def latest_par_swap_rate(
        self,
        maturity_years: float,
        tenor_years: float,
        as_of: date | None = None,
    ) -> float:
        self.ensure_market_data()
        params: list[float | str] = [maturity_years, tenor_years]
        date_filter = "as_of = (SELECT MAX(as_of) FROM swap_rates)"
        if as_of is not None:
            date_filter = "as_of = ?"
            params.insert(0, as_of.isoformat())

        row = self._connection.execute(
            f"""
            SELECT par_rate
            FROM swap_rates
            WHERE {date_filter}
            ORDER BY ABS(maturity_years - ?) + ABS(tenor_years - ?)
            LIMIT 1
            """,
            params,
        ).fetchone()
        if row is None:
            raise ValueError("no swap rates are available")
        return float(row[0])

    def build_yield_curve(self, as_of: date | None = None) -> Any:
        engine = load_cpp_engine()
        arrays = self.get_curve_arrays(as_of)
        return engine.YieldCurve(arrays.times, arrays.zero_rates)


def default_curve_arrays() -> CurveArrays:
    manager = MarketDataManager()
    try:
        manager.ensure_market_data()
        return manager.get_curve_arrays()
    finally:
        manager.close()

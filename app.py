from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from quant_engine.python_layer.data_io import MarketDataManager, load_cpp_engine
from quant_engine.python_layer.ray_orchestrator import (
    build_exercise_schedule,
    build_payment_schedule,
)


def _load_streamlit_secrets() -> None:
    if os.environ.get("FRED_API_KEY"):
        return
    try:
        fred_key = st.secrets.get("FRED_API_KEY", "")
    except Exception:
        fred_key = ""
    if fred_key:
        os.environ["FRED_API_KEY"] = str(fred_key)


def _valid_fred_key(value: str) -> bool:
    key = value.strip()
    return len(key) == 32 and key.isalnum() and key.lower() == key


def _configure_fred_key_if_missing() -> None:
    if os.environ.get("FRED_API_KEY"):
        return

    with st.container(border=True):
        st.subheader("FRED Market Data Key")
        st.write(
            "Enter the raw 32-character FRED API key. The app stores it in the local "
            "ignored `.env` file for this runtime and never writes it to Git."
        )
        key = st.text_input("FRED_API_KEY", type="password", placeholder="32 lower-case letters/numbers")
        save = st.button("Save Key", type="primary")

    if not save:
        st.stop()

    cleaned = key.strip()
    if not _valid_fred_key(cleaned):
        st.error("The key must be exactly 32 lower-case letters/numbers.")
        st.stop()

    env_path = ROOT / ".env"
    env_path.write_text(f"FRED_API_KEY={cleaned}\n", encoding="utf-8")
    try:
        env_path.chmod(0o600)
    except OSError:
        pass
    os.environ["FRED_API_KEY"] = cleaned
    st.cache_data.clear()
    st.rerun()


def _format_rate(value: float) -> str:
    return f"{100.0 * value:.3f}%"


@st.cache_resource(show_spinner=False)
def _cpp_engine() -> Any:
    return load_cpp_engine()


@st.cache_data(show_spinner=False, ttl=900)
def _market_snapshot() -> tuple[np.ndarray, np.ndarray, str, str, float]:
    manager = MarketDataManager()
    try:
        summary = manager.ensure_market_data()
        arrays = manager.get_curve_arrays()
        par_rate = manager.latest_par_swap_rate(2.0, 5.0)
        return (
            arrays.times.copy(),
            arrays.zero_rates.copy(),
            summary.as_of.isoformat(),
            summary.source,
            par_rate,
        )
    finally:
        manager.close()


def _price_native(
    notional: float,
    maturity: float,
    tenor: float,
    strike: float,
    mean_reversion: float,
    volatility: float,
    path_count: int,
    exercise_frequency: int,
    fixed_leg_frequency: int,
    payer: bool,
    seed: int,
) -> tuple[Any, tuple[float, ...], tuple[float, ...]]:
    qe = _cpp_engine()
    curve_times, zero_rates, _, _, _ = _market_snapshot()
    curve = qe.YieldCurve(
        np.ascontiguousarray(curve_times, dtype=np.float64),
        np.ascontiguousarray(zero_rates, dtype=np.float64),
    )
    process = qe.HullWhiteProcess(curve, mean_reversion, volatility)
    engine = qe.LsmcEngine(process)
    exercise_times = build_exercise_schedule(maturity, exercise_frequency)
    payment_times = build_payment_schedule(maturity, tenor, fixed_leg_frequency)
    swaption_type = qe.SwaptionType.Payer if payer else qe.SwaptionType.Receiver
    spec = qe.BermudanSwaptionSpec(
        notional,
        strike,
        swaption_type,
        list(exercise_times),
        list(payment_times),
    )
    config = qe.LsmcSimulationConfig(
        path_count=path_count,
        seed=seed,
        ridge_lambda=1.0e-10,
        sobol_bits=32,
        scrambling=qe.SobolScrambling.DigitalShift,
    )
    return engine.price(spec, config), exercise_times, payment_times


def _path_figure(result: Any) -> go.Figure:
    time_grid = np.asarray(result.rate_time_grid, dtype=float)
    paths = np.asarray(result.sample_short_rate_paths, dtype=float)
    exercise_times = np.asarray(result.exercise_times, dtype=float)
    boundary = np.asarray(result.exercise_boundary, dtype=float)

    fig = go.Figure()
    if paths.ndim == 2 and paths.size > 0:
        for row in range(min(paths.shape[0], 32)):
            fig.add_trace(
                go.Scatter(
                    x=time_grid,
                    y=paths[row],
                    mode="lines",
                    line={"color": "rgba(51, 97, 140, 0.18)", "width": 1},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    valid = np.isfinite(boundary)
    if np.any(valid):
        fig.add_trace(
            go.Scatter(
                x=exercise_times[valid],
                y=boundary[valid],
                mode="lines+markers",
                name="Early exercise boundary",
                line={"color": "#C8501D", "width": 3},
                marker={"size": 8, "color": "#C8501D"},
            )
        )
    fig.update_layout(
        height=430,
        margin={"l": 36, "r": 20, "t": 24, "b": 36},
        xaxis_title="Time (years)",
        yaxis_title="Short rate",
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    fig.update_yaxes(tickformat=".2%")
    return fig


def main() -> None:
    st.set_page_config(
        page_title="Bermudan Swaption Pricer",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _load_streamlit_secrets()

    st.title("Bermudan Swaption Pricing")
    _configure_fred_key_if_missing()

    with st.sidebar:
        st.header("Instrument")
        notional = st.number_input("Notional", min_value=1_000.0, value=1_000_000.0, step=100_000.0)
        maturity = st.slider("Option maturity", 0.5, 10.0, 2.0, 0.5)
        tenor = st.slider("Underlying swap tenor", 1.0, 20.0, 5.0, 0.5)
        strike = st.number_input("Strike rate", min_value=0.0, value=0.04, step=0.001, format="%.4f")
        payer = st.segmented_control("Swaption type", ["Payer", "Receiver"], default="Payer") == "Payer"

        st.header("Hull-White")
        mean_reversion = st.slider("Mean reversion a", 0.001, 0.30, 0.05, 0.001, format="%.3f")
        volatility = st.slider("Volatility sigma", 0.0, 0.05, 0.01, 0.001, format="%.3f")

        st.header("Simulation")
        exercise_label = st.selectbox("Exercise frequency", ["Annual", "Semiannual", "Quarterly"], index=2)
        exercise_frequency = {"Annual": 1, "Semiannual": 2, "Quarterly": 4}[exercise_label]
        fixed_leg_frequency = st.selectbox("Fixed leg payments per year", [1, 2, 4], index=0)
        path_count = int(st.select_slider("QMC paths", options=[512, 1024, 2048, 4096, 8192], value=2048))
        seed = st.number_input("Sobol digital shift seed", min_value=1, value=42, step=1)
        run = st.button("Execute Pricing Run", type="primary", use_container_width=True)

    try:
        curve_times, zero_rates, as_of, source, par_rate = _market_snapshot()
    except Exception as exc:
        st.error(f"Market data load failed: {exc}")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Market date", as_of)
    c2.metric("Curve nodes", f"{len(curve_times)}")
    c3.metric("2Y x 5Y curve-implied par rate", _format_rate(float(par_rate)))
    st.caption(source)

    with st.expander("Quantitative formulation", expanded=False):
        st.latex(
            r"""
            dr_t = \left(\theta(t)-a r_t\right)dt + \sigma dW_t
            """
        )
        st.latex(
            r"""
            \mathbb{E}[r_t|\mathcal{F}_s]
            = r_s e^{-a(t-s)} + \alpha(t) - \alpha(s)e^{-a(t-s)}
            """
        )
        st.latex(
            r"""
            \operatorname{Var}(r_t|\mathcal{F}_s)
            = \frac{\sigma^2}{2a}\left(1-e^{-2a(t-s)}\right)
            """
        )
        st.latex(
            r"""
            C(r_t) \approx \sum_{k=0}^{3}\beta_k e^{-x/2}L_k(x),
            \qquad
            \beta=(X^\top X+\lambda I)^{-1}X^\top Y
            """
        )

    if not run:
        st.info("Set parameters in the sidebar and run the native pricing engine.")
        st.stop()

    with st.spinner("Running Hull-White QMC LSMC pricing..."):
        result, exercise_times, payment_times = _price_native(
            notional=float(notional),
            maturity=float(maturity),
            tenor=float(tenor),
            strike=float(strike),
            mean_reversion=float(mean_reversion),
            volatility=float(volatility),
            path_count=path_count,
            exercise_frequency=exercise_frequency,
            fixed_leg_frequency=int(fixed_leg_frequency),
            payer=payer,
            seed=int(seed),
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Option price", f"${float(result.price):,.2f}")
    m2.metric("Monte Carlo standard error", f"${float(result.standard_error):,.2f}")
    m3.metric("Exercise dates", f"{len(exercise_times)}")
    m4.metric("Payment dates", f"{len(payment_times)}")

    st.plotly_chart(_path_figure(result), use_container_width=True)

    with st.container(border=True):
        st.subheader("Run Summary")
        left, right = st.columns(2)
        left.write(
            {
                "notional": float(notional),
                "maturity_years": float(maturity),
                "tenor_years": float(tenor),
                "strike": float(strike),
                "type": "payer" if payer else "receiver",
            }
        )
        right.write(
            {
                "mean_reversion": float(mean_reversion),
                "volatility": float(volatility),
                "paths": path_count,
                "seed": int(seed),
            }
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import asyncio
import importlib
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, AsyncIterator

import grpc  # type: ignore[import-untyped]
from grpc_tools import protoc  # type: ignore[import-untyped]

from quant_engine.python_layer.ray_orchestrator import PricingInput, price_bermudan_swaption


PROTO_SOURCE = """
syntax = "proto3";

package quant_engine;

message PricingRequest {
  double notional = 1;
  double maturity_years = 2;
  double tenor_years = 3;
  double strike = 4;
  double volatility = 5;
  double mean_reversion = 6;
  uint64 total_paths = 7;
  uint32 exercise_frequency_per_year = 8;
  uint32 fixed_leg_frequency_per_year = 9;
  bool payer = 10;
  uint64 seed = 11;
  double ridge_lambda = 12;
  uint32 sobol_bits = 13;
}

message RiskMetric {
  string name = 1;
  double value = 2;
  string unit = 3;
}

service PricerService {
  rpc PriceBermudanSwaption(PricingRequest) returns (stream RiskMetric);
}
"""


def _generated_directory() -> Path:
    return Path(tempfile.gettempdir()) / "quant_engine_grpc_generated"


def compile_proto() -> tuple[ModuleType, ModuleType]:
    output_dir = _generated_directory()
    output_dir.mkdir(parents=True, exist_ok=True)
    proto_path = output_dir / "pricer.proto"
    proto_path.write_text(PROTO_SOURCE, encoding="utf-8")

    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{output_dir}",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            str(proto_path),
        ]
    )
    if result != 0:
        raise RuntimeError("grpc_tools.protoc failed to compile pricer.proto")

    output_text = str(output_dir)
    if output_text not in sys.path:
        sys.path.insert(0, output_text)
    pb2 = importlib.import_module("pricer_pb2")
    pb2_grpc = importlib.import_module("pricer_pb2_grpc")
    return pb2, pb2_grpc


_PB2, _PB2_GRPC = compile_proto()


def _request_to_input(request: Any) -> PricingInput:
    defaults = PricingInput()
    return PricingInput(
        notional=float(request.notional or defaults.notional),
        maturity_years=float(request.maturity_years or defaults.maturity_years),
        tenor_years=float(request.tenor_years or defaults.tenor_years),
        strike=float(request.strike),
        volatility=float(request.volatility),
        mean_reversion=float(request.mean_reversion or defaults.mean_reversion),
        total_paths=int(request.total_paths or defaults.total_paths),
        exercise_frequency_per_year=max(
            1,
            int(request.exercise_frequency_per_year or defaults.exercise_frequency_per_year),
        ),
        fixed_leg_frequency_per_year=max(
            1,
            int(request.fixed_leg_frequency_per_year or defaults.fixed_leg_frequency_per_year),
        ),
        payer=bool(request.payer),
        seed=int(request.seed or defaults.seed),
        ridge_lambda=float(request.ridge_lambda or defaults.ridge_lambda),
        sobol_bits=max(1, int(request.sobol_bits or defaults.sobol_bits)),
    )


def _local_mode_from_environment() -> bool:
    value = os.environ.get("QUANT_ENGINE_RAY_LOCAL_MODE", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


class PricerService(_PB2_GRPC.PricerServiceServicer):  # type: ignore[name-defined, misc, valid-type]
    async def PriceBermudanSwaption(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[Any]:
        del context
        pricing_input = _request_to_input(request)
        result = await asyncio.to_thread(
            price_bermudan_swaption,
            pricing_input,
            None,
            None,
            _local_mode_from_environment(),
        )
        yield _PB2.RiskMetric(name="price", value=result.price, unit="currency")
        yield _PB2.RiskMetric(
            name="standard_error",
            value=result.standard_error,
            unit="currency",
        )
        yield _PB2.RiskMetric(name="path_count", value=float(result.path_count), unit="paths")
        yield _PB2.RiskMetric(name="chunk_count", value=float(result.chunk_count), unit="chunks")
        for index, boundary in enumerate(result.exercise_boundary):
            if boundary == boundary:
                yield _PB2.RiskMetric(
                    name=f"exercise_boundary_{index + 1}",
                    value=float(boundary),
                    unit="short_rate",
                )


async def create_server(host: str = "127.0.0.1", port: int = 50051) -> tuple[Any, int]:
    server = grpc.aio.server()
    _PB2_GRPC.add_PricerServiceServicer_to_server(PricerService(), server)
    bound_port = int(server.add_insecure_port(f"{host}:{port}"))
    if bound_port == 0:
        raise RuntimeError(f"failed to bind gRPC server to {host}:{port}")
    await server.start()
    return server, bound_port


async def serve(host: str = "127.0.0.1", port: int = 50051) -> None:
    server, _ = await create_server(host, port)
    await server.wait_for_termination()


def make_pricing_request(input_data: PricingInput) -> Any:
    return _PB2.PricingRequest(
        notional=input_data.notional,
        maturity_years=input_data.maturity_years,
        tenor_years=input_data.tenor_years,
        strike=input_data.strike,
        volatility=input_data.volatility,
        mean_reversion=input_data.mean_reversion,
        total_paths=input_data.total_paths,
        exercise_frequency_per_year=input_data.exercise_frequency_per_year,
        fixed_leg_frequency_per_year=input_data.fixed_leg_frequency_per_year,
        payer=input_data.payer,
        seed=input_data.seed,
        ridge_lambda=input_data.ridge_lambda,
        sobol_bits=input_data.sobol_bits,
    )


async def price_once(target: str, input_data: PricingInput) -> dict[str, float]:
    async with grpc.aio.insecure_channel(target) as channel:
        stub = _PB2_GRPC.PricerServiceStub(channel)
        metrics: dict[str, float] = {}
        async for metric in stub.PriceBermudanSwaption(make_pricing_request(input_data)):
            metrics[str(metric.name)] = float(metric.value)
        return metrics


if __name__ == "__main__":
    asyncio.run(serve())

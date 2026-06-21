#include "HullWhiteProcess.hpp"
#include "LsmcEngine.hpp"
#include "QuasiMonteCarlo.hpp"
#include "YieldCurve.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace quant_engine {
namespace {

py::buffer_info checkedArray(py::array_t<double, py::array::c_style>& array,
                             const char* name,
                             int expected_ndim) {
  py::buffer_info info = array.request();
  if (info.ndim != expected_ndim) {
    throw std::invalid_argument(std::string(name) + " has incorrect rank");
  }
  return info;
}

py::buffer_info checkedConstArray(const py::array_t<double, py::array::c_style>& array,
                                  const char* name,
                                  int expected_ndim) {
  py::buffer_info info = array.request();
  if (info.ndim != expected_ndim) {
    throw std::invalid_argument(std::string(name) + " has incorrect rank");
  }
  return info;
}

py::array_t<double> vectorView(std::vector<double>& values, py::object owner) {
  return py::array_t<double>({values.size()}, {sizeof(double)}, values.data(), owner);
}

py::array_t<double> matrixView(std::vector<double>& values,
                               std::size_t rows,
                               std::size_t cols,
                               py::object owner) {
  return py::array_t<double>({rows, cols}, {cols * sizeof(double), sizeof(double)},
                             values.data(), owner);
}

}  // namespace
}  // namespace quant_engine

PYBIND11_MODULE(quant_engine_cpp, module) {
  using namespace quant_engine;

  module.doc() = "C++17 Hull-White Bermudan swaption pricing core";

  py::enum_<DayCountBasis>(module, "DayCountBasis")
      .value("Act365Fixed", DayCountBasis::Act365Fixed)
      .value("Act360", DayCountBasis::Act360);

  py::enum_<InterpolationKind>(module, "InterpolationKind")
      .value("NaturalCubicSpline", InterpolationKind::NaturalCubicSpline);

  py::enum_<SobolScrambling>(module, "SobolScrambling")
      .value("None_", SobolScrambling::None)
      .value("DigitalShift", SobolScrambling::DigitalShift);

  py::enum_<NormalTransform>(module, "NormalTransform")
      .value("AcklamInverseCdf", NormalTransform::AcklamInverseCdf);

  py::enum_<SwaptionType>(module, "SwaptionType")
      .value("Payer", SwaptionType::Payer)
      .value("Receiver", SwaptionType::Receiver);

  py::class_<TransitionMoments>(module, "TransitionMoments")
      .def_readonly("mean", &TransitionMoments::mean)
      .def_readonly("variance", &TransitionMoments::variance);

  py::class_<SobolConfig>(module, "SobolConfig")
      .def(py::init([](std::size_t dimension,
                       std::uint32_t bits,
                       std::uint64_t seed,
                       SobolScrambling scrambling) {
             return SobolConfig{dimension, bits, seed, scrambling};
           }),
           py::arg("dimension"),
           py::arg("bits") = 32U,
           py::arg("seed") = 0U,
           py::arg("scrambling") = SobolScrambling::DigitalShift)
      .def_readwrite("dimension", &SobolConfig::dimension)
      .def_readwrite("bits", &SobolConfig::bits)
      .def_readwrite("seed", &SobolConfig::seed)
      .def_readwrite("scrambling", &SobolConfig::scrambling);

  py::class_<YieldCurve>(module, "YieldCurve")
      .def(py::init([](const py::array_t<double, py::array::c_style>& times,
                       const py::array_t<double, py::array::c_style>& zero_rates,
                       DayCountBasis day_count,
                       InterpolationKind interpolation) {
             py::buffer_info time_info = checkedConstArray(times, "times", 1);
             py::buffer_info rate_info = checkedConstArray(zero_rates, "zero_rates", 1);
             if (time_info.shape[0] != rate_info.shape[0]) {
               throw std::invalid_argument("times and zero_rates must have the same length");
             }
             return YieldCurve(static_cast<const double*>(time_info.ptr),
                               static_cast<const double*>(rate_info.ptr),
                               static_cast<std::size_t>(time_info.shape[0]), day_count,
                               interpolation);
           }),
           py::arg("times"),
           py::arg("zero_rates"),
           py::arg("day_count") = DayCountBasis::Act365Fixed,
           py::arg("interpolation") = InterpolationKind::NaturalCubicSpline)
      .def("size", &YieldCurve::size)
      .def("empty", &YieldCurve::empty)
      .def("zero_rate", &YieldCurve::zeroRate, py::arg("t"))
      .def("zero_rate_derivative", &YieldCurve::zeroRateDerivative, py::arg("t"))
      .def("instantaneous_forward", &YieldCurve::instantaneousForward, py::arg("t"))
      .def("discount_factor", &YieldCurve::discountFactor, py::arg("t"))
      .def("continuously_compounded_forward", &YieldCurve::continuouslyCompoundedForward,
           py::arg("start"), py::arg("end"))
      .def_property_readonly("times", [](YieldCurve& self) {
        return vectorView(const_cast<std::vector<double>&>(self.times()), py::cast(&self));
      })
      .def_property_readonly("zero_rates", [](YieldCurve& self) {
        return vectorView(const_cast<std::vector<double>&>(self.zeroRates()), py::cast(&self));
      });

  py::class_<HullWhiteProcess>(module, "HullWhiteProcess")
      .def(py::init<YieldCurve, double, double>(), py::arg("curve"),
           py::arg("mean_reversion"), py::arg("volatility"))
      .def("factors", &HullWhiteProcess::factors)
      .def("initial_short_rate", &HullWhiteProcess::initialShortRate)
      .def("mean_reversion", &HullWhiteProcess::meanReversion)
      .def("volatility", &HullWhiteProcess::volatility)
      .def("alpha", &HullWhiteProcess::alpha, py::arg("t"))
      .def("bond_volatility", &HullWhiteProcess::bondVolatility,
           py::arg("option_expiry"), py::arg("bond_maturity"))
      .def("discount_bond", &HullWhiteProcess::discountBond,
           py::arg("t"), py::arg("maturity"), py::arg("short_rate"))
      .def("transition_moments", &HullWhiteProcess::transitionMoments,
           py::arg("s"), py::arg("t"), py::arg("short_rate"))
      .def("simulate_short_rate_paths",
           [](const HullWhiteProcess& self,
              const py::array_t<double, py::array::c_style>& standard_normals,
              const py::array_t<double, py::array::c_style>& time_grid,
              py::array_t<double, py::array::c_style>& output_rates) {
             py::buffer_info normal_info =
                 checkedConstArray(standard_normals, "standard_normals", 2);
             py::buffer_info time_info = checkedConstArray(time_grid, "time_grid", 1);
             py::buffer_info output_info = checkedArray(output_rates, "output_rates", 2);
             const auto path_count = static_cast<std::size_t>(output_info.shape[0]);
             const auto time_count = static_cast<std::size_t>(time_info.shape[0]);
             if (output_info.shape[1] != time_info.shape[0]) {
               throw std::invalid_argument("output_rates must have shape paths x time_count");
             }
             if (time_count > 1U &&
                 (normal_info.shape[0] != output_info.shape[0] ||
                  normal_info.shape[1] != time_info.shape[0] - 1)) {
               throw std::invalid_argument(
                   "standard_normals must have shape paths x (time_count - 1)");
             }
             self.simulateShortRatePaths(static_cast<const double*>(normal_info.ptr), path_count,
                                         static_cast<const double*>(time_info.ptr), time_count,
                                         static_cast<double*>(output_info.ptr));
             return output_rates;
           },
           py::arg("standard_normals"), py::arg("time_grid"), py::arg("output_rates"))
      .def("integrated_short_rates",
           [](const HullWhiteProcess& self,
              const py::array_t<double, py::array::c_style>& short_rate_paths,
              const py::array_t<double, py::array::c_style>& time_grid,
              py::array_t<double, py::array::c_style>& output_integrals) {
             py::buffer_info rate_info =
                 checkedConstArray(short_rate_paths, "short_rate_paths", 2);
             py::buffer_info time_info = checkedConstArray(time_grid, "time_grid", 1);
             py::buffer_info output_info =
                 checkedArray(output_integrals, "output_integrals", 2);
             if (rate_info.shape[0] != output_info.shape[0] ||
                 rate_info.shape[1] != output_info.shape[1] ||
                 rate_info.shape[1] != time_info.shape[0]) {
               throw std::invalid_argument(
                   "rate paths and output integrals must have shape paths x time_count");
             }
             self.integratedShortRates(static_cast<const double*>(rate_info.ptr),
                                       static_cast<std::size_t>(rate_info.shape[0]),
                                       static_cast<const double*>(time_info.ptr),
                                       static_cast<std::size_t>(time_info.shape[0]),
                                       static_cast<double*>(output_info.ptr));
             return output_integrals;
           },
           py::arg("short_rate_paths"), py::arg("time_grid"), py::arg("output_integrals"));

  py::class_<SobolSequence>(module, "SobolSequence")
      .def(py::init<SobolConfig>(), py::arg("config"))
      .def("dimension", &SobolSequence::dimension)
      .def("bits", &SobolSequence::bits)
      .def("scrambling", &SobolSequence::scrambling)
      .def("fill_uniform",
           [](const SobolSequence& self,
              std::uint64_t first_index,
              py::array_t<double, py::array::c_style>& output) {
             py::buffer_info output_info = checkedArray(output, "output", 2);
             if (static_cast<std::size_t>(output_info.shape[1]) != self.dimension()) {
               throw std::invalid_argument("output second dimension must equal Sobol dimension");
             }
             self.fillUniform(first_index, static_cast<std::size_t>(output_info.shape[0]),
                              static_cast<double*>(output_info.ptr));
             return output;
           },
           py::arg("first_index"), py::arg("output"))
      .def("fill_standard_normal",
           [](const SobolSequence& self,
              std::uint64_t first_index,
              py::array_t<double, py::array::c_style>& output) {
             py::buffer_info output_info = checkedArray(output, "output", 2);
             if (static_cast<std::size_t>(output_info.shape[1]) != self.dimension()) {
               throw std::invalid_argument("output second dimension must equal Sobol dimension");
             }
             self.fillStandardNormal(first_index, static_cast<std::size_t>(output_info.shape[0]),
                                     static_cast<double*>(output_info.ptr));
             return output;
           },
           py::arg("first_index"), py::arg("output"));

  py::class_<BrownianBridge>(module, "BrownianBridge")
      .def(py::init([](const py::array_t<double, py::array::c_style>& time_grid) {
             py::buffer_info info = checkedConstArray(time_grid, "time_grid", 1);
             return BrownianBridge(static_cast<const double*>(info.ptr),
                                   static_cast<std::size_t>(info.shape[0]));
           }),
           py::arg("time_grid"))
      .def("time_count", &BrownianBridge::timeCount)
      .def("gaussian_dimension", &BrownianBridge::gaussianDimension)
      .def("transform_to_brownian_values",
           [](const BrownianBridge& self,
              const py::array_t<double, py::array::c_style>& independent_normals,
              py::array_t<double, py::array::c_style>& output) {
             py::buffer_info normal_info =
                 checkedConstArray(independent_normals, "independent_normals", 2);
             py::buffer_info output_info = checkedArray(output, "output", 2);
             if (static_cast<std::size_t>(normal_info.shape[1]) != self.gaussianDimension() ||
                 output_info.shape[0] != normal_info.shape[0] ||
                 static_cast<std::size_t>(output_info.shape[1]) != self.timeCount()) {
               throw std::invalid_argument(
                   "Brownian bridge arrays must be paths x dimension and paths x time_count");
             }
             self.transformToBrownianValues(static_cast<const double*>(normal_info.ptr),
                                            static_cast<std::size_t>(normal_info.shape[0]),
                                            static_cast<double*>(output_info.ptr));
             return output;
           },
           py::arg("independent_normals"), py::arg("output"))
      .def("transform_to_brownian_increments",
           [](const BrownianBridge& self,
              const py::array_t<double, py::array::c_style>& independent_normals,
              py::array_t<double, py::array::c_style>& output) {
             py::buffer_info normal_info =
                 checkedConstArray(independent_normals, "independent_normals", 2);
             py::buffer_info output_info = checkedArray(output, "output", 2);
             if (static_cast<std::size_t>(normal_info.shape[1]) != self.gaussianDimension() ||
                 output_info.shape[0] != normal_info.shape[0] ||
                 static_cast<std::size_t>(output_info.shape[1]) != self.gaussianDimension()) {
               throw std::invalid_argument(
                   "Brownian increment arrays must be paths x bridge_dimension");
             }
             self.transformToBrownianIncrements(static_cast<const double*>(normal_info.ptr),
                                                static_cast<std::size_t>(normal_info.shape[0]),
                                                static_cast<double*>(output_info.ptr));
             return output;
           },
           py::arg("independent_normals"), py::arg("output"));

  py::class_<BermudanSwaptionSpec>(module, "BermudanSwaptionSpec")
      .def(py::init([](double notional,
                       double fixed_rate,
                       SwaptionType type,
                       std::vector<double> exercise_times,
                       std::vector<double> payment_times) {
             return BermudanSwaptionSpec{notional, fixed_rate, type, std::move(exercise_times),
                                         std::move(payment_times)};
           }),
           py::arg("notional"), py::arg("fixed_rate"), py::arg("type"),
           py::arg("exercise_times"), py::arg("payment_times"))
      .def_readwrite("notional", &BermudanSwaptionSpec::notional)
      .def_readwrite("fixed_rate", &BermudanSwaptionSpec::fixed_rate)
      .def_readwrite("type", &BermudanSwaptionSpec::type)
      .def_readwrite("exercise_times", &BermudanSwaptionSpec::exercise_times)
      .def_readwrite("payment_times", &BermudanSwaptionSpec::payment_times);

  py::class_<LsmcSimulationConfig>(module, "LsmcSimulationConfig")
      .def(py::init([](std::size_t path_count,
                       std::uint64_t seed,
                       double ridge_lambda,
                       std::uint32_t sobol_bits,
                       SobolScrambling scrambling) {
             return LsmcSimulationConfig{path_count, seed, ridge_lambda, sobol_bits,
                                         scrambling};
           }),
           py::arg("path_count") = 32768U, py::arg("seed") = 42U,
           py::arg("ridge_lambda") = 1.0e-10, py::arg("sobol_bits") = 32U,
           py::arg("scrambling") = SobolScrambling::DigitalShift)
      .def_readwrite("path_count", &LsmcSimulationConfig::path_count)
      .def_readwrite("seed", &LsmcSimulationConfig::seed)
      .def_readwrite("ridge_lambda", &LsmcSimulationConfig::ridge_lambda)
      .def_readwrite("sobol_bits", &LsmcSimulationConfig::sobol_bits)
      .def_readwrite("scrambling", &LsmcSimulationConfig::scrambling);

  py::class_<PricingResult>(module, "PricingResult")
      .def_readonly("price", &PricingResult::price)
      .def_readonly("standard_error", &PricingResult::standard_error)
      .def_readonly("path_count", &PricingResult::path_count)
      .def_property_readonly("exercise_times", [](PricingResult& self) {
        return vectorView(self.exercise_times, py::cast(&self));
      })
      .def_property_readonly("exercise_boundary", [](PricingResult& self) {
        return vectorView(self.exercise_boundary, py::cast(&self));
      })
      .def_property_readonly("rate_time_grid", [](PricingResult& self) {
        return vectorView(self.rate_time_grid, py::cast(&self));
      })
      .def_property_readonly("sample_short_rate_paths", [](PricingResult& self) {
        const std::size_t cols = self.rate_time_grid.size();
        const std::size_t rows = cols == 0U ? 0U : self.sample_short_rate_paths.size() / cols;
        return matrixView(self.sample_short_rate_paths, rows, cols, py::cast(&self));
      });

  py::class_<LsmcEngine>(module, "LsmcEngine")
      .def(py::init<HullWhiteProcess>(), py::arg("process"))
      .def("price", &LsmcEngine::price, py::arg("spec"), py::arg("config"))
      .def("european_swaption_jamshidian", &LsmcEngine::europeanSwaptionJamshidian,
           py::arg("spec"), py::arg("option_expiry"))
      .def("swap_present_value", &LsmcEngine::swapPresentValue,
           py::arg("exercise_time"), py::arg("short_rate"), py::arg("spec"));
}

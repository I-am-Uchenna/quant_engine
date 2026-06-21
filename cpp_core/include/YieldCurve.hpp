#pragma once

#include <cstddef>
#include <vector>

namespace quant_engine {

enum class DayCountBasis {
  Act365Fixed,
  Act360
};

enum class InterpolationKind {
  NaturalCubicSpline
};

struct CurveNode {
  double time;
  double zero_rate;
};

class YieldCurve final {
 public:
  YieldCurve(std::vector<double> times,
             std::vector<double> zero_rates,
             DayCountBasis day_count = DayCountBasis::Act365Fixed,
             InterpolationKind interpolation = InterpolationKind::NaturalCubicSpline);

  YieldCurve(const double* times,
             const double* zero_rates,
             std::size_t count,
             DayCountBasis day_count = DayCountBasis::Act365Fixed,
             InterpolationKind interpolation = InterpolationKind::NaturalCubicSpline);

  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] bool empty() const noexcept;
  [[nodiscard]] DayCountBasis dayCountBasis() const noexcept;
  [[nodiscard]] InterpolationKind interpolationKind() const noexcept;

  [[nodiscard]] double zeroRate(double t) const;
  [[nodiscard]] double zeroRateDerivative(double t) const;
  [[nodiscard]] double instantaneousForward(double t) const;
  [[nodiscard]] double discountFactor(double t) const;
  [[nodiscard]] double continuouslyCompoundedForward(double start, double end) const;

  [[nodiscard]] const std::vector<double>& times() const noexcept;
  [[nodiscard]] const std::vector<double>& zeroRates() const noexcept;

 private:
  void validateInputs() const;
  void buildNaturalCubicSpline();
  [[nodiscard]] std::size_t locateInterval(double t) const;
  [[nodiscard]] double evaluateSpline(double t) const;
  [[nodiscard]] double evaluateSplineDerivative(double t) const;

  std::vector<double> times_;
  std::vector<double> zero_rates_;
  std::vector<double> spline_b_;
  std::vector<double> spline_c_;
  std::vector<double> spline_d_;
  DayCountBasis day_count_;
  InterpolationKind interpolation_;
};

}  // namespace quant_engine

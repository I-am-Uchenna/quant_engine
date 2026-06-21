#include "YieldCurve.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace quant_engine {
namespace {

void requireFinite(double value, const char* name) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

}  // namespace

YieldCurve::YieldCurve(std::vector<double> times,
                       std::vector<double> zero_rates,
                       DayCountBasis day_count,
                       InterpolationKind interpolation)
    : times_(std::move(times)),
      zero_rates_(std::move(zero_rates)),
      day_count_(day_count),
      interpolation_(interpolation) {
  validateInputs();
  buildNaturalCubicSpline();
}

YieldCurve::YieldCurve(const double* times,
                       const double* zero_rates,
                       std::size_t count,
                       DayCountBasis day_count,
                       InterpolationKind interpolation)
    : times_(count),
      zero_rates_(count),
      day_count_(day_count),
      interpolation_(interpolation) {
  if (count > 0 && (times == nullptr || zero_rates == nullptr)) {
    throw std::invalid_argument("curve input pointers must not be null");
  }
  for (std::size_t i = 0; i < count; ++i) {
    times_[i] = times[i];
    zero_rates_[i] = zero_rates[i];
  }
  validateInputs();
  buildNaturalCubicSpline();
}

std::size_t YieldCurve::size() const noexcept {
  return times_.size();
}

bool YieldCurve::empty() const noexcept {
  return times_.empty();
}

DayCountBasis YieldCurve::dayCountBasis() const noexcept {
  return day_count_;
}

InterpolationKind YieldCurve::interpolationKind() const noexcept {
  return interpolation_;
}

double YieldCurve::zeroRate(double t) const {
  requireFinite(t, "time");
  if (t <= times_.front()) {
    return zero_rates_.front();
  }
  if (t >= times_.back()) {
    return zero_rates_.back();
  }
  return evaluateSpline(t);
}

double YieldCurve::zeroRateDerivative(double t) const {
  requireFinite(t, "time");
  if (t <= times_.front() || t >= times_.back()) {
    return 0.0;
  }
  return evaluateSplineDerivative(t);
}

double YieldCurve::instantaneousForward(double t) const {
  requireFinite(t, "time");
  const double clamped_t = std::max(0.0, t);
  return zeroRate(clamped_t) + clamped_t * zeroRateDerivative(clamped_t);
}

double YieldCurve::discountFactor(double t) const {
  requireFinite(t, "time");
  if (t < 0.0) {
    throw std::invalid_argument("discount time must be non-negative");
  }
  return std::exp(-zeroRate(t) * t);
}

double YieldCurve::continuouslyCompoundedForward(double start, double end) const {
  requireFinite(start, "start");
  requireFinite(end, "end");
  if (start < 0.0 || end <= start) {
    throw std::invalid_argument("forward interval must satisfy 0 <= start < end");
  }
  const double log_df_start = -zeroRate(start) * start;
  const double log_df_end = -zeroRate(end) * end;
  return (log_df_start - log_df_end) / (end - start);
}

const std::vector<double>& YieldCurve::times() const noexcept {
  return times_;
}

const std::vector<double>& YieldCurve::zeroRates() const noexcept {
  return zero_rates_;
}

void YieldCurve::validateInputs() const {
  if (times_.size() != zero_rates_.size()) {
    throw std::invalid_argument("curve times and zero rates must have the same length");
  }
  if (times_.size() < 2) {
    throw std::invalid_argument("at least two curve nodes are required");
  }
  if (interpolation_ != InterpolationKind::NaturalCubicSpline) {
    throw std::invalid_argument("unsupported interpolation kind");
  }
  for (std::size_t i = 0; i < times_.size(); ++i) {
    requireFinite(times_[i], "curve time");
    requireFinite(zero_rates_[i], "zero rate");
    if (times_[i] < 0.0) {
      throw std::invalid_argument("curve times must be non-negative");
    }
    if (i > 0 && times_[i] <= times_[i - 1]) {
      throw std::invalid_argument("curve times must be strictly increasing");
    }
  }
}

void YieldCurve::buildNaturalCubicSpline() {
  const std::size_t n = times_.size();
  spline_b_.assign(n - 1, 0.0);
  spline_c_.assign(n, 0.0);
  spline_d_.assign(n - 1, 0.0);

  std::vector<double> h(n - 1, 0.0);
  std::vector<double> alpha(n, 0.0);
  std::vector<double> lower(n, 0.0);
  std::vector<double> mu(n, 0.0);
  std::vector<double> z(n, 0.0);

  for (std::size_t i = 0; i + 1 < n; ++i) {
    h[i] = times_[i + 1] - times_[i];
  }

  for (std::size_t i = 1; i + 1 < n; ++i) {
    alpha[i] = (3.0 / h[i]) * (zero_rates_[i + 1] - zero_rates_[i]) -
               (3.0 / h[i - 1]) * (zero_rates_[i] - zero_rates_[i - 1]);
  }

  lower[0] = 1.0;
  mu[0] = 0.0;
  z[0] = 0.0;

  for (std::size_t i = 1; i + 1 < n; ++i) {
    lower[i] = 2.0 * (times_[i + 1] - times_[i - 1]) - h[i - 1] * mu[i - 1];
    if (std::abs(lower[i]) <= 1.0e-14) {
      throw std::runtime_error("degenerate cubic spline system");
    }
    mu[i] = h[i] / lower[i];
    z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / lower[i];
  }

  lower[n - 1] = 1.0;
  z[n - 1] = 0.0;
  spline_c_[n - 1] = 0.0;

  for (std::size_t offset = 1; offset < n; ++offset) {
    const std::size_t j = n - 1 - offset;
    spline_c_[j] = z[j] - mu[j] * spline_c_[j + 1];
    spline_b_[j] = (zero_rates_[j + 1] - zero_rates_[j]) / h[j] -
                   h[j] * (spline_c_[j + 1] + 2.0 * spline_c_[j]) / 3.0;
    spline_d_[j] = (spline_c_[j + 1] - spline_c_[j]) / (3.0 * h[j]);
  }
}

std::size_t YieldCurve::locateInterval(double t) const {
  const auto upper = std::upper_bound(times_.begin(), times_.end(), t);
  const std::size_t index = static_cast<std::size_t>(std::distance(times_.begin(), upper));
  return std::min(index - 1, times_.size() - 2);
}

double YieldCurve::evaluateSpline(double t) const {
  const std::size_t i = locateInterval(t);
  const double dx = t - times_[i];
  return zero_rates_[i] + spline_b_[i] * dx + spline_c_[i] * dx * dx +
         spline_d_[i] * dx * dx * dx;
}

double YieldCurve::evaluateSplineDerivative(double t) const {
  const std::size_t i = locateInterval(t);
  const double dx = t - times_[i];
  return spline_b_[i] + 2.0 * spline_c_[i] * dx + 3.0 * spline_d_[i] * dx * dx;
}

}  // namespace quant_engine

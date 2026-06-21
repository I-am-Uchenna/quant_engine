#include "HullWhiteProcess.hpp"

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

HullWhiteProcess::HullWhiteProcess(YieldCurve curve, double mean_reversion, double volatility)
    : curve_(std::move(curve)),
      mean_reversion_(mean_reversion),
      volatility_(volatility),
      initial_short_rate_(curve_.instantaneousForward(0.0)) {
  validate();
}

std::size_t HullWhiteProcess::factors() const noexcept {
  return 1;
}

double HullWhiteProcess::initialShortRate() const {
  return initial_short_rate_;
}

double HullWhiteProcess::meanReversion() const noexcept {
  return mean_reversion_;
}

double HullWhiteProcess::volatility() const noexcept {
  return volatility_;
}

const YieldCurve& HullWhiteProcess::curve() const noexcept {
  return curve_;
}

double HullWhiteProcess::alpha(double t) const {
  requireFinite(t, "time");
  if (t < 0.0) {
    throw std::invalid_argument("time must be non-negative");
  }
  const double one_minus_exp = 1.0 - std::exp(-mean_reversion_ * t);
  return curve_.instantaneousForward(t) +
         (volatility_ * volatility_ / (2.0 * mean_reversion_ * mean_reversion_)) *
             one_minus_exp * one_minus_exp;
}

double HullWhiteProcess::bondVolatility(double option_expiry, double bond_maturity) const {
  requireFinite(option_expiry, "option expiry");
  requireFinite(bond_maturity, "bond maturity");
  if (option_expiry < 0.0 || bond_maturity < option_expiry) {
    throw std::invalid_argument("bond option dates must satisfy 0 <= expiry <= maturity");
  }
  if (option_expiry == 0.0 || bond_maturity == option_expiry || volatility_ == 0.0) {
    return 0.0;
  }
  const double b = (1.0 - std::exp(-mean_reversion_ * (bond_maturity - option_expiry))) /
                   mean_reversion_;
  const double variance_scale =
      (1.0 - std::exp(-2.0 * mean_reversion_ * option_expiry)) /
      (2.0 * mean_reversion_);
  return volatility_ * b * std::sqrt(std::max(0.0, variance_scale));
}

double HullWhiteProcess::discountBond(double t, double maturity, double r_t) const {
  requireFinite(t, "time");
  requireFinite(maturity, "maturity");
  requireFinite(r_t, "short rate");
  if (t < 0.0 || maturity < t) {
    throw std::invalid_argument("bond dates must satisfy 0 <= t <= maturity");
  }
  if (maturity == t) {
    return 1.0;
  }

  const double b = (1.0 - std::exp(-mean_reversion_ * (maturity - t))) / mean_reversion_;
  const double p0_t = curve_.discountFactor(t);
  const double p0_t_maturity = curve_.discountFactor(maturity);
  const double convexity =
      (volatility_ * volatility_ / (4.0 * mean_reversion_)) *
      (1.0 - std::exp(-2.0 * mean_reversion_ * t)) * b * b;
  const double a =
      (p0_t_maturity / p0_t) * std::exp(b * curve_.instantaneousForward(t) - convexity);
  return a * std::exp(-b * r_t);
}

TransitionMoments HullWhiteProcess::transitionMoments(double s, double t, double r_s) const {
  requireFinite(s, "start time");
  requireFinite(t, "end time");
  requireFinite(r_s, "short rate");
  if (s < 0.0 || t < s) {
    throw std::invalid_argument("transition dates must satisfy 0 <= s <= t");
  }
  if (t == s) {
    return TransitionMoments{r_s, 0.0};
  }

  const double dt = t - s;
  const double decay = std::exp(-mean_reversion_ * dt);
  const double mean = r_s * decay + alpha(t) - alpha(s) * decay;
  const double variance =
      (volatility_ * volatility_ / (2.0 * mean_reversion_)) *
      (1.0 - std::exp(-2.0 * mean_reversion_ * dt));
  return TransitionMoments{mean, std::max(0.0, variance)};
}

void HullWhiteProcess::simulateShortRatePaths(const double* standard_normals,
                                              std::size_t path_count,
                                              const double* time_grid,
                                              std::size_t time_count,
                                              double* output_rates) const {
  if (path_count == 0 || time_count == 0) {
    return;
  }
  if (time_grid == nullptr || output_rates == nullptr) {
    throw std::invalid_argument("time grid and output rate pointers must not be null");
  }
  if (time_count > 1 && standard_normals == nullptr) {
    throw std::invalid_argument("normal shock pointer must not be null");
  }

  for (std::size_t j = 0; j < time_count; ++j) {
    requireFinite(time_grid[j], "time grid point");
    if (j > 0 && time_grid[j] <= time_grid[j - 1]) {
      throw std::invalid_argument("time grid must be strictly increasing");
    }
  }

  for (std::size_t path = 0; path < path_count; ++path) {
    output_rates[rowMajorIndex(path, 0, time_count)] = initial_short_rate_;
  }

  const std::size_t normal_cols = time_count - 1;
  for (std::size_t step = 1; step < time_count; ++step) {
    const double s = time_grid[step - 1];
    const double t = time_grid[step];
    const double dt = t - s;
    const double decay = std::exp(-mean_reversion_ * dt);
    const double alpha_s = alpha(s);
    const double alpha_t = alpha(t);
    const double variance =
        (volatility_ * volatility_ / (2.0 * mean_reversion_)) *
        (1.0 - std::exp(-2.0 * mean_reversion_ * dt));
    const double stddev = std::sqrt(std::max(0.0, variance));

    for (std::size_t path = 0; path < path_count; ++path) {
      const double r_prev = output_rates[rowMajorIndex(path, step - 1, time_count)];
      const double z = standard_normals[rowMajorIndex(path, step - 1, normal_cols)];
      output_rates[rowMajorIndex(path, step, time_count)] =
          r_prev * decay + alpha_t - alpha_s * decay + stddev * z;
    }
  }
}

void HullWhiteProcess::integratedShortRates(const double* short_rate_paths,
                                            std::size_t path_count,
                                            const double* time_grid,
                                            std::size_t time_count,
                                            double* output_integrals) const {
  if (path_count == 0 || time_count == 0) {
    return;
  }
  if (short_rate_paths == nullptr || time_grid == nullptr || output_integrals == nullptr) {
    throw std::invalid_argument("rate path, time grid and output pointers must not be null");
  }

  for (std::size_t path = 0; path < path_count; ++path) {
    output_integrals[rowMajorIndex(path, 0, time_count)] = 0.0;
  }

  for (std::size_t step = 1; step < time_count; ++step) {
    const double dt = time_grid[step] - time_grid[step - 1];
    if (dt <= 0.0 || !std::isfinite(dt)) {
      throw std::invalid_argument("time grid must be strictly increasing");
    }
    for (std::size_t path = 0; path < path_count; ++path) {
      const double r_prev = short_rate_paths[rowMajorIndex(path, step - 1, time_count)];
      const double r_curr = short_rate_paths[rowMajorIndex(path, step, time_count)];
      output_integrals[rowMajorIndex(path, step, time_count)] =
          output_integrals[rowMajorIndex(path, step - 1, time_count)] +
          0.5 * (r_prev + r_curr) * dt;
    }
  }
}

void HullWhiteProcess::validate() const {
  requireFinite(mean_reversion_, "mean reversion");
  requireFinite(volatility_, "volatility");
  if (mean_reversion_ <= 0.0) {
    throw std::invalid_argument("mean reversion must be strictly positive");
  }
  if (volatility_ < 0.0) {
    throw std::invalid_argument("volatility must be non-negative");
  }
}

}  // namespace quant_engine

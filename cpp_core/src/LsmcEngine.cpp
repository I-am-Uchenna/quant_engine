#include "LsmcEngine.hpp"

#include "StochasticProcess.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace quant_engine {
namespace {

constexpr std::size_t kBasisCount = 4U;
constexpr double kEpsilon = 1.0e-12;

void requireFinite(double value, const char* name) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

double normalCdf(double x) {
  return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

void weightedLaguerreBasis(double short_rate, std::array<double, kBasisCount>& basis) {
  const double x = std::max(0.0, 20.0 * (short_rate + 0.05));
  const double weight = std::exp(-0.5 * x);
  const double x2 = x * x;
  const double x3 = x2 * x;
  basis[0] = weight;
  basis[1] = weight * (1.0 - x);
  basis[2] = weight * (1.0 - 2.0 * x + 0.5 * x2);
  basis[3] = weight * (1.0 - 3.0 * x + 1.5 * x2 - x3 / 6.0);
}

double matrixAt(const std::array<double, kBasisCount * kBasisCount>& matrix,
                std::size_t row,
                std::size_t col) {
  return matrix[rowMajorIndex(row, col, kBasisCount)];
}

void setMatrixAt(std::array<double, kBasisCount * kBasisCount>& matrix,
                 std::size_t row,
                 std::size_t col,
                 double value) {
  matrix[rowMajorIndex(row, col, kBasisCount)] = value;
}

void jacobiEigenSymmetric4(const std::array<double, kBasisCount * kBasisCount>& input,
                           std::array<double, kBasisCount * kBasisCount>& eigenvectors,
                           std::array<double, kBasisCount>& eigenvalues) {
  std::array<double, kBasisCount * kBasisCount> a = input;
  eigenvectors.fill(0.0);
  for (std::size_t i = 0; i < kBasisCount; ++i) {
    setMatrixAt(eigenvectors, i, i, 1.0);
  }

  for (std::size_t iteration = 0; iteration < 64U; ++iteration) {
    std::size_t p = 0U;
    std::size_t q = 1U;
    double max_off_diagonal = std::abs(matrixAt(a, p, q));
    for (std::size_t row = 0; row < kBasisCount; ++row) {
      for (std::size_t col = row + 1U; col < kBasisCount; ++col) {
        const double value = std::abs(matrixAt(a, row, col));
        if (value > max_off_diagonal) {
          max_off_diagonal = value;
          p = row;
          q = col;
        }
      }
    }

    if (max_off_diagonal < 1.0e-14) {
      break;
    }

    const double app = matrixAt(a, p, p);
    const double aqq = matrixAt(a, q, q);
    const double apq = matrixAt(a, p, q);
    const double tau = (aqq - app) / (2.0 * apq);
    const double sign = tau >= 0.0 ? 1.0 : -1.0;
    const double tangent = sign / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
    const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
    const double sine = tangent * cosine;

    for (std::size_t k = 0; k < kBasisCount; ++k) {
      if (k == p || k == q) {
        continue;
      }
      const double akp = matrixAt(a, k, p);
      const double akq = matrixAt(a, k, q);
      const double rotated_p = cosine * akp - sine * akq;
      const double rotated_q = sine * akp + cosine * akq;
      setMatrixAt(a, k, p, rotated_p);
      setMatrixAt(a, p, k, rotated_p);
      setMatrixAt(a, k, q, rotated_q);
      setMatrixAt(a, q, k, rotated_q);
    }

    const double c2 = cosine * cosine;
    const double s2 = sine * sine;
    const double two_sc = 2.0 * sine * cosine;
    setMatrixAt(a, p, p, c2 * app - two_sc * apq + s2 * aqq);
    setMatrixAt(a, q, q, s2 * app + two_sc * apq + c2 * aqq);
    setMatrixAt(a, p, q, 0.0);
    setMatrixAt(a, q, p, 0.0);

    for (std::size_t row = 0; row < kBasisCount; ++row) {
      const double vip = matrixAt(eigenvectors, row, p);
      const double viq = matrixAt(eigenvectors, row, q);
      setMatrixAt(eigenvectors, row, p, cosine * vip - sine * viq);
      setMatrixAt(eigenvectors, row, q, sine * vip + cosine * viq);
    }
  }

  for (std::size_t i = 0; i < kBasisCount; ++i) {
    eigenvalues[i] = matrixAt(a, i, i);
  }
}

std::array<double, kBasisCount> solveRidgeViaSvd(
    const std::array<double, kBasisCount * kBasisCount>& gram,
    const std::array<double, kBasisCount>& rhs,
    double ridge_lambda) {
  std::array<double, kBasisCount * kBasisCount> eigenvectors{};
  std::array<double, kBasisCount> eigenvalues{};
  jacobiEigenSymmetric4(gram, eigenvectors, eigenvalues);

  const double lambda = std::max(ridge_lambda, 1.0e-14);
  std::array<double, kBasisCount> projected{};
  projected.fill(0.0);
  for (std::size_t col = 0; col < kBasisCount; ++col) {
    double dot = 0.0;
    for (std::size_t row = 0; row < kBasisCount; ++row) {
      dot += matrixAt(eigenvectors, row, col) * rhs[row];
    }
    const double singular_value_squared = std::max(0.0, eigenvalues[col]);
    projected[col] = dot / (singular_value_squared + lambda);
  }

  std::array<double, kBasisCount> beta{};
  beta.fill(0.0);
  for (std::size_t row = 0; row < kBasisCount; ++row) {
    for (std::size_t col = 0; col < kBasisCount; ++col) {
      beta[row] += matrixAt(eigenvectors, row, col) * projected[col];
    }
  }
  return beta;
}

double dotBasis(const std::array<double, kBasisCount>& basis,
                const std::array<double, kBasisCount>& beta) {
  double value = 0.0;
  for (std::size_t i = 0; i < kBasisCount; ++i) {
    value += basis[i] * beta[i];
  }
  return value;
}

}  // namespace

LsmcEngine::LsmcEngine(HullWhiteProcess process) : process_(std::move(process)) {}

const HullWhiteProcess& LsmcEngine::process() const noexcept {
  return process_;
}

PricingResult LsmcEngine::price(const BermudanSwaptionSpec& spec,
                                const LsmcSimulationConfig& config) const {
  validateSpec(spec);
  validateConfig(config);

  const std::size_t path_count = config.path_count;
  const std::size_t exercise_count = spec.exercise_times.size();
  const std::size_t time_count = exercise_count + 1U;
  const std::size_t step_count = time_count - 1U;

  std::vector<double> time_grid(time_count, 0.0);
  for (std::size_t i = 0; i < exercise_count; ++i) {
    time_grid[i + 1U] = spec.exercise_times[i];
  }

  std::vector<double> independent_normals(path_count * step_count, 0.0);
  std::vector<double> brownian_values(path_count * time_count, 0.0);
  std::vector<double> transition_normals(path_count * step_count, 0.0);
  std::vector<double> short_rates(path_count * time_count, 0.0);
  std::vector<double> integrated_rates(path_count * time_count, 0.0);
  std::vector<double> cashflows(path_count, 0.0);
  std::vector<double> intrinsic_values(path_count, 0.0);
  std::vector<double> discounted_values(path_count, 0.0);
  std::vector<double> path_values(path_count, 0.0);

  SobolSequence sobol(SobolConfig{step_count, config.sobol_bits, config.seed,
                                  config.scrambling});
  sobol.fillStandardNormal(1U, path_count, independent_normals.data());

  BrownianBridge bridge(time_grid.data(), time_count);
  bridge.transformToBrownianValues(independent_normals.data(), path_count,
                                   brownian_values.data());

  for (std::size_t path = 0; path < path_count; ++path) {
    for (std::size_t step = 1; step < time_count; ++step) {
      const double dt = time_grid[step] - time_grid[step - 1U];
      const double increment = brownian_values[rowMajorIndex(path, step, time_count)] -
                               brownian_values[rowMajorIndex(path, step - 1U, time_count)];
      transition_normals[rowMajorIndex(path, step - 1U, step_count)] =
          increment / std::sqrt(dt);
    }
  }

  process_.simulateShortRatePaths(transition_normals.data(), path_count, time_grid.data(),
                                  time_count, short_rates.data());
  process_.integratedShortRates(short_rates.data(), path_count, time_grid.data(), time_count,
                                integrated_rates.data());

  PricingResult result{};
  result.path_count = path_count;
  result.exercise_times = spec.exercise_times;
  result.exercise_boundary.assign(exercise_count,
                                  std::numeric_limits<double>::quiet_NaN());
  result.rate_time_grid = time_grid;

  const std::size_t sample_count = std::min<std::size_t>(path_count, 64U);
  result.sample_short_rate_paths.assign(sample_count * time_count, 0.0);
  for (std::size_t path = 0; path < sample_count; ++path) {
    for (std::size_t step = 0; step < time_count; ++step) {
      result.sample_short_rate_paths[rowMajorIndex(path, step, time_count)] =
          short_rates[rowMajorIndex(path, step, time_count)];
    }
  }

  const std::size_t final_col = exercise_count;
  double final_boundary_sum = 0.0;
  std::size_t final_exercise_count = 0U;
  for (std::size_t path = 0; path < path_count; ++path) {
    const double rate = short_rates[rowMajorIndex(path, final_col, time_count)];
    const double intrinsic = std::max(0.0, swapPresentValue(time_grid[final_col], rate, spec));
    cashflows[path] = intrinsic;
    if (intrinsic > 0.0) {
      final_boundary_sum += rate;
      ++final_exercise_count;
    }
  }
  if (final_exercise_count > 0U) {
    result.exercise_boundary[exercise_count - 1U] =
        final_boundary_sum / static_cast<double>(final_exercise_count);
  }

  std::size_t current_col = final_col;
  if (exercise_count > 1U) {
    for (std::size_t backward = exercise_count - 1U; backward > 0U; --backward) {
      const std::size_t exercise_index = backward - 1U;
      const std::size_t target_col = exercise_index + 1U;

      for (std::size_t path = 0; path < path_count; ++path) {
        const double discount =
            std::exp(-(integrated_rates[rowMajorIndex(path, current_col, time_count)] -
                       integrated_rates[rowMajorIndex(path, target_col, time_count)]));
        cashflows[path] *= discount;
      }

      std::array<double, kBasisCount * kBasisCount> gram{};
      std::array<double, kBasisCount> rhs{};
      gram.fill(0.0);
      rhs.fill(0.0);
      std::size_t in_the_money_count = 0U;

      for (std::size_t path = 0; path < path_count; ++path) {
        const double rate = short_rates[rowMajorIndex(path, target_col, time_count)];
        const double intrinsic =
            std::max(0.0, swapPresentValue(time_grid[target_col], rate, spec));
        intrinsic_values[path] = intrinsic;
        discounted_values[path] = cashflows[path];

        if (intrinsic > 0.0) {
          std::array<double, kBasisCount> basis{};
          weightedLaguerreBasis(rate, basis);
          for (std::size_t row = 0; row < kBasisCount; ++row) {
            rhs[row] += basis[row] * discounted_values[path];
            for (std::size_t col = 0; col < kBasisCount; ++col) {
              gram[rowMajorIndex(row, col, kBasisCount)] += basis[row] * basis[col];
            }
          }
          ++in_the_money_count;
        }
      }

      std::array<double, kBasisCount> beta{};
      beta.fill(0.0);
      const bool has_regression = in_the_money_count > kBasisCount;
      if (has_regression) {
        beta = solveRidgeViaSvd(gram, rhs, config.ridge_lambda);
      }

      double boundary_sum = 0.0;
      std::size_t boundary_count = 0U;
      for (std::size_t path = 0; path < path_count; ++path) {
        if (intrinsic_values[path] <= 0.0) {
          continue;
        }

        const double rate = short_rates[rowMajorIndex(path, target_col, time_count)];
        double continuation = discounted_values[path];
        if (has_regression) {
          std::array<double, kBasisCount> basis{};
          weightedLaguerreBasis(rate, basis);
          continuation = std::max(0.0, dotBasis(basis, beta));
        }

        if (intrinsic_values[path] > continuation) {
          cashflows[path] = intrinsic_values[path];
          boundary_sum += rate;
          ++boundary_count;
        }
      }

      if (boundary_count > 0U) {
        result.exercise_boundary[exercise_index] =
            boundary_sum / static_cast<double>(boundary_count);
      }
      current_col = target_col;
    }
  }

  double sum = 0.0;
  double sum_squared = 0.0;
  for (std::size_t path = 0; path < path_count; ++path) {
    const double discount_to_zero =
        std::exp(-integrated_rates[rowMajorIndex(path, current_col, time_count)]);
    path_values[path] = cashflows[path] * discount_to_zero;
    sum += path_values[path];
    sum_squared += path_values[path] * path_values[path];
  }

  result.price = sum / static_cast<double>(path_count);
  if (path_count > 1U) {
    const double mean_square = sum_squared / static_cast<double>(path_count);
    const double variance =
        std::max(0.0, mean_square - result.price * result.price) *
        static_cast<double>(path_count) / static_cast<double>(path_count - 1U);
    result.standard_error = std::sqrt(variance / static_cast<double>(path_count));
  } else {
    result.standard_error = 0.0;
  }

  return result;
}

double LsmcEngine::europeanSwaptionJamshidian(const BermudanSwaptionSpec& spec,
                                              double option_expiry) const {
  validateSpec(spec);
  requireFinite(option_expiry, "option expiry");
  if (option_expiry <= 0.0) {
    throw std::invalid_argument("option expiry must be positive");
  }

  std::vector<double> maturities(spec.payment_times.size(), 0.0);
  std::vector<double> coupons(spec.payment_times.size(), 0.0);
  std::size_t count = 0U;
  double previous = option_expiry;
  for (std::size_t i = 0; i < spec.payment_times.size(); ++i) {
    const double payment_time = spec.payment_times[i];
    if (payment_time <= option_expiry + kEpsilon) {
      continue;
    }
    const double accrual = payment_time - previous;
    maturities[count] = payment_time;
    coupons[count] = spec.fixed_rate * accrual;
    previous = payment_time;
    ++count;
  }
  if (count == 0U) {
    return 0.0;
  }
  coupons[count - 1U] += 1.0;

  auto portfolio = [&](double short_rate) {
    double value = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
      value += coupons[i] * process_.discountBond(option_expiry, maturities[i], short_rate);
    }
    return value;
  };

  double low = -0.25;
  double high = 0.25;
  double f_low = portfolio(low) - 1.0;
  double f_high = portfolio(high) - 1.0;
  for (std::size_t expansion = 0; expansion < 64U && f_low * f_high > 0.0; ++expansion) {
    low -= 0.25 * static_cast<double>(expansion + 1U);
    high += 0.25 * static_cast<double>(expansion + 1U);
    f_low = portfolio(low) - 1.0;
    f_high = portfolio(high) - 1.0;
  }
  if (f_low * f_high > 0.0) {
    throw std::runtime_error("failed to bracket Jamshidian root");
  }

  for (std::size_t iteration = 0; iteration < 100U; ++iteration) {
    const double mid = 0.5 * (low + high);
    const double f_mid = portfolio(mid) - 1.0;
    if (std::abs(f_mid) < 1.0e-14) {
      low = mid;
      high = mid;
      break;
    }
    if (f_low * f_mid <= 0.0) {
      high = mid;
      f_high = f_mid;
    } else {
      low = mid;
      f_low = f_mid;
    }
  }
  const double root_rate = 0.5 * (low + high);

  const double p0_expiry = process_.curve().discountFactor(option_expiry);
  double option_value = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    const double maturity = maturities[i];
    const double strike = process_.discountBond(option_expiry, maturity, root_rate);
    const double p0_maturity = process_.curve().discountFactor(maturity);
    const double sigma_p = process_.bondVolatility(option_expiry, maturity);

    double bond_option = 0.0;
    if (sigma_p < 1.0e-14) {
      const double forward_bond = p0_maturity / p0_expiry;
      if (spec.type == SwaptionType::Receiver) {
        bond_option = p0_expiry * std::max(0.0, forward_bond - strike);
      } else {
        bond_option = p0_expiry * std::max(0.0, strike - forward_bond);
      }
    } else {
      const double h =
          std::log(p0_maturity / (strike * p0_expiry)) / sigma_p + 0.5 * sigma_p;
      if (spec.type == SwaptionType::Receiver) {
        bond_option = p0_maturity * normalCdf(h) -
                      strike * p0_expiry * normalCdf(h - sigma_p);
      } else {
        bond_option = strike * p0_expiry * normalCdf(-h + sigma_p) -
                      p0_maturity * normalCdf(-h);
      }
    }
    option_value += coupons[i] * bond_option;
  }

  return spec.notional * option_value;
}

double LsmcEngine::swapPresentValue(double exercise_time,
                                    double short_rate,
                                    const BermudanSwaptionSpec& spec) const {
  requireFinite(exercise_time, "exercise time");
  requireFinite(short_rate, "short rate");

  double fixed_leg = 0.0;
  double last_discount = 1.0;
  bool has_future_payment = false;
  double previous = exercise_time;

  for (double payment_time : spec.payment_times) {
    if (payment_time <= exercise_time + kEpsilon) {
      continue;
    }
    const double accrual = payment_time - previous;
    const double discount = process_.discountBond(exercise_time, payment_time, short_rate);
    fixed_leg += spec.fixed_rate * accrual * discount;
    last_discount = discount;
    previous = payment_time;
    has_future_payment = true;
  }

  if (!has_future_payment) {
    return 0.0;
  }

  const double floating_leg = 1.0 - last_discount;
  const double unit_value = spec.type == SwaptionType::Payer ? floating_leg - fixed_leg
                                                            : fixed_leg - floating_leg;
  return spec.notional * unit_value;
}

void LsmcEngine::validateSpec(const BermudanSwaptionSpec& spec) const {
  requireFinite(spec.notional, "notional");
  requireFinite(spec.fixed_rate, "fixed rate");
  if (spec.notional <= 0.0) {
    throw std::invalid_argument("swaption notional must be strictly positive");
  }
  if (spec.fixed_rate < 0.0) {
    throw std::invalid_argument("fixed rate must be non-negative");
  }
  if (spec.type != SwaptionType::Payer && spec.type != SwaptionType::Receiver) {
    throw std::invalid_argument("unsupported swaption type");
  }
  if (spec.exercise_times.empty()) {
    throw std::invalid_argument("at least one exercise time is required");
  }
  if (spec.payment_times.empty()) {
    throw std::invalid_argument("at least one payment time is required");
  }

  for (std::size_t i = 0; i < spec.exercise_times.size(); ++i) {
    requireFinite(spec.exercise_times[i], "exercise time");
    if (spec.exercise_times[i] <= 0.0) {
      throw std::invalid_argument("exercise times must be positive");
    }
    if (i > 0 && spec.exercise_times[i] <= spec.exercise_times[i - 1U]) {
      throw std::invalid_argument("exercise times must be strictly increasing");
    }
  }

  for (std::size_t i = 0; i < spec.payment_times.size(); ++i) {
    requireFinite(spec.payment_times[i], "payment time");
    if (spec.payment_times[i] <= 0.0) {
      throw std::invalid_argument("payment times must be positive");
    }
    if (i > 0 && spec.payment_times[i] <= spec.payment_times[i - 1U]) {
      throw std::invalid_argument("payment times must be strictly increasing");
    }
  }

  if (spec.payment_times.back() <= spec.exercise_times.front()) {
    throw std::invalid_argument("payment schedule must extend beyond first exercise");
  }
  if (spec.exercise_times.back() >= spec.payment_times.back()) {
    throw std::invalid_argument("last exercise must be before final payment");
  }
}

void LsmcEngine::validateConfig(const LsmcSimulationConfig& config) const {
  requireFinite(config.ridge_lambda, "ridge lambda");
  if (config.path_count < 2U) {
    throw std::invalid_argument("at least two Monte Carlo paths are required");
  }
  if (config.ridge_lambda < 0.0) {
    throw std::invalid_argument("ridge lambda must be non-negative");
  }
  if (config.sobol_bits == 0U || config.sobol_bits > 32U) {
    throw std::invalid_argument("Sobol bits must be in [1, 32]");
  }
}

}  // namespace quant_engine

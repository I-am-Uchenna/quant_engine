#include "QuasiMonteCarlo.hpp"

#include "StochasticProcess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace quant_engine {
namespace {

constexpr double kMinProbability = 1.0e-16;
constexpr double kMaxProbability = 1.0 - 1.0e-16;

void requireFinite(double value, const char* name) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

std::uint64_t splitMix64(std::uint64_t value) noexcept {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

int gf2Degree(std::uint64_t polynomial) noexcept {
  int degree = -1;
  while (polynomial != 0U) {
    polynomial >>= 1U;
    ++degree;
  }
  return degree;
}

std::uint64_t gf2Mod(std::uint64_t dividend, std::uint64_t divisor) noexcept {
  const int divisor_degree = gf2Degree(divisor);
  if (divisor_degree < 0) {
    return dividend;
  }
  while (gf2Degree(dividend) >= divisor_degree) {
    const int shift = gf2Degree(dividend) - divisor_degree;
    dividend ^= divisor << static_cast<unsigned int>(shift);
  }
  return dividend;
}

std::uint64_t gf2MultiplyMod(std::uint64_t lhs,
                             std::uint64_t rhs,
                             std::uint64_t modulus) noexcept {
  std::uint64_t result = 0U;
  while (rhs != 0U) {
    if ((rhs & 1U) != 0U) {
      result ^= lhs;
    }
    rhs >>= 1U;
    lhs <<= 1U;
    if (gf2Degree(lhs) >= gf2Degree(modulus)) {
      lhs = gf2Mod(lhs, modulus);
    }
  }
  return gf2Mod(result, modulus);
}

std::uint64_t gf2PowerMod(std::uint64_t base,
                          std::uint64_t exponent,
                          std::uint64_t modulus) noexcept {
  std::uint64_t result = 1U;
  while (exponent != 0U) {
    if ((exponent & 1U) != 0U) {
      result = gf2MultiplyMod(result, base, modulus);
    }
    exponent >>= 1U;
    base = gf2MultiplyMod(base, base, modulus);
  }
  return result;
}

std::uint64_t gf2Gcd(std::uint64_t lhs, std::uint64_t rhs) noexcept {
  while (rhs != 0U) {
    const std::uint64_t remainder = gf2Mod(lhs, rhs);
    lhs = rhs;
    rhs = remainder;
  }
  return lhs;
}

std::vector<std::uint64_t> primeFactors(std::uint64_t value) {
  std::vector<std::uint64_t> factors;
  for (std::uint64_t candidate = 2U; candidate * candidate <= value; ++candidate) {
    if (value % candidate == 0U) {
      factors.push_back(candidate);
      while (value % candidate == 0U) {
        value /= candidate;
      }
    }
  }
  if (value > 1U) {
    factors.push_back(value);
  }
  return factors;
}

bool isIrreducible(std::uint64_t polynomial, int degree) {
  constexpr std::uint64_t x = 0b10U;
  std::uint64_t x_power = x;
  for (int i = 1; i <= degree / 2; ++i) {
    x_power = gf2PowerMod(x_power, 2U, polynomial);
    if (gf2Gcd(x_power ^ x, polynomial) != 1U) {
      return false;
    }
  }
  return gf2PowerMod(x, 1ULL << static_cast<unsigned int>(degree), polynomial) == x;
}

bool isPrimitive(std::uint64_t polynomial, int degree) {
  if (!isIrreducible(polynomial, degree)) {
    return false;
  }
  constexpr std::uint64_t x = 0b10U;
  const std::uint64_t order = (1ULL << static_cast<unsigned int>(degree)) - 1ULL;
  const std::vector<std::uint64_t> factors = primeFactors(order);
  for (std::uint64_t factor : factors) {
    if (gf2PowerMod(x, order / factor, polynomial) == 1U) {
      return false;
    }
  }
  return true;
}

std::vector<std::uint64_t> primitivePolynomials(std::size_t required_count) {
  std::vector<std::uint64_t> polynomials;
  polynomials.reserve(required_count);
  for (int degree = 1; polynomials.size() < required_count && degree <= 20; ++degree) {
    const std::uint64_t begin = (1ULL << static_cast<unsigned int>(degree)) | 1ULL;
    const std::uint64_t end = 1ULL << static_cast<unsigned int>(degree + 1);
    for (std::uint64_t candidate = begin; candidate < end; candidate += 2ULL) {
      if ((candidate & (1ULL << static_cast<unsigned int>(degree))) == 0U) {
        continue;
      }
      if (isPrimitive(candidate, degree)) {
        polynomials.push_back(candidate);
        if (polynomials.size() == required_count) {
          break;
        }
      }
    }
  }
  if (polynomials.size() < required_count) {
    throw std::invalid_argument("requested Sobol dimension is too large for built-in generator");
  }
  return polynomials;
}

double inverseNormalCdf(double probability) {
  const double p = std::min(kMaxProbability, std::max(kMinProbability, probability));

  constexpr double a1 = -3.969683028665376e+01;
  constexpr double a2 = 2.209460984245205e+02;
  constexpr double a3 = -2.759285104469687e+02;
  constexpr double a4 = 1.383577518672690e+02;
  constexpr double a5 = -3.066479806614716e+01;
  constexpr double a6 = 2.506628277459239e+00;

  constexpr double b1 = -5.447609879822406e+01;
  constexpr double b2 = 1.615858368580409e+02;
  constexpr double b3 = -1.556989798598866e+02;
  constexpr double b4 = 6.680131188771972e+01;
  constexpr double b5 = -1.328068155288572e+01;

  constexpr double c1 = -7.784894002430293e-03;
  constexpr double c2 = -3.223964580411365e-01;
  constexpr double c3 = -2.400758277161838e+00;
  constexpr double c4 = -2.549732539343734e+00;
  constexpr double c5 = 4.374664141464968e+00;
  constexpr double c6 = 2.938163982698783e+00;

  constexpr double d1 = 7.784695709041462e-03;
  constexpr double d2 = 3.224671290700398e-01;
  constexpr double d3 = 2.445134137142996e+00;
  constexpr double d4 = 3.754408661907416e+00;

  constexpr double p_low = 0.02425;
  constexpr double p_high = 1.0 - p_low;

  if (p < p_low) {
    const double q = std::sqrt(-2.0 * std::log(p));
    return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }
  if (p <= p_high) {
    const double q = p - 0.5;
    const double r = q * q;
    return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
           (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
  }

  const double q = std::sqrt(-2.0 * std::log(1.0 - p));
  return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
         ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
}

}  // namespace

SobolSequence::SobolSequence(SobolConfig config) : config_(config) {
  validateConfig();
  buildDirectionNumbers();
  buildDigitalShift();
}

std::size_t SobolSequence::dimension() const noexcept {
  return config_.dimension;
}

std::uint32_t SobolSequence::bits() const noexcept {
  return config_.bits;
}

SobolScrambling SobolSequence::scrambling() const noexcept {
  return config_.scrambling;
}

void SobolSequence::fillUniform(std::uint64_t first_index,
                                std::size_t sample_count,
                                double* output) const {
  if (sample_count == 0) {
    return;
  }
  if (output == nullptr) {
    throw std::invalid_argument("Sobol output pointer must not be null");
  }

  const double scale = std::ldexp(1.0, static_cast<int>(config_.bits));
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    const std::uint64_t index = first_index + static_cast<std::uint64_t>(sample);
    for (std::size_t dim = 0; dim < config_.dimension; ++dim) {
      const std::uint32_t value = sobolInteger(index, dim);
      output[rowMajorIndex(sample, dim, config_.dimension)] =
          (static_cast<double>(value) + 0.5) / scale;
    }
  }
}

void SobolSequence::fillStandardNormal(std::uint64_t first_index,
                                       std::size_t sample_count,
                                       double* output,
                                       NormalTransform transform) const {
  if (transform != NormalTransform::AcklamInverseCdf) {
    throw std::invalid_argument("unsupported normal transform");
  }
  if (sample_count == 0) {
    return;
  }
  if (output == nullptr) {
    throw std::invalid_argument("normal output pointer must not be null");
  }

  const double scale = std::ldexp(1.0, static_cast<int>(config_.bits));
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    const std::uint64_t index = first_index + static_cast<std::uint64_t>(sample);
    for (std::size_t dim = 0; dim < config_.dimension; ++dim) {
      const std::uint32_t value = sobolInteger(index, dim);
      const double uniform = (static_cast<double>(value) + 0.5) / scale;
      output[rowMajorIndex(sample, dim, config_.dimension)] = inverseNormalCdf(uniform);
    }
  }
}

void SobolSequence::validateConfig() const {
  if (config_.dimension == 0) {
    throw std::invalid_argument("Sobol dimension must be positive");
  }
  if (config_.bits == 0 || config_.bits > 32U) {
    throw std::invalid_argument("Sobol bits must be in [1, 32]");
  }
}

void SobolSequence::buildDirectionNumbers() {
  direction_numbers_.assign(config_.dimension * static_cast<std::size_t>(config_.bits), 0U);

  for (std::uint32_t bit = 0; bit < config_.bits; ++bit) {
    direction_numbers_[rowMajorIndex(0, bit, config_.bits)] =
        1U << static_cast<unsigned int>(config_.bits - bit - 1U);
  }

  if (config_.dimension == 1) {
    return;
  }

  const std::vector<std::uint64_t> polynomials = primitivePolynomials(config_.dimension - 1U);
  for (std::size_t dim = 1; dim < config_.dimension; ++dim) {
    const std::uint64_t polynomial = polynomials[dim - 1U];
    const int degree = gf2Degree(polynomial);
    for (std::uint32_t bit = 0; bit < config_.bits; ++bit) {
      const std::size_t out_index = rowMajorIndex(dim, bit, config_.bits);
      if (bit < static_cast<std::uint32_t>(degree)) {
        const std::uint32_t m = (1U << static_cast<unsigned int>(bit + 1U)) - 1U;
        direction_numbers_[out_index] =
            m << static_cast<unsigned int>(config_.bits - bit - 1U);
      } else {
        std::uint32_t value =
            direction_numbers_[rowMajorIndex(dim, bit - static_cast<std::uint32_t>(degree),
                                             config_.bits)];
        value ^= value >> static_cast<unsigned int>(degree);
        for (int k = 1; k < degree; ++k) {
          if (((polynomial >> static_cast<unsigned int>(degree - k)) & 1ULL) != 0ULL) {
            value ^= direction_numbers_[rowMajorIndex(
                dim, bit - static_cast<std::uint32_t>(k), config_.bits)];
          }
        }
        direction_numbers_[out_index] = value;
      }
    }
  }
}

void SobolSequence::buildDigitalShift() {
  digital_shift_.assign(config_.dimension, 0U);
  if (config_.scrambling == SobolScrambling::None) {
    return;
  }
  if (config_.scrambling != SobolScrambling::DigitalShift) {
    throw std::invalid_argument("unsupported Sobol scrambling mode");
  }
  const std::uint32_t mask =
      config_.bits == 32U ? std::numeric_limits<std::uint32_t>::max()
                          : ((1U << static_cast<unsigned int>(config_.bits)) - 1U);
  for (std::size_t dim = 0; dim < config_.dimension; ++dim) {
    digital_shift_[dim] =
        static_cast<std::uint32_t>(splitMix64(config_.seed + dim * 0x9e3779b97f4a7c15ULL)) &
        mask;
  }
}

std::uint32_t SobolSequence::sobolInteger(std::uint64_t index,
                                          std::size_t dimension) const noexcept {
  std::uint64_t gray = index ^ (index >> 1U);
  std::uint32_t value = 0U;
  std::uint32_t bit = 0U;
  while (gray != 0U && bit < config_.bits) {
    if ((gray & 1ULL) != 0ULL) {
      value ^= direction_numbers_[rowMajorIndex(dimension, bit, config_.bits)];
    }
    gray >>= 1U;
    ++bit;
  }
  return value ^ digital_shift_[dimension];
}

BrownianBridge::BrownianBridge(const double* time_grid, std::size_t time_count)
    : time_grid_(time_count), construction_order_() {
  if (time_count > 0 && time_grid == nullptr) {
    throw std::invalid_argument("Brownian bridge time grid pointer must not be null");
  }
  for (std::size_t i = 0; i < time_count; ++i) {
    time_grid_[i] = time_grid[i];
  }
  validateTimeGrid();
  buildConstructionOrder();
}

std::size_t BrownianBridge::timeCount() const noexcept {
  return time_grid_.size();
}

std::size_t BrownianBridge::gaussianDimension() const noexcept {
  return time_grid_.empty() ? 0U : time_grid_.size() - 1U;
}

const std::vector<double>& BrownianBridge::timeGrid() const noexcept {
  return time_grid_;
}

const std::vector<BrownianBridgeNode>& BrownianBridge::constructionOrder() const noexcept {
  return construction_order_;
}

void BrownianBridge::transformToBrownianValues(const double* independent_normals,
                                               std::size_t path_count,
                                               double* output_brownian_values) const {
  const std::size_t time_count = time_grid_.size();
  const std::size_t dimension = gaussianDimension();
  if (path_count == 0 || time_count == 0) {
    return;
  }
  if (output_brownian_values == nullptr) {
    throw std::invalid_argument("Brownian output pointer must not be null");
  }
  if (dimension > 0 && independent_normals == nullptr) {
    throw std::invalid_argument("independent normal pointer must not be null");
  }

  for (std::size_t path = 0; path < path_count; ++path) {
    for (std::size_t t = 0; t < time_count; ++t) {
      output_brownian_values[rowMajorIndex(path, t, time_count)] = 0.0;
    }
  }

  for (std::size_t path = 0; path < path_count; ++path) {
    for (std::size_t order = 0; order < construction_order_.size(); ++order) {
      const BrownianBridgeNode& node = construction_order_[order];
      const double z = independent_normals[rowMajorIndex(path, order, dimension)];
      double value = 0.0;
      if (node.left_index == node.right_index) {
        value = node.conditional_stddev * z;
      } else {
        const double left = output_brownian_values[rowMajorIndex(path, node.left_index,
                                                                 time_count)];
        const double right = output_brownian_values[rowMajorIndex(path, node.right_index,
                                                                  time_count)];
        value = node.left_weight * left + node.right_weight * right +
                node.conditional_stddev * z;
      }
      output_brownian_values[rowMajorIndex(path, node.bridge_index, time_count)] = value;
    }
  }
}

void BrownianBridge::transformToBrownianIncrements(const double* independent_normals,
                                                  std::size_t path_count,
                                                  double* output_brownian_increments) const {
  const std::size_t time_count = time_grid_.size();
  const std::size_t dimension = gaussianDimension();
  if (path_count == 0 || dimension == 0) {
    return;
  }
  if (output_brownian_increments == nullptr) {
    throw std::invalid_argument("Brownian increment output pointer must not be null");
  }

  std::vector<double> brownian_values(path_count * time_count, 0.0);
  transformToBrownianValues(independent_normals, path_count, brownian_values.data());

  for (std::size_t path = 0; path < path_count; ++path) {
    for (std::size_t step = 1; step < time_count; ++step) {
      output_brownian_increments[rowMajorIndex(path, step - 1, dimension)] =
          brownian_values[rowMajorIndex(path, step, time_count)] -
          brownian_values[rowMajorIndex(path, step - 1, time_count)];
    }
  }
}

void BrownianBridge::validateTimeGrid() const {
  if (time_grid_.size() < 2) {
    throw std::invalid_argument("Brownian bridge requires at least two time points");
  }
  requireFinite(time_grid_.front(), "time grid point");
  if (std::abs(time_grid_.front()) > 1.0e-14) {
    throw std::invalid_argument("Brownian bridge time grid must start at zero");
  }
  for (std::size_t i = 1; i < time_grid_.size(); ++i) {
    requireFinite(time_grid_[i], "time grid point");
    if (time_grid_[i] <= time_grid_[i - 1]) {
      throw std::invalid_argument("Brownian bridge time grid must be strictly increasing");
    }
  }
}

void BrownianBridge::buildConstructionOrder() {
  const std::size_t n = time_grid_.size();
  construction_order_.clear();
  construction_order_.reserve(n - 1U);

  construction_order_.push_back(
      BrownianBridgeNode{0U, 0U, n - 1U, 0.0, 0.0, std::sqrt(time_grid_.back())});

  std::vector<std::pair<std::size_t, std::size_t>> intervals;
  intervals.reserve(n);
  intervals.push_back(std::make_pair(0U, n - 1U));

  for (std::size_t cursor = 0; cursor < intervals.size(); ++cursor) {
    const std::size_t left = intervals[cursor].first;
    const std::size_t right = intervals[cursor].second;
    if (right <= left + 1U) {
      continue;
    }

    const std::size_t mid = left + (right - left) / 2U;
    const double t_left = time_grid_[left];
    const double t_mid = time_grid_[mid];
    const double t_right = time_grid_[right];
    const double interval = t_right - t_left;
    construction_order_.push_back(BrownianBridgeNode{
        left,
        right,
        mid,
        (t_right - t_mid) / interval,
        (t_mid - t_left) / interval,
        std::sqrt((t_mid - t_left) * (t_right - t_mid) / interval)});

    intervals.push_back(std::make_pair(left, mid));
    intervals.push_back(std::make_pair(mid, right));
  }
}

}  // namespace quant_engine

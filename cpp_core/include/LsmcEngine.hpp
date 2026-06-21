#pragma once

#include "HullWhiteProcess.hpp"
#include "QuasiMonteCarlo.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace quant_engine {

enum class SwaptionType {
  Payer = 1,
  Receiver = -1
};

struct BermudanSwaptionSpec {
  double notional;
  double fixed_rate;
  SwaptionType type;
  std::vector<double> exercise_times;
  std::vector<double> payment_times;
};

struct LsmcSimulationConfig {
  std::size_t path_count;
  std::uint64_t seed;
  double ridge_lambda;
  std::uint32_t sobol_bits;
  SobolScrambling scrambling;
};

struct PricingResult {
  double price;
  double standard_error;
  std::size_t path_count;
  std::vector<double> exercise_times;
  std::vector<double> exercise_boundary;
  std::vector<double> rate_time_grid;
  std::vector<double> sample_short_rate_paths;
};

class LsmcEngine final {
 public:
  explicit LsmcEngine(HullWhiteProcess process);

  [[nodiscard]] const HullWhiteProcess& process() const noexcept;

  [[nodiscard]] PricingResult price(const BermudanSwaptionSpec& spec,
                                    const LsmcSimulationConfig& config) const;

  [[nodiscard]] double europeanSwaptionJamshidian(const BermudanSwaptionSpec& spec,
                                                  double option_expiry) const;

  [[nodiscard]] double swapPresentValue(double exercise_time,
                                        double short_rate,
                                        const BermudanSwaptionSpec& spec) const;

 private:
  void validateSpec(const BermudanSwaptionSpec& spec) const;
  void validateConfig(const LsmcSimulationConfig& config) const;

  HullWhiteProcess process_;
};

}  // namespace quant_engine

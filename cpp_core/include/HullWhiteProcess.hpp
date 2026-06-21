#pragma once

#include "StochasticProcess.hpp"
#include "YieldCurve.hpp"

namespace quant_engine {

class HullWhiteProcess final : public StochasticProcess {
 public:
  HullWhiteProcess(YieldCurve curve, double mean_reversion, double volatility);

  [[nodiscard]] std::size_t factors() const noexcept override;
  [[nodiscard]] double initialShortRate() const override;
  [[nodiscard]] double meanReversion() const noexcept;
  [[nodiscard]] double volatility() const noexcept;
  [[nodiscard]] const YieldCurve& curve() const noexcept;

  [[nodiscard]] double alpha(double t) const;
  [[nodiscard]] double bondVolatility(double option_expiry, double bond_maturity) const;
  [[nodiscard]] double discountBond(double t, double maturity, double r_t) const;
  [[nodiscard]] TransitionMoments transitionMoments(double s,
                                                    double t,
                                                    double r_s) const override;

  void simulateShortRatePaths(const double* standard_normals,
                              std::size_t path_count,
                              const double* time_grid,
                              std::size_t time_count,
                              double* output_rates) const override;

  void integratedShortRates(const double* short_rate_paths,
                            std::size_t path_count,
                            const double* time_grid,
                            std::size_t time_count,
                            double* output_integrals) const override;

 private:
  void validate() const;

  YieldCurve curve_;
  double mean_reversion_;
  double volatility_;
  double initial_short_rate_;
};

}  // namespace quant_engine

#pragma once

#include <cstddef>

namespace quant_engine {

struct TransitionMoments {
  double mean;
  double variance;
};

struct TimeGridView {
  const double* times;
  std::size_t count;
};

struct ConstMatrixView {
  const double* data;
  std::size_t rows;
  std::size_t cols;
};

struct MatrixView {
  double* data;
  std::size_t rows;
  std::size_t cols;
};

class StochasticProcess {
 public:
  virtual ~StochasticProcess() = default;

  [[nodiscard]] virtual std::size_t factors() const noexcept = 0;
  [[nodiscard]] virtual double initialShortRate() const = 0;
  [[nodiscard]] virtual TransitionMoments transitionMoments(double s,
                                                            double t,
                                                            double r_s) const = 0;

  virtual void simulateShortRatePaths(const double* standard_normals,
                                      std::size_t path_count,
                                      const double* time_grid,
                                      std::size_t time_count,
                                      double* output_rates) const = 0;

  virtual void integratedShortRates(const double* short_rate_paths,
                                    std::size_t path_count,
                                    const double* time_grid,
                                    std::size_t time_count,
                                    double* output_integrals) const = 0;
};

[[nodiscard]] constexpr std::size_t rowMajorIndex(std::size_t row,
                                                  std::size_t col,
                                                  std::size_t column_count) noexcept {
  return row * column_count + col;
}

}  // namespace quant_engine

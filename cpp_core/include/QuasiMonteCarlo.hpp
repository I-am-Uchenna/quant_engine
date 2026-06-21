#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace quant_engine {

enum class SobolScrambling {
  None,
  DigitalShift
};

enum class NormalTransform {
  AcklamInverseCdf
};

struct SobolConfig {
  std::size_t dimension;
  std::uint32_t bits;
  std::uint64_t seed;
  SobolScrambling scrambling;
};

class SobolSequence final {
 public:
  explicit SobolSequence(SobolConfig config);

  [[nodiscard]] std::size_t dimension() const noexcept;
  [[nodiscard]] std::uint32_t bits() const noexcept;
  [[nodiscard]] SobolScrambling scrambling() const noexcept;

  void fillUniform(std::uint64_t first_index,
                   std::size_t sample_count,
                   double* output) const;

  void fillStandardNormal(std::uint64_t first_index,
                          std::size_t sample_count,
                          double* output,
                          NormalTransform transform = NormalTransform::AcklamInverseCdf) const;

 private:
  void validateConfig() const;
  void buildDirectionNumbers();
  void buildDigitalShift();
  [[nodiscard]] std::uint32_t sobolInteger(std::uint64_t index,
                                           std::size_t dimension) const noexcept;

  SobolConfig config_;
  std::vector<std::uint32_t> direction_numbers_;
  std::vector<std::uint32_t> digital_shift_;
};

struct BrownianBridgeNode {
  std::size_t left_index;
  std::size_t right_index;
  std::size_t bridge_index;
  double left_weight;
  double right_weight;
  double conditional_stddev;
};

class BrownianBridge final {
 public:
  BrownianBridge(const double* time_grid, std::size_t time_count);

  [[nodiscard]] std::size_t timeCount() const noexcept;
  [[nodiscard]] std::size_t gaussianDimension() const noexcept;
  [[nodiscard]] const std::vector<double>& timeGrid() const noexcept;
  [[nodiscard]] const std::vector<BrownianBridgeNode>& constructionOrder() const noexcept;

  void transformToBrownianValues(const double* independent_normals,
                                 std::size_t path_count,
                                 double* output_brownian_values) const;

  void transformToBrownianIncrements(const double* independent_normals,
                                     std::size_t path_count,
                                     double* output_brownian_increments) const;

 private:
  void validateTimeGrid() const;
  void buildConstructionOrder();

  std::vector<double> time_grid_;
  std::vector<BrownianBridgeNode> construction_order_;
};

}  // namespace quant_engine

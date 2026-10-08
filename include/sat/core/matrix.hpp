#pragma once

#include <cstddef>
#include <vector>

namespace sat {

/// Dense row-major matrix of doubles.
///
/// Design matrices of the learning models have one row per sample (a stock on a date) and
/// one column per feature.
class Matrix {
 public:
  Matrix() = default;
  Matrix(std::size_t rows, std::size_t cols, double value = 0.0)
      : rows_(rows), cols_(cols), data_(rows * cols, value) {}

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }
  bool empty() const { return data_.empty(); }

  double& operator()(std::size_t r, std::size_t c) { return data_[r * cols_ + c]; }
  double operator()(std::size_t r, std::size_t c) const { return data_[r * cols_ + c]; }

  double* row(std::size_t r) { return data_.data() + r * cols_; }
  const double* row(std::size_t r) const { return data_.data() + r * cols_; }

  std::vector<double>& data() { return data_; }
  const std::vector<double>& data() const { return data_; }

  /// Copy of column c.
  std::vector<double> column(std::size_t c) const;

  /// The rows listed in `rows`, in that order.
  Matrix selectRows(const std::vector<std::size_t>& rows) const;

 private:
  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  std::vector<double> data_;
};

/// Solves the symmetric positive definite system A x = b by Cholesky decomposition.
/// Throws std::invalid_argument if A is not positive definite.
std::vector<double> solveSpd(Matrix a, std::vector<double> b);

}  // namespace sat

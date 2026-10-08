#include "sat/core/matrix.hpp"

#include <cmath>
#include <stdexcept>

namespace sat {

std::vector<double> Matrix::column(std::size_t c) const {
  std::vector<double> out(rows_);
  for (std::size_t r = 0; r < rows_; ++r) out[r] = (*this)(r, c);
  return out;
}

Matrix Matrix::selectRows(const std::vector<std::size_t>& rows) const {
  Matrix out(rows.size(), cols_);
  for (std::size_t k = 0; k < rows.size(); ++k)
    for (std::size_t c = 0; c < cols_; ++c) out(k, c) = (*this)(rows[k], c);
  return out;
}

std::vector<double> solveSpd(Matrix a, std::vector<double> b) {
  const std::size_t n = a.rows();
  if (a.cols() != n || b.size() != n) throw std::invalid_argument("solveSpd: dimension mismatch");
  // In-place Cholesky: the lower triangle of a becomes L.
  for (std::size_t j = 0; j < n; ++j) {
    double d = a(j, j);
    for (std::size_t k = 0; k < j; ++k) d -= a(j, k) * a(j, k);
    if (!(d > 0.0)) throw std::invalid_argument("solveSpd: matrix is not positive definite");
    a(j, j) = std::sqrt(d);
    for (std::size_t i = j + 1; i < n; ++i) {
      double s = a(i, j);
      for (std::size_t k = 0; k < j; ++k) s -= a(i, k) * a(j, k);
      a(i, j) = s / a(j, j);
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    double s = b[i];
    for (std::size_t k = 0; k < i; ++k) s -= a(i, k) * b[k];
    b[i] = s / a(i, i);
  }
  for (std::size_t i = n; i-- > 0;) {
    double s = b[i];
    for (std::size_t k = i + 1; k < n; ++k) s -= a(k, i) * b[k];
    b[i] = s / a(i, i);
  }
  return b;
}

}  // namespace sat

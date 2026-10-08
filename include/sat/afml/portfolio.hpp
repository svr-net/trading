#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "sat/core/matrix.hpp"

namespace sat::afml {

/// Sample covariance and correlation of the columns of a T x N return matrix.
Matrix covarianceMatrix(const Matrix& returns);
Matrix correlationFromCovariance(const Matrix& cov);

/// Agglomerative single-linkage clustering of the assets. Distances follow the
/// hierarchical risk parity recipe: d_ij = sqrt((1 - rho_ij) / 2), then the Euclidean
/// distance between columns of d (assets are close when they relate alike to all others).
/// Each merge joins clusters `left` and `right` (ids < N are assets, N + k is merge k).
struct Linkage {
  std::vector<std::size_t> left, right;
  std::vector<double> height;
  std::vector<std::size_t> order;  ///< leaves in dendrogram order (quasi-diagonalisation)
};

Linkage clusterAssets(const Matrix& corr);

/// Inverse-variance weights.
std::vector<double> inverseVarianceWeights(const Matrix& cov);
/// Unconstrained minimum-variance weights, Sigma^-1 1 / (1' Sigma^-1 1). A small ridge is
/// added to the diagonal when the covariance is near singular.
std::vector<double> minimumVarianceWeights(const Matrix& cov);
/// Long-only minimum-variance weights (w >= 0, sum 1) by an active-set method: solve the
/// equality-constrained problem on the active assets, drop those with negative weights and
/// repeat; the quadratic-optimiser benchmark HRP is usually compared with.
std::vector<double> longOnlyMinimumVarianceWeights(const Matrix& cov);
/// Hierarchical risk parity: recursive bisection of the dendrogram order, splitting each
/// cluster's weight between its halves in inverse proportion to their inverse-variance
/// portfolio variances. Needs no matrix inversion and is long-only by construction.
std::vector<double> hierarchicalRiskParity(const Matrix& cov, const std::vector<std::size_t>& order);

/// Portfolio variance w' Sigma w.
double portfolioVariance(const Matrix& cov, const std::vector<double>& w);

/// Out-of-sample comparison on simulated data: clusters of correlated assets with random
/// shocks. Each trial estimates the weights of every method on `inSample` periods and
/// measures the realised variance of the next `outOfSample` periods.
struct AllocationTrial {
  std::size_t assets = 10, clusters = 3, inSample = 260, outOfSample = 22, trials = 100;
  double clusterCorrelation = 0.6;
  std::uint64_t seed = 9;
};

struct AllocationComparison {
  /// Out-of-sample variance per trial, annualised.
  std::vector<double> hrpVariance, ivpVariance, minVarVariance, longOnlyMinVarVariance;
  /// Average largest single weight and average gross exposure (sum |w|) of each method.
  double hrpMaxWeight = 0, ivpMaxWeight = 0, minVarMaxWeight = 0, longOnlyMaxWeight = 0;
  double minVarGross = 0;
};

AllocationComparison compareAllocations(const AllocationTrial& spec);

}  // namespace sat::afml

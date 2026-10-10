#pragma once

// Topology and geometry of the market, every day, at the model's dyadic half-lives h. Only data
// up to each day's close is used.
//
//  - Persistent homology in dimension 0: the stocks' exponentially weighted correlation matrix
//    (half-life h) defines distances d_ij = sqrt(2 (1 - rho_ij)); the total persistence of the
//    dimension-0 barcode of their Vietoris-Rips filtration equals the length of the minimum
//    spanning tree. Reported as the mean edge length ("tree length"; Mantegna 1999, Onnela et
//    al. 2003: the tree contracts in crashes).
//  - Effective dimension: the participation ratio of the correlation matrix, n^2 / sum rho_ij^2,
//    divided by n (1: independent stocks; 1/n: one common factor), the spectral counterpart of
//    the absorption ratio (Kritzman et al. 2011).
//  - Riemannian geometry: the affine-invariant geodesic distance on the manifold of covariance
//    matrices, || log(C_long^(-1/2) C_h C_long^(-1/2)) ||_F, between the covariance of the signal
//    families' returns at half-life h and at the longest half-life: how far the factor structure
//    of the recent past has moved from its long-run shape (Pennec et al. 2006).

#include <vector>

#include "ofm/data.hpp"
#include "ofm/model.hpp"

namespace ofm {

struct MarketStructure {
  std::vector<std::uint32_t> halfLives;
  Panel treeLength, dimension, geodesic;  ///< T x halfLives (NaN until enough history)
};

/// familyReturns: per day, the signal families' returns (Forecaster::familyReturns), known two
/// days after their day.
MarketStructure marketStructure(const Market& m, const std::vector<std::uint32_t>& halfLives,
                                const std::vector<std::vector<double>>& familyReturns);

}  // namespace ofm

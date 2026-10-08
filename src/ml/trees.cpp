#include "sat/ml/trees.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace sat {

void FeatureBinner::fit(const Matrix& X, std::size_t maxBins) {
  maxBins = std::clamp<std::size_t>(maxBins, 2, 255);
  thresholds_.assign(X.cols(), {});
  // Quantiles from at most 20,000 evenly spaced rows.
  const std::size_t step = std::max<std::size_t>(1, X.rows() / 20000);
  std::vector<double> v;
  for (std::size_t f = 0; f < X.cols(); ++f) {
    v.clear();
    for (std::size_t r = 0; r < X.rows(); r += step)
      if (std::isfinite(X(r, f))) v.push_back(X(r, f));
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    auto& thr = thresholds_[f];
    if (v.size() <= 1) continue;
    if (v.size() <= maxBins) {
      for (std::size_t k = 0; k + 1 < v.size(); ++k) thr.push_back(0.5 * (v[k] + v[k + 1]));
      continue;
    }
    for (std::size_t b = 1; b < maxBins; ++b) {
      const std::size_t k = b * v.size() / maxBins;
      const double t = 0.5 * (v[k - 1] + v[k]);
      if (thr.empty() || t > thr.back()) thr.push_back(t);
    }
  }
}

std::uint8_t FeatureBinner::code(std::size_t f, double x) const {
  const auto& thr = thresholds_[f];
  if (!std::isfinite(x)) return 0;
  return static_cast<std::uint8_t>(std::lower_bound(thr.begin(), thr.end(), x) - thr.begin());
}

std::vector<std::uint8_t> FeatureBinner::transform(const Matrix& X) const {
  if (X.cols() != thresholds_.size()) throw std::invalid_argument("binner: feature count mismatch");
  std::vector<std::uint8_t> codes(X.rows() * X.cols());
  for (std::size_t f = 0; f < X.cols(); ++f)
    for (std::size_t r = 0; r < X.rows(); ++r) codes[f * X.rows() + r] = code(f, X(r, f));
  return codes;
}

double RegressionTree::predict(const double* x) const {
  int n = 0;
  while (nodes[static_cast<std::size_t>(n)].feature >= 0) {
    const TreeNode& node = nodes[static_cast<std::size_t>(n)];
    const double v = x[node.feature];
    n = (std::isfinite(v) ? v : -1e300) <= node.threshold ? node.left : node.right;
  }
  return nodes[static_cast<std::size_t>(n)].value;
}

namespace {

struct Split {
  double gain = 0.0;
  int feature = -1;
  std::size_t bin = 0;
};

struct OpenLeaf {
  int node;
  std::size_t depth;
  std::vector<std::uint32_t> rows;
  double G, H;
  Split best;
  std::size_t order;
};

double leafScore(double G, double H, double lambda) { return G * G / (H + lambda); }

}  // namespace

RegressionTree growTree(const FeatureBinner& binner, const std::vector<std::uint8_t>& codes, std::size_t numRows,
                        const std::vector<double>& g, const std::vector<double>& h, const std::vector<std::uint32_t>& rows,
                        const std::vector<std::size_t>& features, const TreeParams& p, Rng& rng,
                        std::vector<double>& importance) {
  RegressionTree tree;
  std::vector<double> histG(256), histH(256);

  auto findSplit = [&](OpenLeaf& leaf) {
    leaf.best = Split{};
    if (leaf.depth >= p.maxDepth || leaf.rows.size() < 2) return;
    std::vector<std::size_t> tried = features;
    if (p.featureFractionPerNode < 1.0) {
      shuffle(tried, rng);
      const auto keep = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(p.featureFractionPerNode * static_cast<double>(tried.size()))));
      tried.resize(std::min(keep, tried.size()));
    }
    const double parent = leafScore(leaf.G, leaf.H, p.lambda);
    for (std::size_t f : tried) {
      const std::size_t nb = binner.numBins(f);
      if (nb < 2) continue;
      std::fill(histG.begin(), histG.begin() + static_cast<std::ptrdiff_t>(nb), 0.0);
      std::fill(histH.begin(), histH.begin() + static_cast<std::ptrdiff_t>(nb), 0.0);
      const std::uint8_t* col = codes.data() + f * numRows;
      for (std::uint32_t r : leaf.rows) {
        histG[col[r]] += g[r];
        histH[col[r]] += h[r];
      }
      double GL = 0.0, HL = 0.0;
      for (std::size_t b = 0; b + 1 < nb; ++b) {
        GL += histG[b];
        HL += histH[b];
        const double GR = leaf.G - GL, HR = leaf.H - HL;
        if (HL < p.minChildWeight || HR < p.minChildWeight) continue;
        const double gain = 0.5 * (leafScore(GL, HL, p.lambda) + leafScore(GR, HR, p.lambda) - parent) - p.gamma;
        if (gain > leaf.best.gain + 1e-12) leaf.best = Split{gain, static_cast<int>(f), b};
      }
    }
  };

  auto makeLeaf = [&](std::vector<std::uint32_t> leafRows, std::size_t depth, std::size_t order) {
    OpenLeaf leaf{static_cast<int>(tree.nodes.size()), depth, std::move(leafRows), 0.0, 0.0, {}, order};
    for (std::uint32_t r : leaf.rows) {
      leaf.G += g[r];
      leaf.H += h[r];
    }
    TreeNode node;
    node.value = -leaf.G / (leaf.H + p.lambda);
    if (!std::isfinite(node.value)) node.value = 0.0;
    tree.nodes.push_back(node);
    findSplit(leaf);
    return leaf;
  };

  std::vector<OpenLeaf> open;
  std::size_t order = 0;
  open.push_back(makeLeaf(rows, 0, order++));
  std::size_t leaves = 1;
  while (!open.empty()) {
    if (p.maxLeaves > 0 && leaves >= p.maxLeaves) break;
    // Leaf-wise: the open leaf with the largest gain. Depth-wise: the oldest open leaf.
    std::size_t pick = 0;
    for (std::size_t k = 1; k < open.size(); ++k) {
      if (p.leafWise ? open[k].best.gain > open[pick].best.gain : open[k].order < open[pick].order) pick = k;
    }
    OpenLeaf leaf = std::move(open[pick]);
    open.erase(open.begin() + static_cast<std::ptrdiff_t>(pick));
    if (leaf.best.feature < 0) {
      if (p.leafWise) break;  // the best leaf cannot split, so none can
      continue;
    }
    const auto f = static_cast<std::size_t>(leaf.best.feature);
    const std::uint8_t* col = codes.data() + f * numRows;
    std::vector<std::uint32_t> left, right;
    for (std::uint32_t r : leaf.rows) (col[r] <= leaf.best.bin ? left : right).push_back(r);
    importance[f] += leaf.best.gain;
    TreeNode& node = tree.nodes[static_cast<std::size_t>(leaf.node)];
    node.feature = leaf.best.feature;
    node.threshold = binner.threshold(f, leaf.best.bin);
    const int l = static_cast<int>(tree.nodes.size());
    open.push_back(makeLeaf(std::move(left), leaf.depth + 1, order++));
    const int r = static_cast<int>(tree.nodes.size());
    open.push_back(makeLeaf(std::move(right), leaf.depth + 1, order++));
    tree.nodes[static_cast<std::size_t>(leaf.node)].left = l;
    tree.nodes[static_cast<std::size_t>(leaf.node)].right = r;
    ++leaves;
  }
  return tree;
}

namespace {

void normaliseImportance(std::vector<double>& imp) {
  const double s = std::accumulate(imp.begin(), imp.end(), 0.0);
  if (s > 0.0)
    for (double& v : imp) v /= s;
}

}  // namespace

void RandomForest::fit(const Matrix& X, const std::vector<double>& y) {
  if (X.rows() != y.size() || X.rows() == 0) throw std::invalid_argument("random forest: empty or mismatched training data");
  const std::size_t n = X.rows(), F = X.cols();
  FeatureBinner binner;
  binner.fit(X, spec_.bins);
  const auto codes = binner.transform(X);
  Rng rng(spec_.seed);
  trees_.clear();
  importance_.assign(F, 0.0);
  std::vector<std::size_t> features(F);
  std::iota(features.begin(), features.end(), std::size_t{0});
  TreeParams p;
  p.maxDepth = spec_.maxDepth;
  p.maxLeaves = spec_.maxLeaves;
  p.lambda = 0.0;
  p.minChildWeight = static_cast<double>(std::max<std::size_t>(1, spec_.minLeaf));
  p.featureFractionPerNode = single_ ? 1.0 : std::clamp(spec_.colsample, 0.05, 1.0);
  const std::size_t numTrees = single_ ? 1 : std::max<std::size_t>(1, spec_.trees);
  std::vector<double> g(n), h(n);
  std::vector<std::uint32_t> rows;
  for (std::size_t k = 0; k < numTrees; ++k) {
    // Bootstrap: a row drawn m times enters with weight m.
    std::fill(h.begin(), h.end(), 0.0);
    rows.clear();
    if (single_) {
      std::fill(h.begin(), h.end(), 1.0);
    } else {
      const auto draws = std::max<std::size_t>(1, static_cast<std::size_t>(std::clamp(spec_.subsample, 0.05, 1.0) * static_cast<double>(n)));
      for (std::size_t d = 0; d < draws; ++d) h[rng.below(n)] += 1.0;
    }
    for (std::size_t r = 0; r < n; ++r) {
      g[r] = -y[r] * h[r];
      if (h[r] > 0.0) rows.push_back(static_cast<std::uint32_t>(r));
    }
    trees_.push_back(growTree(binner, codes, n, g, h, rows, features, p, rng, importance_));
  }
  normaliseImportance(importance_);
}

std::vector<double> RandomForest::predictProba(const Matrix& X) const {
  std::vector<double> out(X.rows(), 0.0);
  if (trees_.empty()) throw std::logic_error("random forest is not fitted");
  for (std::size_t r = 0; r < X.rows(); ++r) {
    double s = 0.0;
    for (const auto& t : trees_) s += t.predict(X.row(r));
    out[r] = std::clamp(s / static_cast<double>(trees_.size()), 0.0, 1.0);
  }
  return out;
}

void GradientBoosting::fit(const Matrix& X, const std::vector<double>& y) {
  if (X.rows() != y.size() || X.rows() == 0) throw std::invalid_argument("gradient boosting: empty or mismatched training data");
  const std::size_t n = X.rows(), F = X.cols();
  FeatureBinner binner;
  binner.fit(X, spec_.bins);
  const auto codes = binner.transform(X);
  Rng rng(spec_.seed);
  double pos = 0.0;
  for (double v : y) pos += v;
  const double p0 = std::clamp(pos / static_cast<double>(n), 1e-4, 1.0 - 1e-4);
  base_ = std::log(p0 / (1.0 - p0));
  trees_.clear();
  importance_.assign(F, 0.0);
  TreeParams p;
  p.maxDepth = leafWise_ ? (spec_.maxDepth == 0 ? 64 : spec_.maxDepth) : spec_.maxDepth;
  p.maxLeaves = leafWise_ ? std::max<std::size_t>(2, spec_.maxLeaves == 0 ? 15 : spec_.maxLeaves) : spec_.maxLeaves;
  p.lambda = spec_.lambda;
  p.gamma = spec_.gamma;
  // min_child_weight in hessian units: p(1-p) <= 1/4 per row.
  p.minChildWeight = 0.25 * static_cast<double>(std::max<std::size_t>(1, spec_.minLeaf));
  p.leafWise = leafWise_;
  std::vector<double> margin(n, base_), g(n), h(n);
  std::vector<std::size_t> allFeatures(F);
  std::iota(allFeatures.begin(), allFeatures.end(), std::size_t{0});
  std::vector<std::uint32_t> allRows(n);
  std::iota(allRows.begin(), allRows.end(), 0u);
  const double eta = spec_.learningRate;
  for (std::size_t k = 0; k < std::max<std::size_t>(1, spec_.trees); ++k) {
    for (std::size_t r = 0; r < n; ++r) {
      const double pr = sigmoid(margin[r]);
      g[r] = pr - y[r];
      h[r] = std::max(pr * (1.0 - pr), 1e-6);
    }
    std::vector<std::uint32_t> rows = allRows;
    if (spec_.subsample < 1.0) {
      shuffle(rows, rng);
      rows.resize(std::max<std::size_t>(2, static_cast<std::size_t>(spec_.subsample * static_cast<double>(n))));
      std::sort(rows.begin(), rows.end());
    }
    std::vector<std::size_t> features = allFeatures;
    if (spec_.colsample < 1.0) {
      shuffle(features, rng);
      features.resize(std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(spec_.colsample * static_cast<double>(F)))));
      std::sort(features.begin(), features.end());
    }
    RegressionTree tree = growTree(binner, codes, n, g, h, rows, features, p, rng, importance_);
    for (auto& node : tree.nodes) node.value *= eta;
    for (std::size_t r = 0; r < n; ++r) margin[r] += tree.predict(X.row(r));
    trees_.push_back(std::move(tree));
  }
  normaliseImportance(importance_);
}

double GradientBoosting::margin(const double* x) const {
  double m = base_;
  for (const auto& t : trees_) m += t.predict(x);
  return m;
}

std::vector<double> GradientBoosting::predictProba(const Matrix& X) const {
  if (trees_.empty()) throw std::logic_error("gradient boosting is not fitted");
  std::vector<double> out(X.rows());
  for (std::size_t r = 0; r < X.rows(); ++r) out[r] = sigmoid(margin(X.row(r)));
  return out;
}

}  // namespace sat

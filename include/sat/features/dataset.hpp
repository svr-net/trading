#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "sat/core/matrix.hpp"
#include "sat/core/panel.hpp"
#include "sat/data/market_data.hpp"

namespace sat {

/// How raw factor values are made comparable across dates before learning.
enum class Normalisation {
  None,          ///< raw values (NaN replaced by 0)
  Rank,          ///< cross-sectional percentile rank, centred to [-0.5, 0.5]
  ZScore,        ///< cross-sectional z-score, clipped to [-3, 3]
};

/// What the classifiers learn to predict.
enum class LabelKind {
  /// 1 if the stock's return over the next `horizon` days is positive, else 0.
  Direction,
  /// 1 if that return beats the cross-sectional median, else 0 (relative strength).
  ExcessDirection,
  /// N-period min-max labelling (Han, Kim and Enke, 2023): 1 on a close that is the minimum
  /// of the centred window of `window` days (a buying point), 0 on a close that is its
  /// maximum (a selling point); other days carry no label and are left out of training.
  MinMax,
};

struct LabelSpec {
  LabelKind kind = LabelKind::Direction;
  std::size_t horizon = 1;  ///< Direction / ExcessDirection
  std::size_t window = 10;  ///< MinMax

  /// Number of future days a label at date t looks at; training data must end this many
  /// days before the first prediction date (purging).
  std::size_t lookahead() const;
};

Normalisation parseNormalisation(const std::string& name);
LabelKind parseLabelKind(const std::string& name);

/// Feature panels, one per alpha factor.
struct FeatureSet {
  std::vector<std::string> names;
  std::vector<Panel> panels;
  std::size_t warmup = 0;  ///< dates before every feature is defined

  std::size_t size() const { return panels.size(); }
  std::size_t dates() const { return panels.empty() ? 0 : panels[0].dates(); }
  std::size_t assets() const { return panels.empty() ? 0 : panels[0].assets(); }
};

/// Computes the given alphas and normalises them cross-sectionally. After normalisation NaN
/// becomes 0 (the cross-sectional centre), so every row from `warmup` on is complete.
FeatureSet buildFeatures(const MarketData& data, const std::vector<int>& alphaIds, Normalisation norm);

/// Label panel: 1, 0 or NaN (unlabelled, including the last dates whose future is unknown).
Panel makeLabels(const MarketData& data, const LabelSpec& spec);

/// A design matrix with one row per (date, stock) sample.
struct Dataset {
  Matrix X;
  std::vector<double> y;          ///< labels (NaN if unlabelled)
  std::vector<std::size_t> date;  ///< date index of each row
  std::vector<std::size_t> asset; ///< stock index of each row
  std::size_t lags = 1;           ///< feature vectors per row (sequence length)
};

/// Rows for dates [from, to). With `lags` > 1 a row holds the feature vectors of dates
/// t - lags + 1 .. t, oldest first (the input sequence of a recurrent model); dates before
/// the start of the panel repeat the first one. With `labelledOnly`, unlabelled rows are dropped.
Dataset assemble(const FeatureSet& features, const Panel& labels, std::size_t from, std::size_t to, std::size_t lags,
                 bool labelledOnly);

}  // namespace sat

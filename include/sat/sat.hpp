#pragma once

/// Umbrella header of the self-adaptive trading library.

#include "sat/core/matrix.hpp"
#include "sat/core/panel.hpp"
#include "sat/core/random.hpp"
#include "sat/core/stats.hpp"

#include "sat/data/market_data.hpp"
#include "sat/data/synthetic_market.hpp"

#include "sat/factors/alpha101.hpp"
#include "sat/factors/operators.hpp"

#include "sat/features/dataset.hpp"

#include "sat/ml/classifier.hpp"
#include "sat/ml/linear.hpp"
#include "sat/ml/metrics.hpp"
#include "sat/ml/neural.hpp"
#include "sat/ml/trees.hpp"
#include "sat/ml/walk_forward.hpp"

#include "sat/strategy/performance.hpp"
#include "sat/strategy/strategy.hpp"

#include "sat/adaptive/experiment.hpp"
#include "sat/adaptive/self_adaptive.hpp"

#include "sat/gpu/fused_backtest.hpp"

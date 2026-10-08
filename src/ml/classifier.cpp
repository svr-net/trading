#include "sat/ml/classifier.hpp"

#include <stdexcept>

#include "sat/ml/linear.hpp"
#include "sat/ml/neural.hpp"
#include "sat/ml/trees.hpp"

namespace sat {

const std::vector<std::string>& modelTypes() {
  static const std::vector<std::string> types = {"logistic", "svm", "tree", "forest", "xgboost", "lightgbm", "mlp", "lstm"};
  return types;
}

std::string ModelSpec::displayName() const {
  if (!name.empty()) return name;
  if (type == "logistic") return "Logistic regression";
  if (type == "svm") return "Linear SVM";
  if (type == "tree") return "Decision tree";
  if (type == "forest") return "Random forest";
  if (type == "xgboost") return "XGBoost";
  if (type == "lightgbm") return "LightGBM";
  if (type == "mlp") return "MLP";
  if (type == "lstm") return "LSTM";
  return type;
}

std::unique_ptr<Classifier> makeClassifier(const ModelSpec& spec) {
  if (spec.type == "logistic") return std::make_unique<LogisticRegression>(spec);
  if (spec.type == "svm") return std::make_unique<LinearSvm>(spec);
  if (spec.type == "tree") return std::make_unique<RandomForest>(spec, true);
  if (spec.type == "forest") return std::make_unique<RandomForest>(spec, false);
  if (spec.type == "xgboost") return std::make_unique<GradientBoosting>(spec, false);
  if (spec.type == "lightgbm") return std::make_unique<GradientBoosting>(spec, true);
  if (spec.type == "mlp") return std::make_unique<Mlp>(spec);
  if (spec.type == "lstm") return std::make_unique<Lstm>(spec);
  throw std::invalid_argument("unknown model type '" + spec.type + "'");
}

}  // namespace sat

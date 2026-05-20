#pragma once

#include "gradient_boosting.hpp"
#include "../loss/crossentropyloss.hpp"
#include "decision_tree_regressor.hpp"
#include <vector>
#include <memory>
#include <string>
#include <unordered_set>
#include <algorithm>
#include <cmath>

class GradientBoostingClassifier
    : public GradientBoosting<GradientBoostingClassifier, int> {
private:
    std::vector<std::vector<DecisionTreeRegressor>> per_class_trees;
    std::vector<double> initial_logits_;
    int n_classes_ = 0;
    mutable size_t n_samples_ = 0;
public:
    GradientBoostingClassifier(
        int n_estimators = 100,
        double learning_rate = 0.1,
        int max_depth = 3,
        int min_samples_split = 5,
        int min_samples_leaf = 2,
        const std::string& loss_name = "log_loss"
    ) : GradientBoosting(n_estimators, learning_rate, max_depth,
                         min_samples_split, min_samples_leaf) {}

size_t idx(size_t sample, size_t class_id) const {
        return class_id * n_samples_ + sample;
    }

    int count_unique_classes(const std::vector<int>& y) const {
        std::unordered_set<int> unique(y.begin(), y.end());
        return static_cast<int>(unique.size());
    }

    std::vector<double> softmax(const std::vector<double>& logits) const {
        std::vector<double> probs(n_classes_);
        double max_logit = *std::max_element(logits.begin(), logits.end());
        double sum_exp = 0.0;
        for (int k = 0; k < n_classes_; ++k) {
            probs[k] = std::exp(logits[k] - max_logit);
            sum_exp += probs[k];
        }
        for (int k = 0; k < n_classes_; ++k) {
            probs[k] /= sum_exp;
        }
        return probs;
    }

    // CRTP-методы
    std::vector<double> initialize_predictions(const std::vector<int>& y) {
        n_samples_ = y.size();
        n_classes_ = count_unique_classes(y);
        per_class_trees.resize(n_classes_);

        // 1. Сначала создаём loss (нужен для initial_prediction)
        loss = std::make_unique<CrossEntropyLoss>(lr, n_classes_);

        // 2. Вызываем initial_prediction, чтобы установить n_samples в loss
        //    и получить правильный pred (но мы его перезапишем)
        auto pred = loss->initial_prediction(y);

        // 3. Перезаписываем pred своими значениями (логитами prior)
        std::vector<int> counts(n_classes_, 0);
        for (int label : y) {
            counts[label]++;
        }

        initial_logits_.resize(n_classes_);

        for (int k = 0; k < n_classes_; ++k) {
            double prior = static_cast<double>(counts[k]) / n_samples_;
            initial_logits_[k] = std::log(prior + 1e-15);
            for (size_t i = 0; i < n_samples_; ++i) {
                pred[idx(i, k)] = initial_logits_[k];
            }
        }

        return pred;
    }

    void fit_iteration(const std::vector<std::vector<double>>& X,
                       const std::vector<double>& residuals,
                       int iter) {
        for (int k = 0; k < n_classes_; ++k) {
            std::vector<double> residuals_k(n_samples_);
            for (size_t i = 0; i < n_samples_; ++i) {
                residuals_k[i] = residuals[idx(i, k)];
            }
            DecisionTreeRegressor tree(max_depth, min_samples_split, min_samples_leaf);
            tree.fit(X, residuals_k);
            per_class_trees[k].push_back(std::move(tree));
        }
    }

    std::vector<double> get_last_predictions(const std::vector<std::vector<double>>& X) {
        std::vector<double> predictions(n_samples_ * n_classes_, 0.0);
        for (int k = 0; k < n_classes_; ++k) {
            const auto& last_tree = per_class_trees[k].back();
            for (size_t i = 0; i < n_samples_; ++i) {
                predictions[idx(i, k)] = last_tree.predict(X[i]);
            }
        }
        return predictions;
    }

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<int>& y) override {
        fit_impl(X, y);
    }

    int predict(const std::vector<double>& x) const override {
        auto probs = predict_proba(x);
        return std::max_element(probs.begin(), probs.end()) - probs.begin();
    }

    std::vector<double> predict_proba(const std::vector<double>& x) const {
        std::vector<double> logits = initial_logits_;
        for (int k = 0; k < n_classes_; ++k) {
            for (const auto& tree : per_class_trees[k]) {
                logits[k] += lr * tree.predict(x);
            }
        }
        return softmax(logits);
    }
};
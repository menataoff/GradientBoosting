#pragma once

#include "gradient_boosting.hpp"
#include "../loss/mseloss.hpp"
#include "../loss/maeloss.hpp"
#include "decision_tree_regressor.hpp"
#include <vector>
#include <memory>
#include <string>

class GradientBoostingRegressor
    : public GradientBoosting<GradientBoostingRegressor, double> {
private:
    // Хранилище деревьев
    std::vector<DecisionTreeRegressor> trees;

    double init_prediction_ = 0.0;

    // Вспомогательные методы для CRTP (не объявлены в базовом классе)
public:
    // Конструктор
    GradientBoostingRegressor(
        int n_estimators = 128,
        double learning_rate = 0.01,
        int max_depth = 8,
        int min_samples_split = 3,
        int min_samples_leaf = 2,
        const std::string& loss_name = "mse"
    ) : GradientBoosting(n_estimators, learning_rate, max_depth, min_samples_split, min_samples_leaf) {

        if (loss_name == "mse") {
            loss = std::make_unique<MSELoss>(learning_rate);
        } else if (loss_name == "mae") {
            loss = std::make_unique<MAELoss>(learning_rate);
        } else {
            throw std::invalid_argument("Unknown loss: " + loss_name);
        }
    };

    std::vector<double> initialize_predictions(const std::vector<double>& y) {
        auto pred = loss->initial_prediction(y);
        init_prediction_ = pred[0];
        return pred;
    }

    void fit_iteration(const std::vector<std::vector<double>>& X,
                       const std::vector<double>& residuals,
                       int iter) { // зачем тут iter?

        DecisionTreeRegressor tree(max_depth, min_samples_split, min_samples_leaf);
        tree.fit(X, residuals);
        trees.push_back(std::move(tree));
    }
    std::vector<double> get_last_predictions(const std::vector<std::vector<double>>& X) {
        const auto& last_tree = trees.back();
        std::vector<double> predictions(X.size());
        for (size_t i = 0; i < X.size(); ++i) {
            predictions[i] = last_tree.predict(X[i]);
        }
        return predictions;
    }

    // Публичный интерфейс (объявлен в базовом)
    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& y) override {
        fit_impl(X, y);
    }

    double predict(const std::vector<double>& X) const override {
        double pred = init_prediction_;
        for (const auto& tree : trees) {
            pred += lr * tree.predict(X);
        }
        return loss->transform({pred})[0];
    }
};
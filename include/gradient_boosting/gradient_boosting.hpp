#pragma once

#include "../loss/lossfunction.hpp"
#include "../loss/mseloss.hpp"
#include "../loss/maeloss.hpp"
#include "../loss/crossentropyloss.hpp"
#include <numeric>
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
// template<typename TargetType>
// struct DataPoint {
//     std::vector<double> features;
//     TargetType target;
//
//     DataPoint(const std::vector<double>& features_val, TargetType target_val)
//         : features(features_val), target(target_val) {}
// }; - это есть в decision_tree.hpp

template<typename Derived, typename Type>
class GradientBoosting {
protected:
    double lr;
    int n_estimators;
    int max_depth;
    int min_samples_split;
    int min_samples_leaf;
    std::unique_ptr<LossFunction<Type>> loss;
public:
    GradientBoosting(
        int n_estimators_val = 128,
        double learning_rate = 0.01,
        int max_depth_val = 8,
        int min_samples_split_val = 3,
        int min_samples_leaf_val = 2) :
    n_estimators(n_estimators_val),
    lr(learning_rate),
    max_depth(max_depth_val),
    min_samples_split(min_samples_split_val),
    min_samples_leaf(min_samples_leaf_val) {}

    virtual ~GradientBoosting() = default;

    void fit_impl(const std::vector<std::vector<double>>& X,
             const std::vector<Type>& y) {

        auto& derived = static_cast<Derived&>(*this);

        // 1. Инициализация
        auto current_pred = derived.initialize_predictions(y);

        // 2. Цикл обучения
        for (int iter = 0; iter < n_estimators; ++iter) {
            // 2.1 Градиенты
            auto residuals = loss->gradient(y, current_pred);

            // 2.2 Обучаем деревья (специфика наследника)
            derived.fit_iteration(X, residuals, iter);

            // 2.3 Получаем предсказания (специфика наследника)
            auto tree_pred = derived.get_last_predictions(X);

            // 2.4 Обновляем предсказания (ОБЩАЯ ЛОГИКА из LossFunction!)
            loss->update_predictions(current_pred, tree_pred);
        }
    }

    virtual void fit(const std::vector<std::vector<double>>& X,
                     const std::vector<Type>& y) = 0;
    virtual Type predict(const std::vector<double>& x) const = 0;
};
//TODO: Логику выбора лосса. Сделать так, чтобы разрешились зависимости, не делать миллион include "...hpp"
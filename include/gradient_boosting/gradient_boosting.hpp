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
    std::string loss_str;
public:
    GradientBoosting(
        int n_estimators_val = 128,
        double learning_rate = 0.01,
        int max_depth_val = 8,
        int min_samples_split_val = 3,
        int min_samples_leaf_val = 2,
        const std::string& loss_str_val = "mse") :
    n_estimators(n_estimators_val),
    lr(learning_rate),
    max_depth(max_depth_val),
    min_samples_split(min_samples_split_val),
    min_samples_leaf(min_samples_leaf_val),
    loss_str(loss_str_val) {
        static const std::unordered_map<std::string, std::function<std::unique_ptr<LossFunction<Type>>(double)>> factory = {
            {"mse", [](double lr) { return std::make_unique<MSELoss>(lr); }},
            {"mae", [](double lr) { return std::make_unique<MAELoss>(lr); }},
            {"crossentropy", [](double lr) { return std::make_unique<CrossEntropyLoss>(lr); }},
        };
    }

};
//TODO: Логику выбора лосса. Сделать так, чтобы разрешились зависимости, не делать миллион include "...hpp"
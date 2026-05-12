#pragma once

#include "lossfunction.hpp"
#include <numeric>

class MSELoss : public LossFunction {
public:
    MSELoss(double lr) : LossFunction(lr, 1) {}

    std::vector<double> initial_prediction(const std::vector<double> &y) const override {
        double mean = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
        return std::vector<double>(y.size(), mean);
    }

    std::vector<double> gradient(const std::vector<double>& y, const std::vector<double>& y_pred) const override {
        std::vector<double> grad(y.size());
        for (size_t i = 0; i < y.size(); ++i) {
            grad[i] = y[i] - y_pred[i];
        }
        return grad;
    }

    std::vector<double> transform(const std::vector<double>& raw) const override {
        return raw;
    }
};

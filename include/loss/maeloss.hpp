#pragma once

#include "lossfunction.hpp"
#include <numeric>
#include <algorithm>

class MAELoss : public LossFunction<double> {
public:
    MAELoss(double lr) : LossFunction(lr, 1) {}

    static double compute_median(const std::vector<double>& v) {
        if (v.empty()) return 0.0;
        std::vector<double> sorted = v;
        std::sort(sorted.begin(), sorted.end());
        return sorted[sorted.size() / 2];
    }

    std::vector<double> initial_prediction(const std::vector<double> &y) const override {
        double median = compute_median(y);
        return std::vector<double>(y.size(), median);
    }

    std::vector<double> gradient(const std::vector<double>& y, const std::vector<double>& y_pred) const override {
        std::vector<double> grad(y.size());
        for (size_t i = 0; i < y.size(); ++i) {
            grad[i] = (y[i] >= y_pred[i]) ? 1.0 : -1.0;
        }
        return grad;
    }

    std::vector<double> transform(const std::vector<double>& raw) const override {
        return raw;
    }
};

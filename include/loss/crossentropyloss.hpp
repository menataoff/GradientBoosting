#pragma once

#include "lossfunction.hpp"
#include <numeric>
#include <unordered_map>
#include <cmath>
#include <stdexcept>

class CrossEntropyLoss : public LossFunction<int> {
public:
    mutable size_t n_samples = 0;

    CrossEntropyLoss(double lr, size_t n_classes) : LossFunction(lr, n_classes) {}

    size_t idx(size_t sample, size_t class_id) const {
        return class_id * n_samples + sample;
    }

    std::vector<double> initial_prediction(const std::vector<int>& y) const override {
        n_samples = y.size();

        std::vector<int> counts(n_classes, 0);
        for (int label : y) {
            if (label < 0 || label >= static_cast<int>(n_classes)) {
                throw std::invalid_argument("Label out of range");
            }
            counts[label]++;
        }

        std::vector<double> pred(n_samples * n_classes, 0.0);

        for (int k = 0; k < n_classes; ++k) {
            double prior = static_cast<double>(counts[k]) / n_samples;
            double log_prior = std::log(prior + 1e-15);

            for (size_t i = 0; i < n_samples; ++i) {
                pred[idx(i, k)] = log_prior;
            }
        }

        return pred;
    }

    std::vector<double> softmax(const std::vector<double>& logits) const {
        std::vector<double> probs(n_classes, 0.0);

        double max_logit = logits[0];
        for (size_t k = 1; k < n_classes; ++k) {
            if (logits[k] > max_logit) max_logit = logits[k];
        }

        double sum_exp = 0.0;
        for (size_t k = 0; k < n_classes; ++k) {
            probs[k] = std::exp(logits[k] - max_logit);
            sum_exp += probs[k];
        }

        for (size_t k = 0; k < n_classes; ++k) {
            probs[k] /= sum_exp;
        }

        return probs;
    }

    std::vector<double> gradient(
        const std::vector<int>& y,
        const std::vector<double>& y_pred
    ) const override {
        if (y.size() != n_samples) {
            throw std::runtime_error("y size mismatch");
        }
        if (y_pred.size() != n_samples * n_classes) {
            throw std::runtime_error("y_pred size mismatch");
        }

        std::vector<double> grad(n_samples * n_classes, 0.0);

        for (size_t i = 0; i < n_samples; ++i) {
            std::vector<double> logits(n_classes, 0.0);
            for (int k = 0; k < n_classes; ++k) {
                logits[k] = y_pred[idx(i, k)];
            }

            std::vector<double> probs = softmax(logits);

            for (int k = 0; k < n_classes; ++k) {
                double one_hot = (y[i] == k) ? 1.0 : 0.0;
                grad[idx(i, k)] = one_hot - probs[k];
            }
        }

        return grad;
    }

    std::vector<double> transform(const std::vector<double>& raw) const override {
        return softmax(raw);
    }
};

//TODO: US кроссэнтропию
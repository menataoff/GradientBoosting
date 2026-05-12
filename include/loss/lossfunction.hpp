#pragma once

#include <vector>

class LossFunction {
protected:
    double lr;
    int n_classes;
public:
    LossFunction(double learning_rate, int n_classes_arg) : lr(learning_rate), n_classes(n_classes_arg) {}

    int get_n_classes() const { return n_classes; }

    virtual ~LossFunction() = default;

    virtual std::vector<double> initial_prediction(const std::vector<double>& y) const = 0;
    //является ли проблемой, что мы делаем y - double? или просто потом в классификаторе сделаем перевод в int?
    virtual std::vector<double> gradient(
        const std::vector<double>& y,
        const std::vector<double>& y_pred) const = 0;

    virtual std::vector<double> transform(const std::vector<double>& raw) const = 0;
    virtual void update_predictions(
        std::vector<double>& pred,
        const std::vector<double>& tree_pred
    ) const {
        for (size_t i = 0; i < pred.size(); ++i) {
            pred[i] += lr * tree_pred[i];
        }
    }
};
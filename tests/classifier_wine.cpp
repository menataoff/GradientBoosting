#include "gradient_boosting/gradient_boosting_classifier.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cctype>
#include <filesystem>

// Путь к файлу (относительно папки cmake-build-debug)
const std::string DATA_PATH = "../../data/winequality-red.csv";

class Timer {
    std::chrono::high_resolution_clock::time_point start_time;
public:
    Timer() : start_time(std::chrono::high_resolution_clock::now()) {}
    double elapsed() {
        return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
    }
};

// Функция для вычисления accuracy
double compute_accuracy(const std::vector<int>& predictions, const std::vector<int>& actual) {
    if (predictions.empty()) return 0.0;
    size_t correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == actual[i]) ++correct;
    }
    return static_cast<double>(correct) / predictions.size();
}

// Функция для вычисления F1 macro
double compute_f1_macro(const std::vector<int>& predictions, const std::vector<int>& actual, int n_classes) {
    std::vector<std::vector<int>> conf_matrix(n_classes, std::vector<int>(n_classes, 0));

    for (size_t i = 0; i < predictions.size(); ++i) {
        conf_matrix[actual[i]][predictions[i]]++;
    }

    double total_f1 = 0.0;
    for (int k = 0; k < n_classes; ++k) {
        int tp = conf_matrix[k][k];
        int fp = 0, fn = 0;
        for (int j = 0; j < n_classes; ++j) {
            if (j != k) {
                fp += conf_matrix[j][k];
                fn += conf_matrix[k][j];
            }
        }
        double precision = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
        double recall = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
        double f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0;
        total_f1 += f1;
    }
    return total_f1 / n_classes;
}

// Удаление кавычек и пробелов
std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n\"");
    size_t end = s.find_last_not_of(" \t\r\n\"");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

// Загрузка данных из CSV файла (разделитель ';')
bool load_wine_data(std::vector<std::vector<double>>& X, std::vector<int>& y) {
    std::ifstream file(DATA_PATH);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << DATA_PATH << std::endl;
        std::cerr << "Current working directory: " << std::filesystem::current_path() << std::endl;
        return false;
    }

    std::string line;
    bool is_header = true;
    size_t line_num = 0;

    while (std::getline(file, line)) {
        line_num++;
        if (is_header) {
            is_header = false;
            continue;
        }

        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;
        int col = 0;

        while (std::getline(ss, value, ';')) {
            std::string cleaned = trim(value);
            if (cleaned.empty()) continue;

            try {
                if (col < 11) {  // 11 признаков
                    row.push_back(std::stod(cleaned));
                } else {  // последняя колонка — качество (цель)
                    int quality = static_cast<int>(std::stod(cleaned));
                    // Качество от 3 до 8, преобразуем в 0..5
                    y.push_back(quality - 3);
                }
            } catch (const std::exception& e) {
                std::cerr << "Line " << line_num << ", error parsing: " << cleaned << std::endl;
                break;
            }
            col++;
        }

        if (row.size() == 11) {
            X.push_back(row);
        }
    }

    file.close();
    std::cout << "Loaded " << X.size() << " samples, " << X[0].size() << " features" << std::endl;
    return !X.empty() && !y.empty() && X.size() == y.size();
}

// Разделение на train/test (80/20)
void train_test_split(const std::vector<std::vector<double>>& X, const std::vector<int>& y,
                     std::vector<std::vector<double>>& X_train, std::vector<int>& y_train,
                     std::vector<std::vector<double>>& X_test, std::vector<int>& y_test,
                     double test_ratio = 0.2, unsigned seed = 42) {

    std::mt19937 rng(seed);
    std::vector<size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    size_t test_size = static_cast<size_t>(X.size() * test_ratio);

    for (size_t i = 0; i < indices.size(); ++i) {
        if (i < test_size) {
            X_test.push_back(X[indices[i]]);
            y_test.push_back(y[indices[i]]);
        } else {
            X_train.push_back(X[indices[i]]);
            y_train.push_back(y[indices[i]]);
        }
    }
}

int main() {
    std::cout << "========== Testing GradientBoostingClassifier ==========\n" << std::endl;

    // 1. Загрузка данных
    std::vector<std::vector<double>> X;
    std::vector<int> y;

    std::cout << "Loading Wine Quality dataset from " << DATA_PATH << "..." << std::endl;
    if (!load_wine_data(X, y)) {
        std::cerr << "Failed to load data. Exiting." << std::endl;
        return 1;
    }

    // 2. Разделение на train/test
    std::vector<std::vector<double>> X_train, X_test;
    std::vector<int> y_train, y_test;
    train_test_split(X, y, X_train, y_train, X_test, y_test, 0.2, 42);

    std::cout << "Train samples: " << X_train.size() << ", Test samples: " << X_test.size() << std::endl << std::endl;

    // 3. Обучение модели Gradient Boosting Classifier
    std::cout << "Training Gradient Boosting Classifier..." << std::endl;
    Timer timer;

    GradientBoostingClassifier gb(100, 0.1, 4, 5, 2);
    gb.fit(X_train, y_train);

    double train_time = timer.elapsed();
    std::cout << "Training completed in " << train_time << " ms" << std::endl;

    // 4. Предсказания
    std::cout << "Making predictions..." << std::endl;
    std::vector<int> y_pred;
    y_pred.reserve(X_test.size());

    for (const auto& x : X_test) {
        y_pred.push_back(gb.predict(x));
    }

    // 5. Оценка качества
    double accuracy = compute_accuracy(y_pred, y_test);
    double f1_macro = compute_f1_macro(y_pred, y_test, 6);  // 6 классов (3-8 → 0-5)

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n========== Results ==========" << std::endl;
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;
    std::cout << "F1 Macro Score: " << f1_macro << std::endl;

    // 6. Вывод нескольких предсказаний для проверки
    std::cout << "\n========== Sample Predictions ==========" << std::endl;
    std::cout << "Actual vs Predicted (first 20 test samples):" << std::endl;
    for (size_t i = 0; i < std::min(size_t(20), y_test.size()); ++i) {
        std::cout << "  Actual: " << std::setw(2) << y_test[i] + 3   // обратно в шкалу 3-8
                  << " | Predicted: " << std::setw(2) << y_pred[i] + 3
                  << (y_test[i] == y_pred[i] ? " ✓" : " ✗") << std::endl;
    }

    // 7. Сравнение с baseline (предсказание самого частого класса)
    std::cout << "\n========== Baseline Comparison ==========" << std::endl;
    std::vector<int> counts(6, 0);
    for (int label : y_train) {
        counts[label]++;
    }
    int most_frequent_class = std::max_element(counts.begin(), counts.end()) - counts.begin();
    double baseline_accuracy = static_cast<double>(std::count(y_test.begin(), y_test.end(), most_frequent_class)) / y_test.size();
    std::cout << "Baseline (most frequent class " << most_frequent_class + 3 << "): " << baseline_accuracy * 100 << "%" << std::endl;
    std::cout << "Gradient Boosting improvement: " << (accuracy - baseline_accuracy) * 100 << "%" << std::endl;

    // 8. Распределение классов в тестовой выборке
    std::cout << "\n========== Class Distribution in Test Set ==========" << std::endl;
    std::vector<int> test_counts(6, 0);
    for (int label : y_test) {
        test_counts[label]++;
    }
    for (int k = 0; k < 6; ++k) {
        std::cout << "  Quality " << k + 3 << ": " << test_counts[k] << " samples" << std::endl;
    }

    return 0;
}
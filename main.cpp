#include "gradient_boosting/gradient_boosting_regressor.hpp"
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

using namespace std;

// -------------------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ --------------------------

double compute_rmse(const vector<double>& predictions, const vector<double>& actual) {
    if (predictions.empty() || predictions.size() != actual.size()) return 0.0;
    double sum_sq = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        double error = predictions[i] - actual[i];
        sum_sq += error * error;
    }
    return sqrt(sum_sq / predictions.size());
}

double compute_r2(const vector<double>& predictions, const vector<double>& actual) {
    if (predictions.empty()) return 0.0;
    double mean_actual = accumulate(actual.begin(), actual.end(), 0.0) / actual.size();
    double ss_res = 0.0, ss_tot = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        ss_res += (predictions[i] - actual[i]) * (predictions[i] - actual[i]);
        ss_tot += (actual[i] - mean_actual) * (actual[i] - mean_actual);
    }
    return 1.0 - ss_res / ss_tot;
}

// -------------------------- ЗАГРУЗКА ДАННЫХ --------------------------

bool load_employee_data(vector<vector<double>>& X, vector<double>& y) {
    const string DATA_PATH = "../data/employee_salary_encoded.csv";

    ifstream file(DATA_PATH);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << DATA_PATH << endl;
        return false;
    }

    string line;
    bool is_header = true;
    size_t loaded = 0;

    while (getline(file, line)) {
        if (is_header) {
            is_header = false;
            continue;
        }

        if (line.empty()) continue;

        stringstream ss(line);
        string value;
        vector<double> features;
        int col = 0;
        double salary = 0.0;

        while (getline(ss, value, ',')) {
            if (value.empty()) continue;

            try {
                double num = stod(value);
                if (col < 7) {
                    features.push_back(num);
                } else if (col == 7) {
                    salary = num;
                }
            } catch (...) {
                break;
            }
            col++;
        }

        if (features.size() == 7 && salary > 0) {
            X.push_back(features);
            y.push_back(salary);
            loaded++;
        }
    }

    file.close();
    cout << "Loaded " << loaded << " samples, 7 features" << endl;
    return loaded > 0;
}

// Разделение на train/test (80/20)
void train_test_split(const vector<vector<double>>& X, const vector<double>& y,
                     vector<vector<double>>& X_train, vector<double>& y_train,
                     vector<vector<double>>& X_test, vector<double>& y_test,
                     double test_ratio = 0.2, unsigned seed = 42) {

    mt19937 rng(seed);
    vector<size_t> indices(X.size());
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), rng);

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

// -------------------------- ИНТЕРАКТИВНОЕ ПРЕДСКАЗАНИЕ --------------------------

void print_mapping() {
    cout << "\n========================================" << endl;
    cout << "     CATEGORY MAPPING REFERENCE" << endl;
    cout << "========================================" << endl;
    cout << "1. Age (years): любое число (например, 25, 30, 45)" << endl;
    cout << endl;
    cout << "2. Gender:" << endl;
    cout << "   0 = Male" << endl;
    cout << "   1 = Female" << endl;
    cout << endl;
    cout << "3. Department:" << endl;
    cout << "   0 = Engineering" << endl;
    cout << "   1 = HR" << endl;
    cout << "   2 = Marketing" << endl;
    cout << "   3 = Sales" << endl;
    cout << "   4 = Finance" << endl;
    cout << endl;
    cout << "4. Job Title:" << endl;
    cout << "   0 = Analyst" << endl;
    cout << "   1 = Engineer" << endl;
    cout << "   2 = Manager" << endl;
    cout << "   3 = Director" << endl;
    cout << "   4 = Executive" << endl;
    cout << "   5 = Intern" << endl;
    cout << endl;
    cout << "5. Experience Years: любое число (например, 2, 5, 10, 15)" << endl;
    cout << endl;
    cout << "6. Education Level:" << endl;
    cout << "   0 = High School" << endl;
    cout << "   1 = Bachelor" << endl;
    cout << "   2 = Master" << endl;
    cout << "   3 = PhD" << endl;
    cout << endl;
    cout << "7. Location:" << endl;
    cout << "   0 = New York" << endl;
    cout << "   1 = San Francisco" << endl;
    cout << "   2 = Austin" << endl;
    cout << "   3 = Seattle" << endl;
    cout << "========================================" << endl;
}

double interactive_prediction(GradientBoostingRegressor& model) {
    print_mapping();

    cout << "\n========== ENTER EMPLOYEE DATA ==========" << endl;

    double age;
    cout << "Enter Age (years): ";
    cin >> age;

    double gender;
    cout << "Enter Gender (0=Male, 1=Female): ";
    cin >> gender;

    double department;
    cout << "Enter Department (0=Engineering, 1=HR, 2=Marketing, 3=Sales, 4=Finance): ";
    cin >> department;

    double job_title;
    cout << "Enter Job Title (0=Analyst, 1=Engineer, 2=Manager, 3=Director, 4=Executive, 5=Intern): ";
    cin >> job_title;

    double experience;
    cout << "Enter Experience Years: ";
    cin >> experience;

    double education;
    cout << "Enter Education Level (0=High School, 1=Bachelor, 2=Master, 3=PhD): ";
    cin >> education;

    double location;
    cout << "Enter Location (0=New York, 1=San Francisco, 2=Austin, 3=Seattle): ";
    cin >> location;

    vector<double> features = {age, gender, department, job_title, experience, education, location};

    return model.predict(features);
}

// -------------------------- MAIN --------------------------

int main() {
    cout << "==================================================" << endl;
    cout << "     EMPLOYEE SALARY PREDICTION SYSTEM" << endl;
    cout << "     Using Gradient Boosting Regressor" << endl;
    cout << "==================================================" << endl << endl;

    // 1. Загрузка данных
    cout << "[1] Loading employee salary dataset..." << endl;
    vector<vector<double>> X;
    vector<double> y;

    if (!load_employee_data(X, y)) {
        cerr << "Failed to load data. Exiting." << endl;
        return 1;
    }

    cout << "    Total samples: " << X.size() << endl;
    cout << "    Features: Age, Gender, Department, Job_Title, Experience_Years, Education_Level, Location" << endl;
    cout << "    Target: Salary (USD)" << endl << endl;

    // 2. Разделение на train/test (80/20)
    cout << "[2] Splitting data into train/test sets..." << endl;
    vector<vector<double>> X_train, X_test;
    vector<double> y_train, y_test;
    train_test_split(X, y, X_train, y_train, X_test, y_test, 0.2, 42);

    cout << "    Train samples: " << X_train.size() << endl;
    cout << "    Test samples: " << X_test.size() << endl << endl;

    // 3. Обучение модели
    cout << "[3] Training Gradient Boosting Regressor..." << endl;
    cout << "    Parameters: n_estimators=150, learning_rate=0.08, max_depth=5" << endl;

    auto start = chrono::high_resolution_clock::now();

    GradientBoostingRegressor gb(150, 0.08, 5, 5, 2, "mse");
    gb.fit(X_train, y_train);

    auto end = chrono::high_resolution_clock::now();
    double train_time = chrono::duration<double, milli>(end - start).count();
    cout << "    Training completed in " << train_time << " ms" << endl << endl;

    // 4. Оценка модели на тестовых данных
    cout << "[4] Evaluating model on test set..." << endl;
    vector<double> y_pred;
    y_pred.reserve(X_test.size());

    for (const auto& x : X_test) {
        y_pred.push_back(gb.predict(x));
    }

    double rmse = compute_rmse(y_pred, y_test);
    double r2 = compute_r2(y_pred, y_test);

    cout << fixed << setprecision(2);
    cout << "    RMSE: $" << rmse << endl;
    cout << "    R² Score: " << r2 << endl << endl;

    // 5. Примеры предсказаний
    cout << "[5] Sample predictions (first 10 test samples):" << endl;
    cout << "    Actual       Predicted    Error" << endl;
    cout << "    ---------------------------------" << endl;
    for (size_t i = 0; i < min(size_t(10), y_test.size()); ++i) {
        cout << "    $" << setw(10) << y_test[i]
             << "   $" << setw(10) << y_pred[i]
             << "   $" << setw(8) << abs(y_test[i] - y_pred[i]) << endl;
    }
    cout << endl;

    // 6. Baseline сравнение
    double mean_salary = accumulate(y_train.begin(), y_train.end(), 0.0) / y_train.size();
    vector<double> baseline_pred(y_test.size(), mean_salary);
    double baseline_rmse = compute_rmse(baseline_pred, y_test);

    cout << "[6] Baseline comparison:" << endl;
    cout << "    Baseline (predicting mean $" << mean_salary << "): RMSE = $" << baseline_rmse << endl;
    cout << "    Gradient Boosting improvement: " << (1 - rmse / baseline_rmse) * 100 << "%" << endl << endl;

    // 7. Интерактивное предсказание
    cout << "[7] Interactive salary prediction" << endl;
    cout << "==================================================" << endl;

    char choice;
    do {
        double salary = interactive_prediction(gb);
        double salary_rub = (salary / 12.0) * 71.0;
        cout << "\n==================================================" << endl;
        cout << "*** PREDICTED ANNUAL SALARY: $" << fixed << setprecision(2) << salary
        << " (~" << salary_rub << " rub/month) " << " ***" << endl;
        cout << "==================================================" << endl;

        cout << "\nPredict another employee? (y/n): ";
        cin >> choice;
    } while (choice == 'y' || choice == 'Y');

    cout << "\nThank you for using Employee Salary Prediction System!" << endl;

    return 0;
}
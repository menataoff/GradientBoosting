cmake_minimum_required(VERSION 3.14)
project(GradientBoosting)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
        decision_tree
        GIT_REPOSITORY https://github.com/menataoff/DecisionTree.git
        GIT_TAG main
)
FetchContent_MakeAvailable(decision_tree)

add_library(gradient_boosting INTERFACE)

target_include_directories(gradient_boosting INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(gradient_boosting INTERFACE decision_tree_core)

add_executable(test_boston tests/regressor_boston.cpp)
target_link_libraries(test_boston PRIVATE gradient_boosting)

add_executable(test_wine tests/classifier_wine.cpp)
target_link_libraries(test_wine PRIVATE gradient_boosting)

set_target_properties(test_boston test_wine PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

add_executable(gradient_boosting_example main.cpp)
target_link_libraries(gradient_boosting_example PRIVATE gradient_boosting)
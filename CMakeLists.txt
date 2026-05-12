cmake_minimum_required(VERSION 3.14)
project(GradientBoosting)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
        decision_tree
        GIT_REPOSITORY https://github.com/menataoff/DecisionTree.git
        GIT_TAG main
)
FetchContent_MakeAvailable(decision_tree)

add_executable(gradient_boosting main.cpp)
target_link_libraries(gradient_boosting PRIVATE decision_tree_core)
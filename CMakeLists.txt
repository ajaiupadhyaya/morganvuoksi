cmake_minimum_required(VERSION 3.10)
project(quant_finance)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# Add backtesting engine
pybind11_add_module(backtest_engine src/backtesting/engine.cpp)

# Set include directories
target_include_directories(backtest_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Python_INCLUDE_DIRS}
)

# Set compile options
if(MSVC)
    target_compile_options(backtest_engine PRIVATE /W4)
else()
    target_compile_options(backtest_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Install
install(TARGETS backtest_engine DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/src/backtesting) 
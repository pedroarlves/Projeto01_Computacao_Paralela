#!/bin/bash
################################################################################
# run_quick_test.sh
# 
# Quick test script - runs a single configuration for testing purposes
################################################################################

# Default configuration
DATASET="dataset/mnist_train.csv"
K=5
MAX_ITER=50
THREADS=4

echo "=========================================="
echo "Quick K-Means Test"
echo "=========================================="
echo ""

# Clean and build
echo "Building..."
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "CMake configuration failed!"
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    exit 1
fi

cmake --build build > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Build failed!"
    cmake --build build
    exit 1
fi

if [ ! -f "build/main" ]; then
    echo "Error: Build failed!"
    exit 1
fi

echo "Build complete!"
echo ""

# Run OpenMP version
echo "Testing OpenMP version with $THREADS threads..."
echo "Command: OMP_NUM_THREADS=$THREADS ./build/main $DATASET $K $MAX_ITER"
echo ""
OMP_NUM_THREADS=$THREADS ./build/main $DATASET $K $MAX_ITER

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="

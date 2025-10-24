#!/bin/bash
################################################################################
# run_benchmarks.sh
# 
# Automated benchmark script for K-Means parallel implementation
# Compiles both OpenMP and MPI+OpenMP versions and runs all required tests
# Results are saved to a structured CSV file
################################################################################

# Configuration variables
DATASET="dataset/mnist_train.csv"
K_CLUSTERS=15          # Number of clusters
MAX_ITERATIONS=100     # Maximum iterations
SEED=42               # Random seed for reproducibility
OUTPUT_CSV="benchmark_results.csv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}K-Means Parallel Benchmark Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -k|--clusters)
            K_CLUSTERS="$2"
            shift 2
            ;;
        -m|--max-iter)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_CSV="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -d, --dataset PATH     Dataset path (default: dataset/mnist_train.csv)"
            echo "  -k, --clusters NUM     Number of clusters (default: 15)"
            echo "  -m, --max-iter NUM     Maximum iterations (default: 100)"
            echo "  -s, --seed NUM         Random seed (default: 42)"
            echo "  -o, --output FILE      Output CSV file (default: benchmark_results.csv)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 -k 10 -m 100 -o results.csv"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Display configuration
echo -e "${YELLOW}Configuration:${NC}"
echo "  Dataset:        $DATASET"
echo "  Clusters (K):   $K_CLUSTERS"
echo "  Max iterations: $MAX_ITERATIONS"
echo "  Seed:           $SEED"
echo "  Output file:    $OUTPUT_CSV"
echo ""

# Check if dataset exists
if [ ! -f "$DATASET" ]; then
    echo -e "${RED}Error: Dataset file '$DATASET' not found!${NC}"
    exit 1
fi

# Step 1: Clean previous build
echo -e "${YELLOW}Step 1: Cleaning previous build...${NC}"
rm -rf build
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Clean complete${NC}"
else
    echo -e "${RED}✗ Clean failed${NC}"
    exit 1
fi
echo ""

# Step 2: Configure with CMake
echo -e "${YELLOW}Step 2: Configuring with CMake...${NC}"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Configuration complete${NC}"
else
    echo -e "${RED}✗ Configuration failed${NC}"
    echo -e "${RED}Running cmake again with verbose output:${NC}"
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    exit 1
fi
echo ""

# Step 3: Build
echo -e "${YELLOW}Step 3: Building executables...${NC}"
cmake --build build > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build complete${NC}"
    echo -e "${GREEN}  - build/main (OpenMP version)${NC}"
    echo -e "${GREEN}  - build/main_hybrid (MPI+OpenMP version)${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    echo -e "${RED}Running build again with verbose output:${NC}"
    cmake --build build
    exit 1
fi
echo ""

# Check if executables exist
if [ ! -f "build/main" ] || [ ! -f "build/main_hybrid" ]; then
    echo -e "${RED}Error: Executables not found after build!${NC}"
    exit 1
fi

# Step 4: Create CSV header
echo -e "${YELLOW}Step 4: Initializing results file...${NC}"
echo "version,configuration,num_processes,num_threads,total_cores,k_clusters,max_iterations,execution_time_s,iterations_completed,dataset" > "$OUTPUT_CSV"
echo -e "${GREEN}✓ Created $OUTPUT_CSV${NC}"
echo ""

# Function to extract execution time from output
extract_time() {
    local output="$1"
    echo "$output" | grep -oP "Elapsed time.*:\s*\K[0-9]+\.[0-9]+" | head -1
}

# Function to extract iterations from output
extract_iterations() {
    local output="$1"
    # Try "Converged in X iterations" first (MPI version)
    local iters=$(echo "$output" | grep -oP "Converged in \K[0-9]+" | head -1)
    if [ -z "$iters" ]; then
        # Try "KMeans finished in X iterations" (OpenMP version)
        iters=$(echo "$output" | grep -oP "KMeans finished in \K[0-9]+" | head -1)
    fi
    echo "$iters"
}

# Function to run a test and record results
run_test() {
    local version="$1"
    local config_name="$2"
    local num_procs="$3"
    local num_threads="$4"
    local command="$5"
    
    local total_cores=$((num_procs * num_threads))
    
    echo -e "${BLUE}Running: $config_name${NC}"
    echo "  Command: $command"
    
    # Run the command and capture output
    local output=$(eval "$command" 2>&1)
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo -e "${RED}  ✗ Failed (exit code: $exit_code)${NC}"
        echo "$version,$config_name,$num_procs,$num_threads,$total_cores,$K_CLUSTERS,$MAX_ITERATIONS,ERROR,ERROR,$DATASET" >> "$OUTPUT_CSV"
        return 1
    fi
    
    # Extract metrics
    local exec_time=$(extract_time "$output")
    local iterations=$(extract_iterations "$output")
    
    if [ -z "$exec_time" ]; then
        exec_time="N/A"
    fi
    
    if [ -z "$iterations" ]; then
        iterations="N/A"
    fi
    
    echo -e "${GREEN}  ✓ Completed in ${exec_time}s (${iterations} iterations)${NC}"
    
    # Save to CSV
    echo "$version,$config_name,$num_procs,$num_threads,$total_cores,$K_CLUSTERS,$MAX_ITERATIONS,$exec_time,$iterations,$DATASET" >> "$OUTPUT_CSV"
    
    # Small delay between tests
    sleep 1
}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running OpenMP Tests${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# OpenMP tests (1, 2, 4, 8 threads)
run_test "OpenMP" "1_thread" 1 1 "OMP_NUM_THREADS=1 ./build/main $DATASET $K_CLUSTERS $MAX_ITERATIONS $SEED"
run_test "OpenMP" "2_threads" 1 2 "OMP_NUM_THREADS=2 ./build/main $DATASET $K_CLUSTERS $MAX_ITERATIONS $SEED"
run_test "OpenMP" "4_threads" 1 4 "OMP_NUM_THREADS=4 ./build/main $DATASET $K_CLUSTERS $MAX_ITERATIONS $SEED"
run_test "OpenMP" "8_threads" 1 8 "OMP_NUM_THREADS=8 ./build/main $DATASET $K_CLUSTERS $MAX_ITERATIONS $SEED"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running MPI+OpenMP Hybrid Tests${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# MPI+OpenMP hybrid tests
run_test "MPI+OpenMP" "1proc_4threads" 1 4 "OMP_NUM_THREADS=4 mpirun -np 1 ./build/main_hybrid $DATASET $K_CLUSTERS $MAX_ITERATIONS $SEED"
run_test "MPI+OpenMP" "2proc_2threads" 2 2 "OMP_NUM_THREADS=2 mpirun -np 2 ./build/main_hybrid $DATASET $K_CLUSTERS $MAX_ITERATIONS $SEED"
run_test "MPI+OpenMP" "4proc_1thread" 4 1 "OMP_NUM_THREADS=1 mpirun -np 4 ./build/main_hybrid $DATASET $K_CLUSTERS $MAX_ITERATIONS $SEED"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Benchmark Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Display results summary
echo -e "${YELLOW}Results Summary:${NC}"
echo ""
column -t -s',' "$OUTPUT_CSV" | head -20

echo ""
echo -e "${GREEN}Full results saved to: $OUTPUT_CSV${NC}"
echo ""

# Calculate and display speedups if baseline exists
baseline_time=$(awk -F',' 'NR==2 {print $8}' "$OUTPUT_CSV")
if [ ! -z "$baseline_time" ] && [ "$baseline_time" != "ERROR" ] && [ "$baseline_time" != "N/A" ]; then
    echo -e "${YELLOW}Speedup Analysis (baseline: 1 thread = ${baseline_time}s):${NC}"
    awk -F',' -v baseline="$baseline_time" '
        NR>1 && $8 != "ERROR" && $8 != "N/A" {
            speedup = baseline / $8
            printf "  %s: %.2fx speedup\n", $2, speedup
        }
    ' "$OUTPUT_CSV"
    echo ""
fi

echo -e "${GREEN}✓ All tests completed successfully!${NC}"

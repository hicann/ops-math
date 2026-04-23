#!/bin/bash
# reduce_mean_with_count UT test execution script

set -e

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
CLEAN_BUILD=true
VERBOSE=""
RUN_OP_HOST=true

# ============================================================================
# Help
# ============================================================================
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "UT test: test Tiling, InferShape logic"
    echo ""
    echo "Options:"
    echo "  -c, --clean      Clean build dir (default: true)"
    echo "  -v, --verbose    Show verbose output"
    echo "  --ophost         Only run op_host tests"
    echo "  -h, --help       Show help"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 --ophost"
    echo "  $0 -c false"
}

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--clean)
            if [ "$2" = "false" ]; then
                CLEAN_BUILD=false
                shift 2
            else
                CLEAN_BUILD=true
                shift
            fi
            ;;
        -v|--verbose)
            VERBOSE="VERBOSE=1"
            shift
            ;;
        --ophost)
            RUN_OP_HOST=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            show_help
            exit 1
            ;;
    esac
done

# ============================================================================
# Display config
# ============================================================================
echo "========================================"
echo "reduce_mean_with_count UT test"
echo "========================================"
echo "Clean build: ${CLEAN_BUILD}"
echo "Working dir: ${SCRIPT_DIR}"
echo "Test scope: op_host=${RUN_OP_HOST}"
echo "========================================"
echo ""

# ============================================================================
# Check dependencies
# ============================================================================
echo "Checking dependencies..."

if ! command -v cmake &> /dev/null; then
    echo "Error: cmake not found"
    exit 1
fi

if ! command -v g++ &> /dev/null; then
    echo "Error: g++ not found"
    exit 1
fi

echo "Dependencies OK"
echo ""

# ============================================================================
# Set environment variables
# ============================================================================
echo "Setting environment variables..."

if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "Warning: ASCEND_HOME_PATH not set"
    echo "Using default: /home/zhongyao/cann-9.0.0"
    export ASCEND_HOME_PATH=/home/zhongyao/cann-9.0.0
fi

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}
echo "LD_LIBRARY_PATH: ${ASCEND_HOME_PATH}/lib64"

echo "Environment set"
echo ""

# ============================================================================
# Clean build directory
# ============================================================================
if [ "$CLEAN_BUILD" = true ]; then
    echo "Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
fi

# ============================================================================
# Create build directory
# ============================================================================
echo "Creating build directory..."
mkdir -p "${BUILD_DIR}"

# ============================================================================
# CMake configure
# ============================================================================
echo ""
echo "CMake configure..."
cd "${BUILD_DIR}"

cmake .. ${VERBOSE}

if [ $? -ne 0 ]; then
    echo "Error: CMake configure failed"
    exit 1
fi

# ============================================================================
# Build
# ============================================================================
echo ""
echo "Building UT tests..."
make -j$(nproc) ${VERBOSE}

if [ $? -ne 0 ]; then
    echo "Error: Build failed"
    exit 1
fi

echo "Build succeeded"
echo ""

# ============================================================================
# Run UT tests
# ============================================================================
echo "========================================"
echo "Running UT tests"
echo "========================================"

FAILED_TESTS=()
PASSED_TESTS=()

if [ "$RUN_OP_HOST" = true ]; then
    echo ""
    echo ">>> Running op_host tests <<<"
    echo ""
    cd "${BUILD_DIR}/op_host"

    if [ ! -f "./reduce_mean_with_count_op_host_ut" ]; then
        echo "Error: op_host UT executable not found"
        FAILED_TESTS+=("op_host")
    else
        if ./reduce_mean_with_count_op_host_ut; then
            PASSED_TESTS+=("op_host")
            echo "[PASS] op_host tests passed"
        else
            FAILED_TESTS+=("op_host")
            echo "[FAIL] op_host tests failed"
        fi
    fi
fi

# ============================================================================
# Results summary
# ============================================================================
echo ""
echo "========================================"
echo "Test Results Summary"
echo "========================================"
echo ""

if [ ${#PASSED_TESTS[@]} -gt 0 ]; then
    echo "Passed:"
    for test in "${PASSED_TESTS[@]}"; do
        echo "  - ${test}"
    done
fi

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo ""
    echo "Failed:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - ${test}"
    done
    echo ""
    echo "========================================"
    echo "Result: FAIL"
    echo "========================================"
    exit 1
else
    echo ""
    echo "========================================"
    echo "Result: PASS"
    echo "========================================"
    exit 0
fi

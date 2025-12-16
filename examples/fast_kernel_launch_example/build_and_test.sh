#!/bin/bash

set -e
cd examples/fast_kernel_launch_example

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Build the project
echo "Building the project..."
python3 setup.py clean
python3 -m build --wheel --no-isolation
python3 -m pip install dist/*.whl --force-reinstall --no-deps

# Run tests
echo "Running tests..."
python3 ascend_ops/csrc/is_finite/test/test_isfinite.py
python3 ascend_ops/csrc/add/test/test_add.py
echo "execute samples success"

#!/usr/bin/env bash

# -----------------------------------------
# tests.sh - Run all pylmtic tests
# -----------------------------------------

# Exit on any error
set -e

# Set PYTHONPATH to current directory so pylmtic can be imported
export PYTHONPATH=$(pwd)
echo "PYTHONPATH set to $PYTHONPATH"

# Optional: verbose pytest output
PYTEST_OPTS="-v"

# Run all tests
echo "Running all tests..."
pytest $PYTEST_OPTS tests/

# Run a specific test file (uncomment if needed)
# pytest $PYTEST_OPTS tests/test_core.py

echo "All tests finished!"

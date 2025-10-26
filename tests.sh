#!/usr/bin/env bash

# -----------------------------------------
# tests.sh - Run all or specific pylmtic tests
# -----------------------------------------

# Exit on any error
set -e

# Set PYTHONPATH to current directory so pylmtic can be imported
export PYTHONPATH=$(pwd)
echo "PYTHONPATH set to $PYTHONPATH"

# Optional: verbose pytest output
PYTEST_OPTS="-v"

# Default: run all tests
TEST_PATH="tests/"

# Parse options
while getopts ":f:" opt; do
  case ${opt} in
    f )
      # If -f is provided, run only the specified test file
      TEST_PATH="$OPTARG"
      ;;
    \? )
      echo "Usage: $0 [-f test_file]"
      exit 1
      ;;
  esac
done

# Run tests
echo "Running tests in: $TEST_PATH"
pytest $PYTEST_OPTS "$TEST_PATH"

echo "All tests finished!"

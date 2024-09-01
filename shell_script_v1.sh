#!/bin/bash

# Define the Python file you want to run
PYTHON_FILE="/home/ia23938/Documents/GitHub/ValueSystemsAggregation/experiment_runner.py"

# Run the Python file 5 times
for i in {1..4}
do
    echo "Running $PYTHON_FILE - Iteration $i"
    python3 $PYTHON_FILE
done
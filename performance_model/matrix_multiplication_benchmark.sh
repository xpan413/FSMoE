#!/bin/bash

# Constants
MODEL=1024
HIDDEN=1024

# Loop over the specified range of tokens
for i in {1..12}; do
    TOKENS=$((i * 512))
    echo "Running test with tokens=$TOKENS, model=$MODEL, hidden=$HIDDEN"
    python matrix_multiplication_benchmark.py --tokens "$TOKENS" --model "$MODEL" --hidden "$HIDDEN"
done

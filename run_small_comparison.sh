#!/bin/bash

# Create directories for outputs
mkdir -p experiments
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="experiments/small_test_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Download data using the cached script
echo "Downloading dataset..."
python3 data/cached_fineweb10B2.py

# Arguments without quotes that would break command expansion
COMMON_ARGS=(
    --input_bin data/fineweb10B/fineweb_train_\*.bin
    --input_val_bin data/fineweb10B/fineweb_val_000000.bin
    --model d12
    --batch_size 8
    --sequence_length 256
    --num_iterations 875
    --val_loss_every 32
    --weight_decay 0.1
    --warmup_iters 60
    --warmdown_iters 60
    --learning_rate .0018
    --accumulation 4
)

echo "Starting small-scale comparison at $(date)"
echo "Outputs will be saved to $OUTPUT_DIR"

# Run ADOPT version
{
    echo "=== SOAP+ADOPT Run Start: $(date) ==="
    echo "GPU memory before run:"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    
    STARTTIME=$(date +%s)
    torchrun --standalone --nproc_per_node=1 train_gpt2-adopt.py "${COMMON_ARGS[@]}"
    ENDTIME=$(date +%s)
    
    echo "GPU memory after run:"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    echo "Run completed in $((ENDTIME - STARTTIME)) seconds"
} 2>&1 | tee "$OUTPUT_DIR/adopt_run.log"

echo "Sleeping for 30 seconds between runs..."
sleep 30

# Run original version
{
    echo "=== Original SOAP Run Start: $(date) ==="
    echo "GPU memory before run:"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    
    STARTTIME=$(date +%s)
    torchrun --standalone --nproc_per_node=1 train_gpt2.py "${COMMON_ARGS[@]}"
    ENDTIME=$(date +%s)
    
    echo "GPU memory after run:"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    echo "Run completed in $((ENDTIME - STARTTIME)) seconds"
} 2>&1 | tee "$OUTPUT_DIR/original_run.log"

# Generate summary
{
    echo "=== Small Scale Test Summary (1/32 scale) ==="
    echo "Experiment completed at: $(date)"
    echo ""
    echo "=== Original SOAP ==="
    echo "Final validation loss: $(grep "val loss" "$OUTPUT_DIR/original_run.log" | tail -n 1)"
    echo "Runtime: $(grep "Run completed in" "$OUTPUT_DIR/original_run.log" | cut -d' ' -f4) seconds"
    echo "Training losses (last 5):"
    grep "train loss" "$OUTPUT_DIR/original_run.log" | tail -n 5
    echo "Peak GPU memory: $(grep "peak memory consumption" "$OUTPUT_DIR/original_run.log" | tail -n 1)"
    echo "Tokens/second: $(grep "tok/s" "$OUTPUT_DIR/original_run.log" | tail -n 5)"
    echo ""
    echo "=== SOAP+ADOPT ==="
    echo "Final validation loss: $(grep "val loss" "$OUTPUT_DIR/adopt_run.log" | tail -n 1)"
    echo "Runtime: $(grep "Run completed in" "$OUTPUT_DIR/adopt_run.log" | cut -d' ' -f4) seconds"
    echo "Training losses (last 5):"
    grep "train loss" "$OUTPUT_DIR/adopt_run.log" | tail -n 5
    echo "Peak GPU memory: $(grep "peak memory consumption" "$OUTPUT_DIR/adopt_run.log" | tail -n 1)"
    echo "Tokens/second: $(grep "tok/s" "$OUTPUT_DIR/adopt_run.log" | tail -n 5)"
} > "$OUTPUT_DIR/summary.txt"

echo "Experiment complete! Check $OUTPUT_DIR for logs and summary."

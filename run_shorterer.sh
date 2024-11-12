#!/bin/bash

# Download just one training shard and validation shard
python3 - << EOF
import os
from huggingface_hub import hf_hub_download

def get(fname):
    local_dir = os.path.join(os.path.dirname(__file__), 'data/fineweb10B')
    os.makedirs(local_dir, exist_ok=True)
    if not os.path.exists(os.path.join(local_dir, fname)):
        hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", filename=fname,
                        repo_type="dataset", local_dir=local_dir)

# Get just one training shard and the validation shard
get("fineweb_train_000001.bin")
get("fineweb_val_000000.bin")
EOF

# Run training with minimal settings
torchrun --standalone --nproc_per_node=1 train_gpt2.py \
    --input_bin "data/fineweb10B/fineweb_train_000001.bin" \
    --input_val_bin "data/fineweb10B/fineweb_val_000000.bin" \
    --model d12 \
    --batch_size 4 \
    --sequence_length 64 \
    --val_loss_every 10 \
    --num_iterations 100 \
    --learning_rate .001 \
    --accumulation 1

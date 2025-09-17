#!/bin/bash
# Launch script for distributed SAE training on 8 GPUs

# Set environment variables for better performance
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=0

# Optional: Set master address if needed (usually automatic)
# export MASTER_ADDR=localhost
# export MASTER_PORT=29500

echo "Starting distributed training on 8 GPUs..."
echo "Data directory: /home/ec2-user/SageMaker/InterPLM/data/sharded_uniprot"
echo "----------------------------------------"

# Launch with torchrun
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    distributed_train_sae.py \
    --data-dir /home/ec2-user/SageMaker/InterPLM/data/sharded_uniprot \
    --n-shards 8 \
    --n-feats 10240 \
    --steps 5000 \
    --checkpoint-dir checkpoints_distributed_new_loss \
    --checkpoint-freq 1000 \
    --log-freq 50 \
    --save-final final_distributed_sae.pt

echo "Training complete!"
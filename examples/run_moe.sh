#!/bin/bash

# Runs the "340M" parameter model (Bert - Large)
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=$1 #<Specify path>
TENSORBOARD_LOGS_PATH=$2 #<Specify path>
DATA_PATH=/workspace/pxl/Megatron-LM/alpaca

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --num-layers 2 
    --hidden-size 1024 
    --num-attention-heads 16 
    --seq-length 512 
    --max-position-embeddings 512 
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --no-position-embedding
    --rotary-base 1000000
    --disable-bias-linear
)

MOE_ARGS=(
    --num-experts 4
    --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-token-dispatcher-type alltoall
    --moe-expert-capacity-factor 1.2
    --moe-pad-expert-input-to-capacity
    --pipeline-degree 2
    --bp-pipeline-degree 2
)

TRAINING_ARGS=(
    --micro-batch-size 32 
    --global-batch-size 128 
    --train-iters 1000000 
    --weight-decay 1e-2 
    --clip-grad 1.0 
    --bf16
    --lr 0.0001
    --lr-decay-iters 990000 
    --lr-decay-style linear 
    --min-lr 1.0e-5 
    --weight-decay 1e-2 
    --lr-warmup-fraction .01 
    --clip-grad 1.0 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 2 
	--pipeline-model-parallel-size 1 
    --expert-model-parallel-size 2
    --sequence-parallel
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file /workspace/pxl/Megatron-LM/gpt2-vocab.json
    --merge-file /workspace/pxl/Megatron-LM/gpt2-merges.txt
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
    --profile
    --use-pytorch-profiler
    --profile-ranks 0
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

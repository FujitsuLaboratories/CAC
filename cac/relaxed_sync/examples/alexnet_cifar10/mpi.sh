#!/usr/bin/env bash
# COPYRIGHT Fujitsu Limited 2021

DIR=${HOME}/CAC/cac/relaxed_sync/examples/alexnet_cifar10
GPUS_PER_NODE=2
MASTER_ADDR='172.20.51.33'
MASTER_PORT='29500'

SINGLE_GPU=0 # 0: OFF (MULTIPLE GPU), 1: ON (SINGLE GPU)

USE_RELAXED_SYNC=1 # 0: OFF, 1: ON
USE_SIMULATE=1 # 0: OFF, 1: ON

EPOCH=30
BATCH=64
LR=0.005

OP="-e ${EPOCH} -lr ${LR} -b ${BATCH}"
if [ "$SINGLE_GPU" -eq 1 ]; then
	OP="${OP} -S"
fi
if [ "$USE_RELAXED_SYNC" -eq 1 ]; then
	OP="${OP} -r"
fi
if [ "$USE_SIMULATE" -eq 1 ]; then
	OP="${OP} -s"
fi

rank=`env | grep OMPI_COMM_WORLD_RANK | awk -F= '{print $2}'`
size=`env | grep OMPI_COMM_WORLD_SIZE | awk -F= '{print $2}'`
echo "rank: $rank, size: $size"
if [ ${size} -lt 1 ]; then
  echo "comm_world_size < 1 (${size}). exit."
  exit 1
fi


echo "python -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --nnodes=${size} --node_rank=${rank} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} main.py ${OP}"
python -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --nnodes=${size} --node_rank=${rank} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} main.py ${OP}

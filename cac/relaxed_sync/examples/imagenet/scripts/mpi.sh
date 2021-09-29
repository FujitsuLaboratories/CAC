#!/bin/bash
# COPYRIGHT Fujitsu Limited 2021

if [ $# -le 1 ]; then
  echo "$0 [nnodes] [arch] [bs] [learning_rate] [opt_level] [base_data_dir]"
  exit 1
fi

base_data_dir=$6

args=()
for ((i=${#BASH_ARGV[@]}-2; i>=0; i--)); do
  args+=(${BASH_ARGV[$i]})
done

echo "run pytorch imagenet"
mpirun -n $1 ./pytorch_imagenet.sh ${base_data_dir}/ ${args[*]}

echo "finish pytorch imagenet"

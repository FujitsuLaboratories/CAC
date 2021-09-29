#!/bin/bash

#$ -l rt_F=2
#$ -l h_rt=12:00:00
#$ -j y
#$ -cwd

#if [ $# -le 1 ]; then
if [ $# -le 0 ]; then
  echo "$0 [nprocs] [copy_to_local_dir: set to 1 if copy, 0 otherwise] [relax_threshold(default:1)] [relax_interval_iter(default:-1)] [skip_rank_interval(default:-1)] [num_sleep_procs(default:0)] [sync(default:0)] [stick_to_shard(default:0)] [checkpoint_path(default:None)]"
  exit 1
fi

#source /etc/profile.d/modules.sh
#. ~/.bashrc

#base_data_dir=/groups1/gca50115/forPyTorch/data
base_data_dir=/raid/aibench/forPyTorch

args=()
for ((i=${#BASH_ARGV[@]}-3; i>=0; i--)); do
  args+=(${BASH_ARGV[$i]})
done

mpirun -n $1 ./example_imagenet.sh ${base_data_dir}/imagenet/ ${args[*]}

echo "finish pytorch imagenet"

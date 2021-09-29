#!/bin/bash

if [ $# -eq 0 ]; then
  echo "$0 [nprocs] [copy_to_local_dir: set to 1 if copy, 0 otherwise(default:1)] [relax_threshold(default:1)] [relax_interval_iter(default:-1)] [skip_rank_interval(default:-1)] [num_sleep_procs(default:0)] [sync(default:0)] [stick_to_shard(default:0)] [checkpoint_path(default:None)]"
  exit 1
fi

do_copy=1
if [ $# -ge 2 ]; then
  do_copy=$2
fi

args=()
for ((i=${#BASH_ARGV[@]}-3; i>=0; i--)); do
  args+=(${BASH_ARGV[$i]})
done

hosts="g02[0-1][0-9]|g022[0-3]"

#qsub -g gca50115 -l rt_F=$1 -l h_rt=3:00:00 -j y -cwd ./mpi.sh $1 ${do_copy} ${args[*]}
echo "qsub -g gca50115 -l rt_F=$1 -l h_rt=3:00:00 -l hostname=$hosts -j y -cwd ./mpi.sh $1 ${do_copy} ${args[*]}"
qsub -g gca50115 -l rt_F=$1 -l h_rt=3:00:00 -l hostname=$hosts -j y -cwd ./mpi.sh $1 ${do_copy} ${args[*]}

#!/bin/sh
# COPYRIGHT Fujitsu Limited 2021

# This is a PyTorch imagenet batch script

# request Bourne shell as shell for job
# #$ -S /bin/sh

#JOB_ID=001

script_dir=../
log_dir=./logs/${JOB_ID}

if [ $# -eq 0 ]; then
  echo "$0 [data_path or 'local'] [arch] [bs] [learning_rate] [opt_level] "
  echo "data is copied when 'local' is specified as the first argument"
  exit 1
fi

hostname

data_dir=$1
if [ $data_dir = "local" ]; then
  data_dir=${SGE_LOCALDIR}/imagenet/
  echo "use local dir in $data_dir"
fi

arch=$2
batch_size=$3
learning_rate=$4
opt_level=$5
echo "[DEBUG] 1 2 3 4 5"
echo "[DEBUG] $1 $2 $3 $4 $5"
echo "[DEBUG] arch: ${arch}, batch_size: ${batch_size}, learning_rate: ${learning_rate}, opt_level: ${opt_level}"


cur_dir=$(cd $(dirname $0);pwd)
ip_mask="10\.1\."
master_ip_file=${log_dir}/master_ip_${JOB_ID}

nproc_per_node=4
bs=128
#epoch=90
epoch=3

mkdir -p ${log_dir}

hostname

rank=`env | grep OMPI_COMM_WORLD_RANK | awk -F= '{print $2}'`
size=`env | grep OMPI_COMM_WORLD_SIZE | awk -F= '{print $2}'`
echo "rank: $rank, size: $size"

if [ ${size} -lt 1 ]; then
  echo "comm_world_size < 1 (${size}). exit."
  exit 1
fi

# write master ip to file
if [ ${rank} -eq 0 ]; then
  #master_ip=`/sbin/ifconfig | grep ${ip_mask} | awk '{print $2}'`
  master_ip="localhost"
  echo "master_ip: $master_ip"
  echo $master_ip > ${master_ip_file}
fi

sleep 3

# set master ip and port
master_ip=`cat ${master_ip_file}`
if [ -z ${master_ip} ]; then
  echo "master_ip is null. exit."
  exit 1
fi
echo "read master_ip: $master_ip"

date
date +%s

if [ ${rank} -eq 0 ]; then
  cp ${cur_dir}/*.sh ${log_dir}
fi
cd ${script_dir}
pwd

echo "$0 args $1 $2 $3"

set -x


echo "python -m torch.distributed.launch --nproc_per_node=${nproc_per_node} --nnodes=${size} --node_rank=${rank} --master_addr=${master_ip} --master_port=8888 main_amp.py -a ${arch} --b ${batch_size} --workers 20 --opt-level ${opt_level} --lr ${learning_rate} --epochs ${epoch} ${data_dir}"
python -m torch.distributed.launch --nproc_per_node=${nproc_per_node} --nnodes=${size} --node_rank=${rank} --master_addr=${master_ip} --master_port=8888 main_amp.py -a ${arch} --b ${batch_size} --workers 20 --opt-level ${opt_level} --lr ${learning_rate} --epochs ${epoch} ${data_dir}

set +x

date
date +%s

#!/bin/sh

# This is a PyTorch imagenet batch script

# request Bourne shell as shell for job
# #$ -S /bin/sh

if [ $# -eq 0 ]; then
  echo "$0 [data_path or 'local'] [relax_threshold(default:1)] [relax_interval_iter(default:-1)] [skip_rank_interval(default:-1)] [num_sleep_procs(default:0)] [sync(default:0)] [stick_to_shard(default:0)] [checkpoint_path(default:None)]"
  echo "data is copied when 'local' is specified as the first argument"
  exit 1
fi

hostname

data_dir=$1
if [ $data_dir = "local" ]; then
  data_dir=${SGE_LOCALDIR}/imagenet/
  echo "use local dir in $data_dir"
fi

relax_threshold=1
if [ $# -ge 2 ]; then
  relax_threshold=$2
fi

relax_interval_iter=-1
if [ $# -ge 3 ]; then
  relax_interval_iter=$3
fi

skip_rank_interval=-1
if [ $# -ge 4 ]; then
  skip_rank_interval=$4
fi

num_sleep_procs=0
if [ $# -ge 5 ]; then
  num_sleep_procs=$5
fi

if [ $# -ge 6 ]; then
  if [ $6 -gt 0 ]; then
    sync="--sync"
  fi
fi

if [ $# -ge 7 ]; then
  if [ $7 -gt 0 ]; then
    stick_to_shard="--stick-to-shard"
  fi
fi

if [ $# -ge 8 ]; then
  resume=$8
fi

user="acd13314za"
script_dir=/home/acd13314za/CAC_Lib/example/image_classification/
log_dir=/home/acd13314za/logs/${JOB_ID}
cur_dir=$(cd $(dirname $0);pwd)
ip_mask="10\.1\."
#ip_mask="172\.20\."
master_ip_file=${log_dir}/master_ip_${JOB_ID}

nproc_per_node=4
bs=128
#bs=256
#bs=64
epoch=90
#epoch=1
warmup_epochs=8
#warmup_epochs=25
print_freq=10
#print_freq=1

mkdir -p ${log_dir}

#
hostname

rank=`env | grep OMPI_COMM_WORLD_RANK | awk -F= '{print $2}'`
size=`env | grep OMPI_COMM_WORLD_SIZE | awk -F= '{print $2}'`
if [ -z $rank ]; then
  rank=0
fi
if [ -z $size ]; then
  size=1
fi
echo "rank: $rank, size: $size"

if [ ${size} -lt 1 ]; then
  echo "comm_world_size < 1 (${size}). exit."
  exit 1
fi

# write master ip to file
if [ ${rank} -eq 0 ]; then
  #master_ip=`/sbin/ifconfig | grep ${ip_mask} | awk '{print $2}' | awk -F ":" '{print $2}'`
  master_ip=`/sbin/ifconfig | grep ${ip_mask} | awk '{print $2}'`
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

if [ ! -z ${resume} ]; then
  epoch=120
  resume_arg="--resume ${resume}"
fi

echo "$0 args $1 $2 $3"

total_bs=`echo "$((${bs} * ${size} * ${nproc_per_node}))"`
lr=$(echo | awk "{print ${total_bs}*0.001}")
#lr=0.01
#lr=31.2
#export NCCL_ALGO=Ring
#export NCCL_PROTO=Simple
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG=VERSION

export MASTER_ADDR=${master_ip}
export MASTER_PORT=8888
export WORLD_SIZE=${size}
export RANK=${rank}
export LOCAL_RANK=${rank}

#export CAC_BRAKING_DISTANCE=300
#export CAC_STOP_LAYER_NUM=50,100
#export CAC_STOP_LAYER_ITR=10,20
#export CAC_VAR_START_ITR=5000
#export CAC_VAR_START_THR=0.95
#export CAC_VAR_MT_THR=0.96
#export CAC_VAR_MT_COUNT_THR=5
#export CAC_VAR_SLOPE_THR=0.98
#export CAC_VAR_SAMPLES=200

python example_imagenet.py ${data_dir} \
  --data-backend pytorch --raport-file raport.json -j5 -p ${print_freq} \
  --lr ${lr} --optimizer-batch-size ${total_bs} --warmup ${warmup_epochs} --arch resnet50 -c fanin --label-smoothing 0.1 \
  --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 --workspace ${log_dir} -b ${bs} \
  --amp --static-loss-scale 128 --epochs ${epoch} ${resume_arg} \
  --seed 0 ${sync} ${stick_to_shard}

#python example_imagenet.py ${data_dir} \
#  --data-backend pytorch -j5 -p ${print_freq} \
#  --lr ${lr} --warmup ${warmup_epochs} --arch alexnet \
#  --mom 0.9 --wd 5.0e-4 --workspace ${log_dir} -b ${bs} \
#  --epochs ${epoch} ${resume_arg} \
#  --seed 0 ${sync} ${stick_to_shard}

#python -m torch.distributed.launch \
#  --nproc_per_node=8 --nnodes=${size} --node_rank=${rank} --master_addr=${master_ip} --master_port=8888 \
#  /home/acd13314za/CAC_Lib/example/image_classification/example_imagenet.py \
#  -a resnet50 --workers 5 --batch-size 256 \
#  /raid/aibench/forPyTorch/imagenet

  #--data-backend pytorch --raport-file raport.json -j5 -p ${print_freq} \
#  --lr ${lr} --optimizer-batch-size ${total_bs} --warmup 8 --arch resnet50 -c fanin --label-smoothing 0.1 \

#  --fp16 --static-loss-scale 128 --epochs 90
#python ${log_dir}/multiproc.py \

#python -m torch.distributed.launch \
#  --nproc_per_node=2 \
#  --nnodes=$size \
#  --node_rank=$rank \
#  --master_addr=$master_ip \
#  --master_port=8888 \
#  git/apex/examples/imagenet/main_amp.py \
#  -a resnet50 \
#  --workers 4 \
#  --batch-size 52 \
#  --lr 38.0 \
#  --opt-level O3 \
#  --keep-batchnorm-fp32 False \
#  --loss-scale 1.0 \
#  /mnt/share/Images/forPyTorch/


# run imagenet
#python git/examples/imagenet/example_imagenet.py \
#  --dist-url "env://" \
#  --rank $rank \
#  --world-size $size \
#  --arch resnet50 \
#  /mnt/share/Images/forPyTorch/

#  --gpu $SGE_HGR_gpu \
#  --multiprocessing-distributed \
#  --gpu $SGE_HGR_gpu \
#  --batch-size 52 \
#  --learning-rate 0.1 \
# print date and time
date
date +%s

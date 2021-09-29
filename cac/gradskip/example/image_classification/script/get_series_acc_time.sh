#!/bin/bash

if [ $# -eq 0 ]; then
  echo "$0 [job_id] ([job_id_2] ...)"
  exit 1
fi
  
epoch_ary=()
elapsed_ary=()
skip_ary=()
top1_target=75.90
#top1_target=99.90

for i in `seq ${#}`; do
  job_id=$1
  path=~/work/logs/${job_id}/raport.json
  if [ ! -e ${path} ]; then
    echo "${path} does not exist. exit."
    exit 1
  fi

  epoch_list=($(grep val.top1 ${path} | grep train.loss | sed s/,//g | awk '{print $12}' | sed -e 's/[^0-9]//g'))
  #elapsed_list=`grep val.top1 ${path} | grep train.loss | sed s/,//g | awk '{print $8}' | sed -e 's/[^0-9\.]//g' | awk '{printf("%d\n",$1)}'`
  elapsed_list=($(grep val.top1 ${path} | grep train.loss | sed s/,//g | awk '{print $8}' | sed -e 's/[^0-9\.]//g' | sed '$d'))
  acc_list=($(grep val.top1 ${path} | grep train.loss | sed s/,//g | awk '{print $19}' | sed -e 's/[^0-9\.]//g' | sed '$d'))

  param=`grep PARAM ${path}`
  size=`echo $param | awk '{print $NF}' | sed -e 's/[^0-9]//g'`
  skip_interval=`echo $param | awk '{print $87}' | sed -s 's/[^-0-9]//g'`
  skip=`if [ ${skip_interval} -eq -1 ]; then echo 0; else echo $(( ${size} / ${skip_interval} )); fi`

  echo "size: $size"
  echo "skip: $skip"
  echo ""

  epochs=`grep val.top1 ${path} | grep train.loss | sed '$d' | wc -l`
  for i in `seq 0 $((epochs - 1))`; do
    echo "${elapsed_list[$i]} ${epoch_list[$i]} ${acc_list[$i]}"
  done

#  shift
done

#!/bin/bash

if [ $# -eq 0 ]; then
  echo "$0 [job_id] ([job_id_2] ...)"
  exit 1
fi
  
epoch_ary=()
top1_ary=()
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

  finish=`grep val.top1 ${path} | grep train.loss | sed s/,//g | awk -v top1_target=${top1_target} '$19 >= top1_target' | head -n1`
  if [ -z "${finish}" ]; then
    finish=`grep val.top1 ${path} | grep train.loss | sed s/,//g | tail -n2 | head -n1`
  fi
  top1=`echo ${finish} | awk '{print $19}' | awk '{printf("%.2f\n",$1)}'`
  epoch=`echo ${finish} | awk '{print $12}' | sed -e 's/[^0-9]//g'`
  #elapsed=`echo ${finish} | awk '{print $8}' | sed -e 's/[^0-9\.]//g' | awk '{printf("%d\n",$1 + 0.5)}'`
  elapsed=`echo ${finish} | awk '{print $8}' | sed -e 's/[^0-9\.]//g' | awk '{printf("%d\n",$1)}'`

  param=`grep PARAM ${path}`
  size=`echo $param | awk '{print $NF}' | sed -e 's/[^0-9]//g'`
  skip_interval=`echo $param | awk '{print $87}' | sed -s 's/[^-0-9]//g'`
  skip=`if [ ${skip_interval} -eq -1 ]; then echo 0; else echo $(( ${size} / ${skip_interval} )); fi`

  top1_ary=("${top1_ary[@]}"$'\t'${top1})
  epoch_ary=("${epoch_ary[@]}"$'\t'${epoch})
  elapsed_ary=("${elapsed_ary[@]}"$'\t'${elapsed})
  skip_ary=("${skip_ary[@]}"$'\t'${skip})

  shift
done

echo -e "skip${skip_ary[@]}"
echo -e "top1${top1_ary[@]}"
echo -e "epoch${epoch_ary[@]}"
echo -e "elapsed${elapsed_ary[@]}"

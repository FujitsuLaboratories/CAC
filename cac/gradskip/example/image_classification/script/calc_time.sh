#!/bin/bash

if [ $# -lt 1 ]; then
  echo "$0 [path]"
  exit 1
fi

fwd_ary=()
bwd_ary=()
opt_ary=()
comm_ary=()
other_ary=()
all_min_ary=()
fwd_min_ary=()
bwd_min_ary=()
opt_min_ary=()
comm_min_ary=()
other_min_ary=()
all_max_ary=()
fwd_max_ary=()
bwd_max_ary=()
opt_max_ary=()
comm_max_ary=()
other_max_ary=()

size=`ls -l $1 | grep txt | wc -l`

echo -e "rank"$'\t'"fwd"$'\t'"bwd"$'\t'"opt"$'\t'"comm"$'\t'"other"$'\t'"all"
for i in `seq 0 $(($size-1))`; do
#  echo $i
  file_name=$1/${i}.txt
  #time_list=`grep all ${file_name} | awk '{print $3}'`
  #time_list=`cat ${file_name} | awk '{print $3}'`
  time_list=`cat ${file_name} | sed '1d' | awk '{print $3}'`
  count=0
  total=0
  #max=$(awk '{print $2}' <<<${time_list})
  max=$(awk '{print $1}' <<<${time_list})
  min=$max
#  for i in ${time_list}; do
#    total=$(echo ${total}+${i} | bc );
#    if [ `echo "${i} > ${max}" | bc` == 1 ]; then
#      max=${i}
#    fi
#    if [ `echo "${i} < ${min}" | bc` == 1 ]; then
#      min=${i}
#    fi
#    ((count++))
#  done
  total=$(awk '{for(i=1; i<NF+1; ++i){s += $i} print s}' <<<${time_list})
  count=$(awk '{print NF}' <<<${time_list})
  all_time=`echo "scale=10; ${total} / ${count}" | bc`
  all_ary=("${all_ary[@]}"$'\t'${all_time})
  all_min_ary=("${all_min_ary[@]}"$'\t'${min})
  all_max_ary=("${all_max_ary[@]}"$'\t'${max})

  #time_list=`grep fwd ${file_name} | awk '{print $3}'`
  time_list=`cat ${file_name} | sed '1d' | awk '{print $6}'`
  count=0
  total=0
#  max=$(awk '{print $1}' <<<${time_list})
#  min=$max
#  for i in ${time_list}; do
#    total=$(echo ${total}+${i} | bc );
#    if [ `echo "${i} > ${max}" | bc` == 1 ]; then
#      max=${i}
#    fi
#    if [ `echo "${i} < ${min}" | bc` == 1 ]; then
#      min=${i}
#    fi
#    ((count++))
#  done
  total=$(awk '{for(i=1; i<NF+1; ++i){s += $i} print s}' <<<${time_list})
  count=$(awk '{print NF}' <<<${time_list})
  fwd_time=`echo "scale=10; ${total} / ${count}" | bc`
  fwd_ary=("${fwd_ary[@]}"$'\t'${fwd_time})
  fwd_min_ary=("${fwd_min_ary[@]}"$'\t'${min})
  fwd_max_ary=("${fwd_max_ary[@]}"$'\t'${max})

  #time_list=`grep fwd ${file_name} | awk '{print $6}'`
  time_list=`cat ${file_name} | sed '1d' | awk '{print $9}'`
  count=0
  total=0
#  max=$(awk '{print $2}' <<<${time_list})
#  min=$max
#  for i in ${time_list}; do
#    total=$(echo ${total}+${i} | bc );
#    if [ `echo "${i} > ${max}" | bc` == 1 ]; then
#      max=${i}
#    fi
#    if [ `echo "${i} < ${min}" | bc` == 1 ]; then
#      min=${i}
#    fi
#    ((count++))
#  done
  total=$(awk '{for(i=1; i<NF+1; ++i){s += $i} print s}' <<<${time_list})
  count=$(awk '{print NF}' <<<${time_list})
  bwd_time=`echo "scale=10; ${total} / ${count}" | bc`
  bwd_ary=("${bwd_ary[@]}"$'\t'${bwd_time})
  bwd_min_ary=("${bwd_min_ary[@]}"$'\t'${min})
  bwd_max_ary=("${bwd_max_ary[@]}"$'\t'${max})

  #time_list=`grep fwd ${file_name} | awk '{print $9}'`
  time_list=`cat ${file_name} | sed '1d' | awk '{print $12}'`
  count=0
  total=0
#  max=$(awk '{print $2}' <<<${time_list})
#  min=$max
#  for i in ${time_list}; do
#    total=$(echo ${total}+${i} | bc );
#    if [ `echo "${i} > ${max}" | bc` == 1 ]; then
#      max=${i}
#    fi
#    if [ `echo "${i} < ${min}" | bc` == 1 ]; then
#      min=${i}
#    fi
#    ((count++))
#  done
  total=$(awk '{for(i=1; i<NF+1; ++i){s += $i} print s}' <<<${time_list})
  count=$(awk '{print NF}' <<<${time_list})
  opt_time=`echo "scale=10; ${total} / ${count}" | bc`
  opt_ary=("${opt_ary[@]}"$'\t'${opt_time})
  opt_min_ary=("${opt_min_ary[@]}"$'\t'${min})
  opt_max_ary=("${opt_max_ary[@]}"$'\t'${max})

  #time_list=`grep fwd ${file_name} | awk '{print $9}'`
  time_list=`cat ${file_name} | sed '1d' | awk '{print $15}'`
  count=0
  total=0
#  max=$(awk '{print $2}' <<<${time_list})
#  min=$max
#  for i in ${time_list}; do
#    total=$(echo ${total}+${i} | bc );
#    if [ `echo "${i} > ${max}" | bc` == 1 ]; then
#      max=${i}
#    fi
#    if [ `echo "${i} < ${min}" | bc` == 1 ]; then
#      min=${i}
#    fi
#    ((count++))
#  done
  total=$(awk '{for(i=1; i<NF+1; ++i){s += $i} print s}' <<<${time_list})
  count=$(awk '{print NF}' <<<${time_list})
  comm_time=`echo "scale=10; ${total} / ${count}" | bc`
  comm_ary=("${comm_ary[@]}"$'\t'${comm_time})
  comm_min_ary=("${comm_min_ary[@]}"$'\t'${min})
  comm_max_ary=("${comm_max_ary[@]}"$'\t'${max})

  #time_list=`grep fwd ${file_name} | awk '{print $12}'`
  time_list=`cat ${file_name} | sed '1d' | awk '{print $18}'`
  count=0
  total=0
#  max=$(awk '{print $2}' <<<${time_list})
#  min=$max
#  for i in ${time_list}; do
#    total=$(echo ${total}+${i} | bc );
#    if [ `echo "${i} > ${max}" | bc` == 1 ]; then
#      max=${i}
#    fi
#    if [ `echo "${i} < ${min}" | bc` == 1 ]; then
#      min=${i}
#    fi
#    ((count++))
#  done
  total=$(awk '{for(i=1; i<NF+1; ++i){s += $i} print s}' <<<${time_list})
  count=$(awk '{print NF}' <<<${time_list})
  other_time=`echo "scale=10; ${total} / ${count}" | bc`
  other_ary=("${other_ary[@]}"$'\t'${other_time})
  other_min_ary=("${other_min_ary[@]}"$'\t'${min})
  other_max_ary=("${other_max_ary[@]}"$'\t'${max})

  echo -e "$i"$'\t'"${fwd_time}"$'\t'"${bwd_time}"$'\t'"${opt_time}"$'\t'"${comm_time}"$'\t'"${other_time}"$'\t'"${all_time}"
done

#echo -e "rank$'\t'fwd$'\t'bwd$'\t'opt$'\t'other"
#echo -e "$1$'\t'${fwd_time}$'\t'${bwd_time}$'\t'${opt_time}$'\t'${other_time}"

#echo -e "average"
#echo -e "${size[@]}"
#echo -e "fwd${fwd_ary[@]}"
#echo -e "bwd${bwd_ary[@]}"
#echo -e "opt${opt_ary[@]}"
#echo -e "other${other_ary[@]}"
#echo -e "all${all_ary[@]}"

#echo -e ""
#echo -e "min"
#echo -e "${size[@]}"
#echo -e "fwd${fwd_min_ary[@]}"
#echo -e "bwd${bwd_min_ary[@]}"
#echo -e "opt${opt_min_ary[@]}"
#echo -e "other${other_min_ary[@]}"
#echo -e "all${all_min_ary[@]}"
#
#echo -e ""
#echo -e "max"
#echo -e "${size[@]}"
#echo -e "fwd${fwd_max_ary[@]}"
#echo -e "bwd${bwd_max_ary[@]}"
#echo -e "opt${opt_max_ary[@]}"
#echo -e "other${other_max_ary[@]}"
#echo -e "all${all_max_ary[@]}"

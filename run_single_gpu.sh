#!/bin/bash

WORKERS=$1
GPU_IDX=$2
shift; shift;

PYTHON_ARGS=""
while ! [[ -z "$1" ]]; do
  PYTHON_ARGS+="$1 "
  shift
done

TASK_IDX_START=$((WORKERS*GPU_IDX))
TASK_IDX_END=$((WORKERS*(GPU_IDX+1)-2))
for i in $(seq $TASK_IDX_START $TASK_IDX_END ); do
  python Hogwild.py --job_name worker --task_index $i $PYTHON_ARGS &
done
python Hogwild.py --job_name worker --task_index $(($TASK_IDX_END+1)) $PYTHON_ARGS

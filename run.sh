#!/bin/bash

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

HELP=false
WORKERS=1
NUM_GPUS=1
MODEL_DIR=""
PROFILE_DIR=""
LOG_DIR=""
INVERT_PS_DEVICE=false
DATASET_PATH="data/train.tfrecords"
TAG=latest-gpu
KEEP=false

SHORT_OPTS="b:w:e:d:i:s:f:o:p:l:g:cxkh"
LONG_OPTS="batch_size:,workers:,steps:,hidden_sizes:,log_frequency:,dataset_path:,dense_size:,model_dir:,profile_dir:,log_dir:,num_gpus:,cpu,binary_inputs,keep,help"
OPTS=`getopt -o $SHORT_OPTS --long $LONG_OPTS -n 'parse-options' -- "$@"`
if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; exit 1 ; fi
eval set -- "$OPTS"

PYTHON_ARGS=""
while true; do
  case "$1" in
    -b | --batch_size       ) PYTHON_ARGS+="--batch_size $2 "; shift; shift ;;
    -w | --workers          ) WORKERS=$2; shift; shift ;;
    -e | --steps            ) PYTHON_ARGS+="--steps $2 "; shift; shift ;;
    -d | --dataset_path     ) DATASET_PATH=$2; shift; shift ;;
    -i | --dense_size       ) PYTHON_ARGS+="--dense_size $2 "; shift; shift ;;
    -s | --hidden_sizes     ) PYTHON_ARGS+="--hidden_sizes ${2//,/ } "; shift; shift ;;
    -f | --log_frequency    ) PYTHON_ARGS+="--log_frequency $2 "; shift; shift ;;
    -o | --model_dir        ) MODEL_DIR=$2; PYTHON_ARGS+="--model_dir /tmp/model/ "; shift; shift ;;
    -p | --profile_dir      ) PROFILE_DIR=$2; PYTHON_ARGS+="--profile_dir /tmp/profile/ "; shift; shift ;;
    -l | --log_dir          ) LOG_DIR=$2; PYTHON_ARGS+="--log_dir /tmp/log/ "; shift; shift ;;
    -g | --num_gpus         ) NUM_GPUS=$2; shift; shift ;;
    -c | --cpu              ) TAG=latest; shift;;
    -x | --binary_inputs    ) PYTHON_ARGS+="--binary_inputs "; shift ;;
    -k | --keep             ) KEEP=true; shift ;;
    -h | --help             ) HELP=true; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if [ "$HELP" = true ]; then
  echo "Asychronously train multiple copies of a mulitlayer percepton on random data. Distributes over multiple copies on multiple GPUs"
  echo "Parameters"
  echo "----------"
  echo "    --batch_size, -b       : batch size of a single gradient step"
  echo "    --workers, -w          : number of copies of the model to train concurrently per GPU"
  echo "    --steps, -e            : total number of gradient steps. Since each worker takes a few seconds to spin up, running for more epochs can make speedups more pronounced"
  echo "    --dataset_path, -d     : path to tfrecord dataset"
  echo "    --dense_size, -i       : dimensionality of input space"
  echo "    --hidden_sizes, -s     : size of hidden layers of MLP seperated by commas (e.g. 512,512,256)"
  echo "    --log_frequency, -f    : number of gradient updates between stdout logging"
  echo "    --min_nnz, -m          : minimum number of nonzero elements in a sample of random data"
  echo "    --max_nnz, -n          : maximum number of nonzero elements in a sample of random data"
  echo "    --dense_size, -s       : full dimensionality of sparse input space. If set to less than 30, size will be 1 << dense_size"
  echo "    --model_dir, -o        : where to log model checkpoints. Must be specified so containers can refer to same checkpoints. Note that it should be empty or TF will complain"
  echo "    --profile_dir, -p      : where to store timeline profiles. Saved with the same frequency as log frequency"
  echo "    --log_dir, -l          : where to save print logging"
  echo "    --num_gpus, -g         : number of gpus to distribute over"
  echo "    --cpu, -c              : run workers on cpu"
  echo "    --binary_inputs, -x    : whether inputs only take on values 0 or 1. If true, can help keep gradients sparse"
  echo "    --keep, -k             : if flag is on, keep the model data stored in model_dir. Otherwise delete it"
  echo "    --help, -h             : show this help"
  exit 0
fi

PYTHON_ARGS+="--num_workers $((WORKERS*NUM_GPUS))"

# avoid ulimit issues (tested up to at least 16 processes)
ulimit -u 8192

# utility function for creating the necessary volume mounts for model checkpointing and profiling
check_dir (){
  HOST_DIR="$1"
  CONTAINER_DIR="$2"

  # check if host directory is empty. Note that this would make "$CONTAINER_DIR" $1
  if ! [[ -z "$CONTAINER_DIR" ]]; then
    # create it if it doesn't exitst.
    if ! [[ -d "$HOST_DIR" ]]; then mkdir -p $HOST_DIR; fi

    # if it's a relative path, append it to the pwd
    if [[ ! "$HOST_DIR" = /* ]]; then HOST_DIR=$(pwd)/$HOST_DIR; fi

    echo " -v $HOST_DIR:$CONTAINER_DIR"
  else
    echo ""
  fi
}

# build the base command
DOCKER_CMD="docker run --rm -d -u $(id -u):$(id -g) -v $PWD:/workspace/ --workdir /workspace --net=host"

# add volume mounts for saving data out from container
DOCKER_CMD+=$(check_dir $MODEL_DIR /tmp/model)
DOCKER_CMD+=$(check_dir $PROFILE_DIR /tmp/profile)
DOCKER_CMD+=$(check_dir $LOG_DIR /tmp/log)


if [[ ! "$DATASET_PATH" = /* ]]; then DATASET_PATH=$PWD/$DATASET_PATH; fi
DATASET_DIR=$(dirname "$DATASET_PATH")
DATASET_FILE=$(basename "$DATASET_PATH")
DOCKER_CMD+=" -v $DATASET_DIR:/data/"
PYTHON_ARGS+=" --dataset_path /data/$DATASET_FILE"

# initialize chief node and parameter server
HASHES=""

CHIEF+="$DOCKER_CMD --name=chief tensorflow/tensorflow:latest-py3 python Hogwild.py --job_name chief --task_index 0 $PYTHON_ARGS"
echo $CHIEF
HASHES+=$($CHIEF)" "

PS="$DOCKER_CMD --name=ps tensorflow/tensorflow:latest-py3 python Hogwild.py --job_name ps --task_index 0 $PYTHON_ARGS"
echo $PS
HASHES+=$($PS)" "

# execute worker nodes on each GPU
for g in $( seq 0 $((NUM_GPUS-1)) ); do
  WORKER_CMD=$DOCKER_CMD
  if [[ "$TAG" == "latest-gpu" ]]; then
    WORKER_CMD+=" --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$g"
  fi
  WORKER_CMD+=" --name=worker$g tensorflow/tensorflow:$TAG-py3 ./run_single_gpu.sh $WORKERS $g $PYTHON_ARGS"
  echo $WORKER_CMD
  HASHES+=$($WORKER_CMD)" "
done
echo $HASHES

while ! [[ -z "$(docker ps -q -f name=worker*)" ]]; do
  sleep 1
done

docker kill ps chief
if [[ "$KEEP" == "false" ]]; then
  rm -r $MODEL_DIR/*
fi

if ! [[ -z "$LOG_DIR" ]]; then
  cat $LOG_DIR/log
fi

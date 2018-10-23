#!/bin/bash

# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
TAG=latest-gpu

SHORT_OPTS="b:w:e:d:f:m:n:s:o:p:l:g:ch"
LONG_OPTS="batch_size:,workers:,steps:,hidden_sizes:,log_frequency:,min_nnz:,max_nnz:,dense_size:,model_dir:,profile_dir:,log_dir:,num_gpus:,cpu,help"
OPTS=`getopt -o $SHORT_OPTS --long $LONG_OPTS -n 'parse-options' -- "$@"`
if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; exit 1 ; fi
eval set -- "$OPTS"

PYTHON_ARGS=""
while true; do
  case "$1" in
    -b | --batch_size       ) PYTHON_ARGS+="--batch_size $2 "; shift; shift ;;
    -w | --workers          ) WORKERS=$2; shift; shift ;;
    -e | --steps            ) PYTHON_ARGS+="--steps $2 "; shift; shift ;;
    -d | --hidden_sizes     ) PYTHON_ARGS+="--hidden_sizes ${2//,/ } "; shift; shift ;;
    -f | --log_frequency    ) PYTHON_ARGS+="--log_frequency $2 "; shift; shift ;;
    -m | --min_nnz          ) PYTHON_ARGS+="--min_nnz $2 "; shift; shift ;;
    -n | --max_nnz          ) PYTHON_ARGS+="--max_nnz $2 "; shift; shift ;;
    -s | --dense_size       ) PYTHON_ARGS+="--dense_size $2 "; shift; shift ;;
    -o | --model_dir        ) MODEL_DIR=$2; PYTHON_ARGS+="--model_dir /tmp/model/ "; shift; shift ;;
    -p | --profile_dir      ) PROFILE_DIR=$2; PYTHON_ARGS+="--profile_dir /tmp/profile/ "; shift; shift ;;
    -l | --log_dir          ) LOG_DIR=$2; PYTHON_ARGS+="--log_dir /tmp/log/ "; shift; shift ;;
    -g | --num_gpus         ) NUM_GPUS=$2; shift; shift ;;
    -c | --cpu              ) TAG=latest; shift;;
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
  echo "    --hidden_sizes, -d     : size of hidden layers of MLP seperated by commas (e.g. 512,512,256)"
  echo "    --log_frequency, -f    : number of gradient updates between stdout logging"
  echo "    --min_nnz, -m          : minimum number of nonzero elements in a sample of random data"
  echo "    --max_nnz, -n          : maximum number of nonzero elements in a sample of random data"
  echo "    --dense_size, -s       : full dimensionality of sparse input space. If set to less than 30, size will be 1 << dense_size"
  echo "    --model_dir, -o        : where to log model checkpoints. Must be specified so containers can refer to same checkpoints. Note that it should be empty or TF will complain"
  echo "    --profile_dir, -p      : where to store timeline profiles. Saved with the same frequency as log frequency"
  echo "    --log_dir, -l          : where to save print logging"
  echo "    --num_gpus, -g         : number of gpus to distribute over"
  echo "    --cpu, -c              : run workers on cpu"
  echo "    --help, -h             : show this help"
  exit 0
fi

PYTHON_ARGS+="--num_tasks $((WORKERS*NUM_GPUS))"

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

# initialize chief node and parameter server
HASHES=""
CHIEF+="$DOCKER_CMD --name=chief tensorflow/tensorflow python Hogwild.py --job_name chief --task_index 0 $PYTHON_ARGS"
echo $CHIEF
HASHES+=$($CHIEF)" "

PS="$DOCKER_CMD --name=ps tensorflow/tensorflow  python Hogwild.py --job_name ps --task_index 0 $PYTHON_ARGS"
echo $PS
HASHES+=$($PS)" "

# execute worker nodes on each GPU
for g in $(seq 1 $NUM_GPUS); do
  for i in $(seq 1 $WORKERS ); do
    TASK_IDX=$(((g-1)*WORKERS+i-1))
    WORKER_CMD=$DOCKER_CMD
    if [[ "$TAG" == "latest-gpu" ]]; then
      WORKER_CMD+=" --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$((g-1))"
    fi
    WORKER_CMD+=" --name=worker$TASK_IDX tensorflow/tensorflow:$TAG python Hogwild.py --job_name worker --task_index $TASK_IDX $PYTHON_ARGS"
    echo $WORKER_CMD
    HASHES+=$($WORKER_CMD)" "
  done
done
echo $HASHES

while ! [[ -z "$(docker ps -q -f name=chief)" ]]; do
  sleep 1
done

docker kill ps $(docker ps -q -f name=worker*)
if ! [[ -z "$LOG_DIR" ]]; then
  cat $LOG_DIR/log
fi

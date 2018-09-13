#!/bin/bash

HELP=false
WORKERS=1
NUM_GPUS=1
MODEL_DIR=""
PROFILE_DIR=""
LOG_DIR=""
INVERT_PS_DEVICE=false

SHORT_OPTS="b:w:e:d:f:m:n:s:o:p:l:g:ih"
LONG_OPTS="batch_size:,workers:,steps:,hidden_sizes:,log_frequency:,min_nnz:,max_nnz:,dense_size:,model_dir:,profile_dir:,log_dir:,num_gpus:,invert_ps_device,help"
OPTS=`getopt -o $SHORT_OPTS --long $LONG_OPTS -n 'parse-options' -- "$@"`
if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; exit 1 ; fi
eval set -- "$OPTS"

STATIC_ARGS="--model_dir /tmp/models --log_dir /tmp/logs/"
while true; do
  case "$1" in
    -b | --batch_size       ) STATIC_ARGS+="--batch_size $2 "; shift; shift ;;
    -w | --workers          ) WORKERS=$2; shift; shift ;;
    -e | --steps            ) STATIC_ARGS+="--steps $2 "; shift; shift ;;
    -d | --hidden_sizes     ) STATIC_ARGS+="--hidden_sizes ${2//,/ } "; shift; shift ;;
    -f | --log_frequency    ) STATIC_ARGS+="--log_frequency $2 "; shift; shift ;;
    -m | --min_nnz          ) STATIC_ARGS+="--min_nnz $2 "; shift; shift ;;
    -n | --max_nnz          ) STATIC_ARGS+="--max_nnz $2 "; shift; shift ;;
    -s | --dense_size       ) STATIC_ARGS+="--dense_size $2 "; shift; shift ;;
    -o | --model_dir        ) MODEL_DIR=$2; shift; shift ;;
    -p | --profile_dir      ) PROFILE_DIR=$2; STATIC_ARGS+="--profile_dir /tmp/profile/ "; shift; shift ;;
    -l | --log_dir          ) LOG_DIR=$2; shift; shift ;;
    -g | --num_gpus         ) NUM_GPUS=$2; shift; shift ;;
    -i | --invert_ps_device ) INVERT_PS_DEVICE=true; shift ;;
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
  echo "    --invert_ps_device, -i : if using one GPU, host parameter server on CPU. If using multiple GPUs, hosts parameter server on GPU 0"
  echo "    --help, -h             : show this help"
  exit 0
fi

STATIC_ARGS+="--num_tasks $((WORKERS*NUM_GPUS))"

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
# TODO: add log_dir to cla
CMD="docker run --rm -d -u $(id -u):$(id -g) -v $(pwd):/workspace/"
CMD+=$(check_dir $MODEL_DIR /tmp/models)
CMD+=$(check_dir $PROFILE_DIR /tmp/profile)
CMD+=$(check_dir $LOG_DIR /tmp/logs)

# decide whether to run ps & chief on CPU or GPU
HEAD_ARGS="$USER/hogwild:cpu"
if [[ "$NUM_GPUS" -gt 1 && "$INVERT_PS_DEVICE" = "true" ]] || [[ "$NUM_GPUS" -eq 1 && ! "$INVERT_PS_DEVICE" = "true"  ]]; then
  HEAD_ARGS="--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 $USER/hogwild:gpu"
fi  

# initialize chief node and parameter server
docker kill ps chief $(docker ps -q -f name=worker*)
HASHES=""
CHIEF="$CMD --net="host" --name=chief $HEAD_ARGS --job_name chief --task_index 0 $STATIC_ARGS"
echo $CHIEF
HASHES+=$($CHIEF)" "

PS="$CMD --net="host" --name=ps $HEAD_ARGS --job_name ps --task_index 0 $STATIC_ARGS"
echo $PS
HASHES+=$($PS)" "

# execute worker nodes on each GPU
for g in $(seq 1 $NUM_GPUS); do
  for i in $(seq 1 $WORKERS ); do
    TASK_IDX=$(((g-1)*WORKERS+i-1))
    WORKER="$CMD --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$((g-1)) --net="host" --name=worker$TASK_IDX $USER/hogwild:gpu --job_name worker --task_index $TASK_IDX $STATIC_ARGS"
    echo $WORKER
    HASHES+=$($WORKER)" "
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

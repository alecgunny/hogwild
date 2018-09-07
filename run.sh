#!/bin/bash
HELP=false
WORKERS=1
NUM_GPUS=1
MODEL_DIR=""

OPTS=`getopt -o b:w:e:d:f:m:n:s:o:g:h --long batch_size:,workers:,steps:,hidden_sizes:,log_frequency:,min_nnz:,max_nnz:,dense_size:,model_dir:,num_gpus:,help -n 'parse-options' -- "$@"`
if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; exit 1 ; fi
eval set -- "$OPTS"

STATIC_ARGS=""
while true; do
  case "$1" in
    -b | --batch_size )    STATIC_ARGS+="--batch_size $2 "; shift; shift ;;
    -w | --workers )       WORKERS=$2; shift; shift ;;
    -e | --steps )         STATIC_ARGS+="--steps $2 "; shift; shift ;;
    -d | --hidden_sizes )  STATIC_ARGS+="--hidden_sizes ${2//,/ } "; shift; shift ;;
    -f | --log_frequency ) STATIC_ARGS+="--log_frequency $2 "; shift; shift ;;
    -m | --min_nnz )       STATIC_ARGS+="--min_nnz $2 "; shift; shift ;;
    -n | --max_nnz )       STATIC_ARGS+="--max_nnz $2 "; shift; shift ;;
    -s | --dense_size )    STATIC_ARGS+="--dense_size $2 "; shift; shift ;;
    -o | --model_dir )     MODEL_DIR=$2; shift; shift ;;
    -g | --num_gpus )      NUM_GPUS=$2; shift; shift;;
    -h | --help )          HELP=true; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if [ "$HELP" = true ]; then
  echo "Asychronously train multiple copies of a mulitlayer percepton on random data on a single gpu"
  echo "Parameters"
  echo "----------"
  echo "    --batch_size, -b    : batch size of a single gradient step"
  echo "    --workers, -w       : number of copies of the model to train concurrently"
  echo "    --steps, -e         : total number of gradient steps. Since each worker takes a few seconds to spin up, running for more epochs can make speedups more pronounced"
  echo "    --hidden_sizes, -d  : size of hidden layers of MLP seperated by commas (e.g. 512,512,256)"
  echo "    --log_frequency, -f : epochs between stdout logging"
  echo "    --min_nnz, -m       : minimum number of nonzero elements in a sample of random data"
  echo "    --max_nnz, -n       : maximum number of nonzero elements in a sample of random data"
  echo "    --dense_size, -s    : full dimensionality of sparse input space. If set to less than 30, size will be 1 << dense_size"
  echo "    --model_dir, -o     : where to log model checkpoints. To save after run, volume map this directory into the container"
  echo "    --num_gpus, -g      : number of gpus to distribute over"
  echo "    --help, -h          : show this help"
  exit 0
fi

STATIC_ARGS+="--num_tasks $((WORKERS*NUM_GPUS)) --model_dir /tmp/models  --num_gpus $NUM_GPUS"

CMD="docker run --rm -d"
if ! [[ -z "$MODEL_DIR" ]]; then
  if [[ -d "$MODEL_DIR" ]]; then
    rm -r $MODEL_DIR/*
  else
    mkdir -p $MODEL_DIR
  fi

  if [[ ! "$MODEL_DIR" = /* ]]; then
    MODEL_DIR=$(pwd)/$MODEL_DIR
  fi

  CMD+=" -v $MODEL_DIR:/tmp/models"
fi
CMD+=" -v $(pwd):/workspace/"

rm out.txt

# initialize chief node and parameter serving node
HASHES=""
CHIEF="$CMD --net="host" --name=chief$g $USER/hogwild:cpu --job_name chief --task_index 0 $STATIC_ARGS"
echo $CHIEF
HASHES+=$($CHIEF)" "

PS="$CMD --net="host" --name=ps $USER/hogwild:cpu --job_name ps --task_index 0 $STATIC_ARGS"
echo $PS
HASHES+=$($PS)" "

# execute last job foreground to keep container from exiting
for g in $(seq 1 $NUM_GPUS); do
  for i in $(seq 1 $WORKERS ); do
    TASK_IDX=$(((g-1)*WORKERS+i-1))
    WORKER="$CMD --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$((g-1)) --net="host" --name=worker$TASK_IDX $USER/hogwild:gpu --job_name worker --task_index $TASK_IDX $STATIC_ARGS"
    echo $WORKER
    HASHES+=$($WORKER)" "
  done
done
echo $HASHES

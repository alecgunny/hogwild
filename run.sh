#!/bin/bash
HELP=false
OPTS=`getopt -o b:w:e:d:f:m:n:s:h --long batch_size:,workers:,epochs:,hidden_sizes:,log_frequency:,min_nnz:,max_nnz:,dense_size:,help -n 'parse-options' -- "$@"`
if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; exit 1 ; fi
eval set -- "$OPTS"

STATIC_ARGS=""
eval set -- "$OPTS"
while true; do
  case "$1" in
    -b | --batch_size )    STATIC_ARGS+="--batch_size $2 "; shift; shift ;;
    -w | --workers )       WORKERS=$2; shift; shift ;;
    -e | --epochs )        STATIC_ARGS+="--epochs $2 "; shift; shift ;;
    -d | --hidden_sizes )   STATIC_ARGS+="--hidden_sizes ${2//,/ } "; shift; shift ;;
    -f | --log_frequency ) STATIC_ARGS+="--log_frequency $2 "; shift; shift ;;
    -m | --min_nnz )       STATIC_ARGS+="--min_nnz $2 "; shift; shift ;;
    -n | --max_nnz )       STATIC_ARGS+="--max_nnz $2 "; shift; shift ;;
    -s | --dense_size )    STATIC_ARGS+="--dense_size $2 "; shift; shift ;;
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
  echo "    --epochs, -e        : number of training epochs. Important to allow all workers to spin up"
  echo "    --hidden_sizes, -d  : size of hidden layers of MLP seperated by commas (e.g. 512,512,256)"
  echo "    --log_frequency, -f : epochs between stdout logging"
  echo "    --min_nnz, -m       : minimum number of nonzero elements in a sample of random data"
  echo "    --max_nnz, -n       : maximum number of nonzero elements in a sample of random data"
  echo "    --dense_size, -s    : full dimensionality of input space"
  echo "    --help, -h          : show this help"
  exit 0
fi

STATIC_ARGS+="--num_tasks $WORKERS"
echo $STATIC_ARGS

# initialize chief node and parameter serving node
python Hogwild.py --job_name chief --task_index 0 $STATIC_ARGS &
python Hogwild.py --job_name ps --task_index 0 $STATIC_ARGS &

# execute last job foreground to keep container from exiting
for i in $(seq 2 $WORKERS ); do
  python Hogwild.py --job_name worker --task_index $((i-2)) $STATIC_ARGS &
done
python Hogwild.py --job_name worker --task_index $(($WORKERS-1)) $STATIC_ARGS

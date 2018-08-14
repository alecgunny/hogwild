#!/bin/bash
HELP=false
OPTS=`getopt -o b:w:e:d:f:h --long batch_size:,workers:,epochs:,hidden_dims:,log_frequency:,help -n 'parse-options' -- "$@"`
if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; exit 1 ; fi
eval set -- "$OPTS"

STATIC_ARGS=""
eval set -- "$OPTS"
while true; do
  case "$1" in
    -b | --batch_size )    STATIC_ARGS+="--batch_size $2 "; shift; shift ;;
    -w | --workers )       WORKERS=$2; shift; shift ;;
    -e | --epochs )        STATIC_ARGS+="--epochs $2 "; shift; shift ;;
    -d | --hidden_dims )   STATIC_ARGS+="--hidden_dims ${2//,/ } "; shift; shift ;;
    -f | --log_frequency ) STATIC_ARGS+="--log_frequency $2 "; shift; shift ;;
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
  echo "    --hidden_dims, -d   : size of hidden layers of MLP seperated by commas (e.g. 512,512,256)"
  echo "    --log_frequency, -f : epochs between stdout logging"
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

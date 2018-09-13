#!/bin/bash

DENSE_SIZES=( 1000000 4000000 16000000 )
BATCH_SIZES=( 8 32 64 )
WORKERS=( 1 2 4 8 )
HIDDEN_SIZES=( 128,1024 128,1024,1024,1024 )
GPUS=( 1 2 4 )

for d in "${DENSE_SIZES[@]}"; do
  for b in "${BATCH_SIZES[@]}"; do
    for w in "${WORKERS[@]}"; do
      for h in "${HIDDEN_SIZES[@]}"; do
        for g in "${GPUS[@]}"; do
          RUN_DIR="outputs/dense-${d}_batch-${b}_workers-${w}_hidden-${h}_gpus-${g}"
          if ! [[ -d "$RUN_DIR" ]]; then 
            CMD="./run.sh --profile_dir $RUN_DIR/profile --log_dir $RUN_DIR/logs --dense_size ${d} --batch_size ${b} --workers ${w} --hidden_sizes ${h} --num_gpus ${g} --steps 100000 --log_frequency 10000"
            if [[ "$g" = 1 ]]; then
              CMD+=" -i"
            fi
            echo $CMD
            $CMD
          fi
        done
      done
    done
  done
done

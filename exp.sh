#!/bin/bash

DENSE_SIZES=( 1000000 2000000 4000000 8000000 16000000 )
BATCH_SIZES=( 1 2 4 8 16 32 64 128 )
WORKERS=( 1 2 4 8 )
HIDDEN_SIZES=( 128 128,1024 128,1024,1024 128,1024,1024,1024 )
GPUS=( 1 2 4 )

for d in "${DENSE_SIZES[@]}"; do
  for b in "${BATCH_SIZES[@]}"; do
    for w in "${WORKERS[@]}"; do
      for h in "${HIDDEN_SIZES[@]}"; do
        for g in "${GPUS[@]}"; do
          RUN_DIR="outputs/dense-${d}_batch-${b}_workers-${w}_hidden-${h}_gpus-${g}"
          CMD="./run.sh --profile_dir $RUN_DIR/profile --log_dir $RUN_DIR/logs --dense_size ${d} --batch_size ${b} --workers ${w} --hidden_sizes ${h} --num_gpus ${g} --steps 1000000 --log_frequency 100000"
          if [[ "$NUM_GPUS" = 1 ]]; then
            CMD+=" -i"
          fi
          echo $CMD
          $CMD
        done
      done
    done
  done
done        

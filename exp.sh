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

DENSE_SIZES=( 1000000 4000000 16000000 )
BATCH_SIZES=( 8 32 64 )
WORKERS=( 1 2 4 )
HIDDEN_SIZES=( 128,1024 128,1024,1024,1024 )
GPUS=( 1 2 4 )

for d in "${DENSE_SIZES[@]}"; do
  for b in "${BATCH_SIZES[@]}"; do
    for w in "${WORKERS[@]}"; do
      for h in "${HIDDEN_SIZES[@]}"; do
        for g in "${GPUS[@]}"; do
          RUN_DIR="outputs/dense-${d}_batch-${b}_workers-${w}_hidden-${h}_gpus-${g}"
          if (! [[ -f "$RUN_DIR/logs/log" ]]) | [[ -z "$(cat $RUN_DIR/logs/log | grep Training)" ]] ; then
            rm -r $RUN_DIR 
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

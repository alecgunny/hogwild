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

import argparse
import numpy as np
import tensorflow as tf


def get_example_data():
  """
  creates dataset example with uniformly distributed nonzero entries
  edit this code to create datasets with different distributions
  """
  nnz = np.random.randint(FLAGS.min_nnz, FLAGS.max_nnz)
  nz_idx = np.random.choice(np.arange(FLAGS.dense_size), replace=False, size=nnz)
  nz_values = np.random.uniform(10, size=nnz)
  label = np.random.randint(FLAGS.num_classes)
  return nz_idx, nz_values, label


def main():
  writer = tf.python_io.TFRecordWriter(FLAGS.dataset_path)
  for row in range(FLAGS.dataset_size):
    nz_idx, nz_values, label = get_example_data()

    example = tf.train.Example(features=tf.train.Features(feature=
      {'nz_idx': tf.train.Feature(int64_list=tf.train.Int64List(value=list(nz_idx))),
       'nz_values': tf.train.Feature(float_list=tf.train.FloatList(value=list(nz_values))),
       'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
      }))

    writer.write(example.SerializeToString())
  writer.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--dataset_path',
    type=str,
    default='train.tfrecords',
    help='path to save dataset to')
  
  parser.add_argument(
    '--dataset_size',
    type=int,
    default=10000,
    help="number of examples to create in dataset")

  parser.add_argument(
    '--min_nnz',
    type=int,
    default=10,
    help='minimum number of nonzero elements per sample')

  parser.add_argument(
    "--max_nnz",
    type=int,
    default=100,
    help="maximum number of nonzero elements per sample")

  parser.add_argument(
    "--dense_size",
    type=int,
    default=1000000,
    help="dimensionality of input space")

  parser.add_argument(
    "--num_classes",
    type=int,
    default=10,
    help="dimensionality of output space, less relevant")

  FLAGS = parser.parse_args()
  main()


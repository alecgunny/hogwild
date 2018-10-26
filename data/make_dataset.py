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


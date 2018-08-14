import tensorflow as tf
import argparse
import time
import datetime
import os
import json
import numpy as np

FLAGS = None
model_dir = '/logs'


class _LoggerHook(tf.train.SessionRunHook):
  """Logs loss and runtime."""
  def __init__(self, log_frequency):
    self.log_frequency = log_frequency
    super(_LoggerHook, self).__init__()

  def _load_global_step_from_checkpoint_dir(self, checkpoint_dir):
    checkpoint_reader = tf.train.NewCheckpointReader(
      tf.train.latest_checkpoint(checkpoint_dir))
    return checkpoint_reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)

  def begin(self):
    self._start_time = time.time()
    try:
      self._init_step = self._load_global_step_from_checkpoint_dir(model_dir)
    except TypeError:
      self._init_step = 0

  def before_run(self, run_context):
    return tf.train.SessionRunArgs(tf.train.get_global_step())

  def after_run(self, run_context, run_values):
    step = run_values.results + 1
    if step % self.log_frequency == 0 and FLAGS.job_name == 'worker':
      steps_since_start = step - self._init_step
      current_time = time.time()
      duration = current_time - self._start_time

      examples_per_sec = steps_since_start * FLAGS.batch_size / duration
      sec_per_batch = duration / steps_since_start

      format_str = "Step {}: {:0.1f} examples/sec; {:0.4f} sec/batch"
      print(format_str.format(step, examples_per_sec, sec_per_batch))

  def end(self, session):
      if FLAGS.job_name == 'worker' and FLAGS.task_index == 0:
        total_time = time.time() - self._start_time
        print("Training complete, total time {:0.3f} s".format(total_time))


def main():
  n_input = 100
  n_classes = 10

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1, allow_growth=True)
  session_config = tf.ConfigProto(gpu_options=gpu_options)
  config = tf.estimator.RunConfig(
    model_dir=model_dir,
    session_config=session_config,
    save_checkpoints_steps=100)    

  columns = [tf.feature_column.numeric_column(key='col{}'.format(i), dtype=tf.float32) for i in range(n_input)]
  estimator = tf.estimator.DNNClassifier(
    FLAGS.hidden_sizes,
    columns,
    n_classes=n_classes,
    config=config)

  def train_input_fn():
    X = np.random.randn(FLAGS.batch_size*100, n_input)
    X = {'col{}'.format(i): X[:, i] for i in range(n_input)}
    y = np.random.randint(n_classes, size=FLAGS.batch_size*100)
    return tf.data.Dataset.from_tensor_slices((X,y)).repeat().batch(FLAGS.batch_size)

  train_hooks = [_LoggerHook(FLAGS.log_frequency)]
  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=FLAGS.epochs, hooks=train_hooks)
  eval_spec = tf.estimator.EvalSpec(input_fn=train_input_fn)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
    "--job_name",
    type=str,
    default="worker",
    choices=["worker", "ps", "chief"],
    help="One of 'chief', 'worker', or 'ps'")

  # Flags for defining the tf.train.Server
  parser.add_argument(
    "--task_index",
    type=int,
    default=0,
    help="Index of task within the job")

  parser.add_argument(
    "--num_tasks",
    type=int,
    default=2,
    help="Number of non chief tasks")

  # Flags for defining model properties
  parser.add_argument(
    "--hidden_sizes",
    type=int,
    nargs="+",
    default=[1024, 1024],
    help="hidden dimensions of mlp")

  parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="batch size")

  parser.add_argument(
    "--epochs",
    type=int,
    default=2000,
    help="number of epochs")

  parser.add_argument(
    "--log_frequency",
    type=int,
    default=100,
    help="number of epochs between print logging")

  FLAGS, unparsed = parser.parse_known_args()

  cluster = {
    'ps': ['localhost: 2221'],
    'chief': ['localhost:2222'],
    'worker': ['localhost:{}'.format(i+2223) for i in range(FLAGS.num_tasks)]
  }
  os.environ['TF_CONFIG'] = json.dumps(
    {'cluster': cluster,
     'task': {'type': FLAGS.job_name, 'index': FLAGS.task_index}})
  main()

# Hogwild! for TensorFlow on GPUs

GPU friendly TensorFlow implementation of Hogwild!, the sparse asynchronous optimization algorithm, introduced in <a href=https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf>this paper</a> from Berkeley (Go Bears). Leverages the TF estimator API to build a multilayer perceptron with a tf.nn.embedding_lookup_sparse operation at the front to replace the traditional matrix multiplication.

Also shows how to leverage sparse data with the dataset API. You can see the proper way of making a TFRecords dataset in `make_dataset.py`, and the proper way of loading it in `Hogwild.py`. The key is in the VarLenFeature object, which produces SparseTensors. We create SparseTensors which represent both the nonzero indices in a sparse dataset, as well as the values at those indices. To get a better idea of this, pretend the matrix below is our sparse dataset.

<img src="img/sparse_embedding.png"></img>

The tensor on the left corresponds to "nz_idx", with indices given by the left two columns and values given by the column beneath it, while the tensor on the right corresponds to the same format for "nz_values".

In this example, we use completely random dummy data created in `make_dataset.py`, which can be found in `data/train.tfrecords`. The dense size of this dataset is 1,000,000. To create a larger dataset, just run the script yourself in a TensorFlow container and set `--dense_size` to whatever you want.

This dataset also uniformly samples nonzero entries from the input space. Feel free to implement your own example-generation function in `make_dataset.py` to implement a different distrubution.

## Binary inputs
If your input space is fully binary, that is, all the nonzero values are 1, you can save a little bit of compute on the gradients by setting `sp_weights=None` in `tf.nn.embedding_lookup_sparse`, which will leverage a sparse gather op under the hood. To recreate this here, use the flag `--use_binary` to see the impact on performance.

## How to run
Nothing to build, just uses public tensorflow docker releases (doesn't leverage NGC to make agnosticism to CPU/GPU implementation simpler). All that's needed is to run `./run.sh -h` to get a sense for command line options. All profiling, logging, and model saving are done inside the container unless a local directory is specified in the `./run.sh` call.

To see how GPU training accelerates compared to CPU training, add on the `-c` flag to your call. Right now, you get the most acceleration out of larger batch sizes. The bottleneck seems to be tensor reads from CPU to GPU, which wash out the massive gains in the embedding and MatMul ops. For this reason, deeper and wider networks don't necessarily enjoy more acceleration, because this just amounts to more weights to read.

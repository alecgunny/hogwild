## HogWild!

GPU friendly TensorFlow implementation of Hogwild!, the sparse asynchronous optimization algorithm, discussed in <a href=https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf>this paper</a>. Leverages the TF estimator API to build a multilayer perceptron with a tf.nn.embedding_lookup_sparse operation at the front to replace the traditional matrix multiplication.

Inputs to the estimator are 4 dimensional, with the first two columns indexing two sparse matrices, the first of which contains the nonzero indices of the dense input (which are in the third column of our input), and the second of which contains the values at that input (which are in the fourth). This is useful in case there are order of magnitude differences in the number of nonzero elements between samples. Since we're working with dummy data, the minimum and maximum number of elements can be set from the command line with `--min_nnz` and `--max_nnz`, with the actual number drawn uniformly between them.

Worth noting that worker instances take some time to spin up (4-5 s per instance), so to get a good benchmark of the speedup it's best to run with many epochs (which here just means gradient updates for a variety of reasons) so that all workers can get online before training finishes.

To build, just use `docker build -t $USER/hogwild .`, which builds an executable Docker image which can take command line args at run time. To see all args, run

`nvidia-docker run --rm -it $USER/hogwild -h`

An example command to run would be

`NV_GPU=0 nvidia-docker run --rm -it $USER/hogwild --batch_size 512 --epochs 4000 --hidden_dims 128,1024,1024,512 --dense_size 2000000 --max_nnz 100 --log_frequency 200 --workers 4`

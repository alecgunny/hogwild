## HogWild!

GPU friendly implementation of the idea outlined <a href=https://github.com/tmulc18/Distributed-TensorFlow-Guide/tree/master/Hogwild>here</a>, which implements the method discussed in <a href=https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf>this paper</a>. Leverages the TF estimator API in the manner outlined <a href=https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate>here</a>.

Should be noted that the dummy data here is dense data, which would invalidate some of the convergence behavior in the paper above since its based primarily on the sparseness of the data in question.

Also worth noting that worker instances take some time to spin up (4-5 s per instance), so to get a good benchmark of the speedup it's best to run with many epochs so that all workers can get online before training finishes.

To build, just use `docker build -t $USER/hogwild .`, which builds an executable Docker image which can take command line args at run time. To see all args, run

`nvidia-docker run --rm -it $USER/hogwild -h`

An example command to run would be

`NV_GPU=0 nvidia-docker run --rm -it $USER/hogwild --batch_size 512 --epochs 4000 --hidden_dims 1024,512,256 --log_frequency 200 --workers 4`

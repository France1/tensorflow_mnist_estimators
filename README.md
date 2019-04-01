# tensorflow_mnist_estimators
A tutorial to train, evaluate, and serve Tensorflow Estimators models inspired by ...

## Model definition
The model consists of a CNN estimator to predict digits from the MNIST dataset from the [official Tensorflow tutorial](https://www.tensorflow.org/tutorials/estimators/cnn), and is trained in the notebook `notebooks/estimator_model.ipynb`

### Training and evaluation 
Performance metrics are calculated during training using `tf.estimator.train_and_evaluate` which allows to print and save results for both the training and evaluation datasets. The saved results can be visualised in real-time using Tensorboard by running in terminal:
```
tensorboard --logdir=models/ckpt/
```
Note that while the loss (default metrics) for the trainining and evaluation set is shown in a single plot, the additionally defined accuracy is displayed in two separated plots, and there doen't seem to be an easy way to combine them.

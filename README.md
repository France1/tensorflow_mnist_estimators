# tensorflow_mnist_estimators
A tutorial to understand how to train, evaluate, and serve Tensorflow Estimators models inspired by [this article](https://medium.com/@yuu.ishikawa/serving-pre-modeled-and-custom-tensorflow-estimator-with-tensorflow-serving-12833b4be421). Compared to this work the main additions are:
- flexible input pipeline functions for training and prediction
- a training and evaluation routine that output metrics in real time
- an alternative way to Tensorflow Serving to deploy predictions

## Model definition
The model consists of a CNN estimator to predict digits from the MNIST dataset from the [official Tensorflow tutorial](https://www.tensorflow.org/tutorials/estimators/cnn), which is trained and explained in the notebook `notebooks/estimator_model.ipynb`

### Input data
Input data are defined using `tf.data.Dataset.from_tensor_slices` as it allows more control compared to `tf.estimator.inputs.numpy_input_fn` used in the official tutorials. This produces a generator that is easier to navigate for debugging purposes

### Training and evaluation 
Performance metrics are calculated during training using `tf.estimator.train_and_evaluate` which allows to print and save results for both the training and evaluation datasets. The saved results can be visualised in real-time using Tensorboard by running in terminal:
```
tensorboard --logdir=models/ckpt/
```
Note that while the loss (default metrics) for the trainining and evaluation set is shown in a single plot, the additionally defined accuracy is displayed in two separated plots, and there doen't seem to be an easy way to combine them.

## Model save and restore
The serialized trained model can be loaded in session and used for prediction as explained in this [article](https://guillaumegenthial.github.io/serving-tensorflow-estimator.html). This is particularly useful if the deployement is done in Flask

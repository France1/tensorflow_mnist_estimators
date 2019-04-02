# tensorflow_mnist_estimators
A tutorial to understand how to train, evaluate, and serve Tensorflow Estimators models inspired by [this article](https://medium.com/@yuu.ishikawa/serving-pre-modeled-and-custom-tensorflow-estimator-with-tensorflow-serving-12833b4be421). Compared to this work the main additions are:
- flexible input pipeline functions for training and prediction
- a training and evaluation routine that output metrics in real time
- an alternative way to Tensorflow Serving to deploy predictions

## Model definition
The model consists of a CNN estimator to predict digits from the MNIST dataset from the [official Tensorflow tutorial](https://www.tensorflow.org/tutorials/estimators/cnn), which is trained in the notebook `notebooks/estimator_model.ipynb`. The model predict the number in the image as `classes` and the corresponding `probabilities` of the predictions.

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

## Model serving with Tensorflow Serving
This is done following [this article](https://medium.com/@yuu.ishikawa/serving-pre-modeled-and-custom-tensorflow-estimator-with-tensorflow-serving-12833b4be421). 
A dockerized version of tensorflo-model-server is built by running:
```
docker build --rm -f Dockerfile -t tensorflow-model-server .
```
Then the exported version of the model to deploy is saved within directory `models_for_serving/1` in which `1` represents the version of the model that tensorflow-serving can deploy:
```
mkdir -p ./models_for_serving/1
mv ./models/pb/1554128599/* ./models_for_serving/1
```
The docker container runs with `/models_for_serving` mounted as `/models` volume. It exposes port 8500 for gRPC service and port 8501 for the REST API:
```
docker run --rm  -v ${PWD}/models_for_serving:/models \
  -e MODEL_NAME='mnist' \
  -e MODEL_PATH='/models' \
  -p 8500:8500 \ 
  -p 8501:8501 \ 
  --name tensorflow-server \
  tensorflow-model-server
```
where `MODEL_PATH=/models` defines the location of the saved model inside the Docker container.

## gRPC serving
The gRPC client `python/grpc_client.py` is used to request predictions from port 8500:
```
python python/grpc_client.py \
  --image ./data/0.png \
  --model mnist
```
which should return as an output:
```
outputs {
  key: "classes"
  value {
    dtype: DT_INT64
    tensor_shape {
      dim {
        size: 1
      }
    }
    int64_val: 0
  }
}
outputs {
  key: "probabilities"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 10
      }
    }
    float_val: 1.0
    float_val: 0.0
    float_val: 0.0
    float_val: 0.0
    float_val: 0.0
    float_val: 0.0
    float_val: 0.0
    float_val: 0.0
    float_val: 0.0
    float_val: 0.0
  }
}
model_spec {
  name: "mnist"
  version {
    value: 1
  }
  signature_name: "serving_default"
}

```

## REST API serving
The REST API client `python/rest_client.py` is used to request predictions from port 8500:
```
python python/rest_client.py \
  --image ./data/0.png \
  --model mnist
```
which should return as an output:
```
200 OK
{
    "predictions": [
        {
            "probabilities": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "classes": 0
        }
    ]
}

```

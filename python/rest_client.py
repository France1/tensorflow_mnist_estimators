import argparse
import time
import numpy as np
from scipy.misc import imread
import requests

def run(host, port, image, model):

    # Read an image
    data = imread(image)
    data = data.astype(np.float32)
    print(data)

    start = time.time()

    data = np.expand_dims(data, axis=3).tolist()

    json = {
        "instances": [{"x":data}]
        # "instances": [data]   # both are valid
    }

    result = requests.post(
        'http://{host}:{port}/v1/models/{model}:predict'.format(host=host, port=port, model=model),
        json=json
    )

    end = time.time()
    time_diff = end - start

    print(result.status_code, result.reason)
    print(result.text)
    print('time elapased: {}'.format(time_diff))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', help='Tensorflow server host name', default='localhost', type=str)
    parser.add_argument('--port', help='Tensorflow server port number', default=8501, type=int)
    parser.add_argument('--image', help='input image', type=str)
    parser.add_argument('--model', help='model name', type=str)

    args = parser.parse_args()
    run(args.host, args.port, args.image, args.model)
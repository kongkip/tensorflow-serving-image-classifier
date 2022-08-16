# TensorFlow Serving
This repos shows how to serve a tensorflow model with tensorflow serving using docker

## Saving a keras model as tensorflow saved model
Saving a keras image classification model

```python
import tensorflow as tf 


model  = tf.keras.Sequential([ 
    tf.keras.layers.Conv2D(32, (3, 3), input_shape=(256, 256, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossEntropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

tf.saved_model.save(model, "my_model/saved_model/1/")
```

## How to serve

Pull the tensorflow serving docker image

```bash
docker pull tensorflow/serving
```

Run a  serving_base image
```bash
docker run -d --name serving_base tensorflow/serving
```

Copy the saved model into serving_base container's model folder
```bash
docker cp my_model/saved_model serving_base:/models/classifier
```

```bash
docker commit --change "ENV MODEL_NAME classifier" serving_base classifier
```

Kill the serving_base since we don't need it
```bash
docker kill serving_base
```

Run the image to serve our SavedModel as a daemon and we map the ports 8501

```bash
docker run -d -p 8501:8501 -p 8500:8500 --name classifier classifier
```


## Using the tf-serving model From REST API

```python
import requests
from PIL import Image
import numpy as np 
import tensorflow as tf
import json


image = Image.open("example_image.jpg")
image_np = np.array(image)
image_np = tf.expand_dims(tf.cast(np.resize(image, (256, 256)), tf.float32), -1)
image_np = tf.image.grayscale_to_rgb(image_np)
image_np = tf.expand_dims(image_np, 0)


data =json.dumps({
        "signature_name": "serving_default",
        "instances": image_np.numpy().tolist()
    })

r = requests.post("http://localhost:8501/v1/models/classifier:predict", data=data)

# Getting probabilities 
result = json.loads(r.text)["predictions"][0]

print(result)
```

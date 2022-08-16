# TensorFlow Serving
This repos shows how to serve a tensorflow model with tensorflow serving using docker

## Saving a keras model as tensorflow saved model
Saving a keras image classification model

```python
import tensorflow as tf 


model  = tf.keras.Sequential([ 
    tf.keras.layers.Conv2D(32, (3, 3), input_shape=(256, 256), activation="relu"),
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

```bash
docker pull tensorflow/serving
```

```bash
docker run -d --name linknet_serving tensorflow/serving
```

```bash
docker cp my_model/saved_model linknet_serving:/models/linknet
```

```bash
docker commit --change "ENV MODEL_NAME linknet" linknet_serving linknet
```

```bash
docker kill linknet_serving
```

Run the image to serve our SavedModel as a daemon and we map the ports 8501

```bash
docker run -d -p 8501:8501 -p 8500:8500 --name linknet linknet
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

r = requests.post("http://localhost:8501/v1/models/linknet:predict", data=data)

# Getting probabilities 
result = json.loads(r.text)["predictions"][0]

print(result)
```
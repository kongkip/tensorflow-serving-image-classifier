import requests
from PIL import Image
import numpy as np 
import tensorflow as tf
import json


image = Image.open("example_image.jpg")
image_np = np.array(image)
image_np = tf.expand_dims(tf.cast(np.resize(image, (256, 256)), tf.float32) / 255.0, -1)
image_np = tf.image.grayscale_to_rgb(image_np)
image_np = tf.expand_dims(image_np, 0)


data =json.dumps({
        "signature_name": "serving_default",
        "instances": image_np.numpy().tolist()
    })

r = requests.post("http://localhost:8501/v1/models/linknet:predict", data=data)

result = json.loads(r.text)["predictions"][0]

print(result)
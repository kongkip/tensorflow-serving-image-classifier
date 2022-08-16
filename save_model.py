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



from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf


def _shortcut(inputs, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = tf.keras.backend.int_shape(inputs)
    residual_shape = tf.keras.backend.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    shortcut = inputs
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = tf.keras.layers.Conv2D(filters=residual_shape[3],
                                          kernel_size=(1, 1),
                                          strides=(stride_width, stride_height),
                                          padding="valid",
                                          kernel_initializer="he_normal",
                                          kernel_regularizer=tf.keras.regularizers.L2(0.0001))(inputs)

    return tf.keras.layers.add([shortcut, residual])


def encoder_block(input_tensor, m, n):
    x = tf.keras.layers.BatchNormalization()(input_tensor)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=n, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)

    added_1 = _shortcut(input_tensor, x)

    x = tf.keras.layers.BatchNormalization()(added_1)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)

    added_2 = _shortcut(added_1, x)

    return added_2


def decoder_block(input_tensor, m, n):
    x = tf.keras.layers.BatchNormalization()(input_tensor)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=int(m / 4), kernel_size=(1, 1))(x)

    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=int(m / 4), kernel_size=(3, 3), padding='same')(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=n, kernel_size=(1, 1))(x)

    return x


def global_average_pooling(x):
    return tf.keras.backend.mean(x, axis=(2, 3))


def global_average_pooling_shape(input_shape):
    return input_shape[0:2]


def LinkNet(input_shape=(256, 256, 3), classes=1, include_classification_layer=True):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2))(x)

    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    encoder_1 = encoder_block(input_tensor=x, m=64, n=64)

    encoder_2 = encoder_block(input_tensor=encoder_1, m=64, n=128)

    encoder_3 = encoder_block(input_tensor=encoder_2, m=128, n=256)

    encoder_4 = encoder_block(input_tensor=encoder_3, m=256, n=512)

    decoder_4 = decoder_block(input_tensor=encoder_4, m=512, n=256)

    decoder_3_in = tf.keras.layers.add([decoder_4, encoder_3])
    decoder_3_in = tf.keras.layers.Activation('relu')(decoder_3_in)

    decoder_3 = decoder_block(input_tensor=decoder_3_in, m=256, n=128)

    decoder_2_in = tf.keras.layers.add([decoder_3, encoder_2])
    decoder_2_in = tf.keras.layers.Activation('relu')(decoder_2_in)

    decoder_2 = decoder_block(input_tensor=decoder_2_in, m=128, n=64)

    decoder_1_in = tf.keras.layers.add([decoder_2, encoder_1])
    decoder_1_in = tf.keras.layers.Activation('relu')(decoder_1_in)

    decoder_1 = decoder_block(input_tensor=decoder_1_in, m=64, n=64)

    x = tf.keras.layers.UpSampling2D((2, 2))(decoder_1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)

    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    segs = tf.keras.layers.Conv2D(filters=classes, kernel_size=(2, 2), padding="same", activation="sigmoid")(x)

    # classification layers
    if include_classification_layer:
        cls = tf.keras.layers.Flatten(name="classification_flatten")(encoder_4)
        cls = tf.keras.layers.Dense(3, activation="softmax", name="classification_softmax")(cls)
        model = tf.keras.Model(inputs=inputs, outputs=[segs, cls])
    else:
        model = tf.keras.Model(inputs=inputs, outputs=[segs])
    return model


if __name__ == '__main__':
    import pathlib
    import yaml

    # from helpers.viz_utils import plot_single_predictions
    # from dataset import ConcatDataset

    _model = LinkNet()

    _model.summary()

    # params = yaml.safe_load(open("../../params.yaml"))["train"]
    # root = "D:\\Work\Tambua\\Repos\\AI\\fetal-seg-transfer-learning\\data"
    # HEAD_DATA_ROOT = pathlib.Path(f"{root}\\head")
    # ABDOMEN_DATA_ROOT = pathlib.Path(f"{root}\\abdomen")
    # FEMUR_DATA_ROOT = pathlib.Path(f"{root}\\femur")
    # IMAGE_SIZE = params["image_size"]
    # BATCH_SIZE = params["batch_size"]
    #
    # data = ConcatDataset(
    #     head_path=HEAD_DATA_ROOT,
    #     abdomen_path=ABDOMEN_DATA_ROOT,
    #     femur_path=FEMUR_DATA_ROOT,
    #     image_size=IMAGE_SIZE
    # )
    #
    # data.load_data()
    #
    # for img, mask in data.train.take(1):
    #     pred = _model.predict(tf.expand_dims(img, axis=0))
    #     # print("Predictions Shape", pred.shape)
    #     # plot_single_predictions([img, mask, pred[0]])
    #     print(pred)

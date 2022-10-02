import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#   Preprocess the data
x_train = x_train.reshape(60000, 28, 28, 1).astype("float32") / 255
x_test = x_test.reshape(10000, 28, 28, 1).astype("float32") / 255
y_train = tf.one_hot(y_train, 10, dtype = "float32")
y_test = tf.one_hot(y_test, 10, dtype = "float32")

inputs = keras.Input(shape=(28, 28, 1), name="digits")

# #   Encoder
# x = layers.Conv2D(30, 3, activation="relu", padding="same")(inputs)
# x = layers.MaxPooling2D(2)(x)
# x = layers.Conv2D(30, 3, activation="relu", padding="same")(x)
# x = layers.MaxPooling2D(2)(x)

# #   Decoder
# x = layers.Conv2DTranspose(30, 3, strides=2, activation="relu", padding="same")(x)
# x = layers.Conv2DTranspose(30, 3, strides=2, activation="relu", padding="same")(x)
# x = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

#   Encoder
x = layers.Conv2D(10, 3, activation="relu")(inputs)
x = layers.Conv2D(10, 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(10, 3, activation="relu")(x)
x = layers.Conv2D(10, 3, activation="relu")(x)
x = layers.GlobalMaxPooling2D()(x)
encoded = layers.Reshape((1, 1, 10))(x)

#   Decoder
x = layers.Conv2DTranspose(30, 2, strides=2, activation="relu")(encoded)
x = layers.Conv2DTranspose(30, 3, strides=3, activation="relu")(x)
x = layers.Conv2DTranspose(30, 3, strides=5, activation="relu")(x)
x = layers.Conv2D(1, 3, activation="sigmoid")(x)

# Autoencoder
autoencoder = keras.Model(inputs, x)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()

autoencoder.fit(
    x=x_train,
    y=x_train,
    epochs=5,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test, x_test),
)

a = autoencoder.predict(x_test)
b = Image.fromarray(a[0, :, :, 0])
# b.show()
b.save('y.tiff')
Image.fromarray(x_test[0, :, :, 0]).save('x.tiff')

encoder = keras.Model(inputs, encoded)
a = encoder(x_train)
b = a[:, 0, 0, :]



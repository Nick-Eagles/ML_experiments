import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#   Preprocess the data (these are NumPy arrays)
x_train = x_train.reshape(60000, 28, 28, 1).astype("float32") / 255
x_test = x_test.reshape(10000, 28, 28, 1).astype("float32") / 255

#   Create a conv net using the functional API
inputs = keras.Input(shape=(28, 28, 1), name="digits")
x = layers.Conv2D(30, 3, activation="relu")(inputs)
x = layers.Conv2D(30, 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(30, 3, activation="relu")(x)
x = layers.GlobalMaxPooling2D()(x)
outputs = layers.Dense(10, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.SGD(momentum=0.9),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=3,
    validation_data=(x_test, y_test),
)

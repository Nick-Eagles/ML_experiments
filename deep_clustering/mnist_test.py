import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#   Preprocess the data
x_train = x_train.reshape(60000, 28, 28, 1).astype("float32") / 255
x_test = x_test.reshape(10000, 28, 28, 1).astype("float32") / 255
y_train = tf.one_hot(y_train, 10, dtype = "float32")
y_test = tf.one_hot(y_test, 10, dtype = "float32")

#   Create a conv net using the functional API
inputs = keras.Input(shape=(28, 28, 1), name="digits")
x = layers.Conv2D(30, 3, activation="relu", kernel_regularizer='l2')(inputs)
x = layers.Conv2D(30, 3, activation="relu", kernel_regularizer='l2')(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(30, 3, activation="relu", kernel_regularizer='l2')(x)
x = layers.GlobalMaxPooling2D()(x)
outputs = layers.Dense(10, activation="softmax", kernel_regularizer='l2')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
model.summary()

alpha = 1
@tf.autograph.experimental.do_not_convert
def custom_loss(y_true, y_pred):
    #ent = -1 * tf.math.reduce_sum(y_pred * tf.math.log(y_pred))
    
    #   Try to maximize variance across the batch while rewarding low-entropy
    #   (decisive) output
    var = tf.math.reduce_variance(y_pred, axis = 0)
    ent_var = -1 * tf.math.reduce_sum(var * tf.math.log(var))
    a = tf.math.reduce_sum(var)
    
    #return alpha * ent - var
    return -10 * ent_var - 10 * a

model.compile(
    # optimizer=tf.keras.optimizers.SGD(momentum=0.9, learning_rate = 0.001),
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
    # loss=keras.losses.CategoricalCrossentropy(),
    loss = custom_loss,
    metrics=[keras.metrics.CategoricalAccuracy()],
)

history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=3,
    validation_data=(x_test, y_test),
)

a = model.predict(x_test)

var = tf.math.reduce_variance(a, axis = 0)
ent_var = -1 * tf.math.reduce_sum(var * tf.math.log(var))

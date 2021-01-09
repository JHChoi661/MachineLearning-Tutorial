### 20210109 머신러닝 야학 이미지 분류 - CNN LeNet 5 ###

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

### mnist
(inV_mnist, deV_mnist), _ = tf.keras.datasets.mnist.load_data()
print(inV_mnist.shape, deV_mnist.shape)
inV_mnist = inV_mnist.reshape(60000, 28, 28, 1) 
deV_mnist = pd.get_dummies(deV_mnist) 
print(inV_mnist.shape, deV_mnist.shape)
print(deV_mnist.iloc[0])

X = tf.keras.layers.Input(shape=[28, 28, 1])

H = tf.keras.layers.Conv2D(6, kernel_size=5, padding='same', activation='swish')(X) # input size와 동일하게 padding
H = tf.keras.layers.MaxPool2D()(H)
H = tf.keras.layers.Conv2D(16, kernel_size=5, activation='swish')(H)
H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Flatten()(H)
H = tf.keras.layers.Dense(120, activation='swish')(H)
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')
model.summary()
model.fit(inV_mnist, deV_mnist, epochs=10)
pred = model.predict(inV_mnist[0:5])
pd.DataFrame(pred).round(2)
print(deV_mnist[0:5])


### cifar 10
(inV_cifar10, deV_cifar10), _ = tf.keras.datasets.cifar10.load_data()
print(inV_cifar10.shape, deV_cifar10.shape)
deV_cifar10 = pd.get_dummies(deV_cifar10.reshape(50000)) 
print(inV_cifar10.shape, deV_cifar10.shape)
inV_cifar10 = inV_cifar10 / 255. # Normalization

print(deV_cifar10.iloc[0])

X = tf.keras.layers.Input(shape=[32, 32, 3])

H = tf.keras.layers.Conv2D(6, kernel_size=5, activation='swish')(X)
H = tf.keras.layers.MaxPool2D()(H)
H = tf.keras.layers.Conv2D(16, kernel_size=5, activation='swish')(H)
H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Flatten()(H)

H = tf.keras.layers.Dense(120)(H)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Activation('swish')(H)

H = tf.keras.layers.Dense(84)(H)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Activation('swish')(H)

Y = tf.keras.layers.Dense(10, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')
model.summary()
model.fit(inV_cifar10, deV_cifar10, epochs=10) # accuracy : 0.7196
pred = model.predict(inV_cifar10[10:15])
pd.DataFrame(pred).round(2)
print(deV_cifar10[10:15])

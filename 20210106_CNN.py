### 20210106 머신러닝 야학 이미지 분류 ###

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

### reshape를 이용한 이미지 학습
(inV_mnist, deV_mnist), _ = tf.keras.datasets.mnist.load_data()
print(inV_mnist.shape, deV_mnist.shape)

inV_mnist = inV_mnist.reshape(60000, 784) # 784에 -1을 주면 알아서 784로 배정
deV_mnist = pd.get_dummies(deV_mnist)
print(inV_mnist.shape, deV_mnist.shape)
print(deV_mnist.iloc[0])

X = tf.keras.layers.Input(shape=[784])
H = tf.keras.layers.Dense(84, activation='swish')(X)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

model.fit(inV_mnist, deV_mnist, epochs=10)
pred = model.predict(inV_mnist[0:5])
pd.DataFrame(pred).round(2)
print(deV_mnist[0:5])

### flatten layer를 이용한 이미지 학습
(inV_mnist, deV_mnist), _ = tf.keras.datasets.mnist.load_data()
print(inV_mnist.shape, deV_mnist.shape)

deV_mnist = pd.get_dummies(deV_mnist)
print(inV_mnist.shape, deV_mnist.shape)
print(deV_mnist.iloc[0])

X = tf.keras.layers.Input(shape=[28, 28])
H = tf.keras.layers.Flatten()(X)
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

model.fit(inV_mnist, deV_mnist, epochs=10)
pred = model.predict(inV_mnist[0:5])
pd.DataFrame(pred).round(2)
print(deV_mnist[0:5])

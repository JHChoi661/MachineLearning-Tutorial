### 20210107 머신러닝 야학 이미지 분류 ###

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

### Convolution Layer를 이용한 이미지 학습
# filter set을 몇 개 사용할 것인가?
# filter size를 얼마로 할 것인가?

(inV_mnist, deV_mnist), _ = tf.keras.datasets.mnist.load_data()
print(inV_mnist.shape, deV_mnist.shape)
inV_mnist = inV_mnist.reshape(60000, 28, 28, 1) # for convolution layer
deV_mnist = pd.get_dummies(deV_mnist) # onehot-encoding
print(inV_mnist.shape, deV_mnist.shape)
print(deV_mnist.iloc[0])

X = tf.keras.layers.Input(shape=[28, 28, 1])
H = tf.keras.layers.Conv2D(3, kernel_size=5, activation='swish')(X)
H = tf.keras.layers.Conv2D(6, kernel_size=5, activation='swish')(H)
H = tf.keras.layers.Flatten()(H)
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')
model.summary()
model.fit(inV_mnist, deV_mnist, epochs=10)
pred = model.predict(inV_mnist[0:5])
pd.DataFrame(pred).round(2)
print(deV_mnist[0:5])

##### Conv layer가 추가됨에 따라, layer구성에 필요한 가중치의 개수가
##### 크게 늘어남
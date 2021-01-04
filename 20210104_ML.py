#20210104 머신러닝 야학 Tensorflow 실습

import pandas as pd
import tensorflow as tf

# lemonade 판매량 예측 -- 독립변수:1, 종속변수:1
url_lemonade = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'

lemonade = pd.read_csv(url_lemonade)

print(lemonade.shape)

print(lemonade.columns)

inV_lemonade = lemonade[['온도']]
deV_lemonade = lemonade[['판매량']]

print(lemonade.head(5))

X_lemonade = tf.keras.layers.Input(shape=[1])
Y_lemonade = tf.keras.layers.Dense(1)(X_lemonade)
model_lemonade = tf.keras.models.Model(X_lemonade, Y_lemonade)
model_lemonade.compile(loss='mse')

model_lemonade.fit(inV_lemonade, deV_lemonade, epochs=10000, verbose=0)
model_lemonade.predict(inV_lemonade)
model_lemonade.predict([5121])

# Boston 집 값 예측 -- 독립변수:>1, 종속변수:1
url_boston = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
boston = pd.read_csv(url_boston)
print(boston.shape)
print(boston.columns)
inV_boston = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', \
        'ptratio', 'b', 'lstat']]
deV_boston = boston[['medv']]
print(inV_boston.shape, deV_boston.shape)
X_boston = tf.keras.layers.Input(shape=[13])
Y_boston = tf.keras.layers.Dense(1)(X_boston)
model_boston = tf.keras.models.Model(X_boston, Y_boston)
model_boston.compile(loss='mse')

model_boston.fit(inV_boston, deV_boston, epochs=1000, verbose=0)
model_boston.fit(inV_boston, deV_boston, epochs=10)
print(boston.medv.head(5))
model_boston.predict(inV_boston[0:5])
model_boston.get_weights()


url_iris = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'
iris = pd.read_csv(url_iris)
print(iris.shape)
print(iris.columns) 
inV_iris = iris[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
deV_iris = iris[['품종']]
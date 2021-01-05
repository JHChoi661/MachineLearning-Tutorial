### 20210105 머신러닝 야학 using batchNormalization layer ###

import pandas as pd
import tensorflow as tf

url_boston = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
boston = pd.read_csv(url_boston)
print(boston.shape)
print(boston.columns)
inV_boston = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', \
        'ptratio', 'b', 'lstat']]
deV_boston = boston[['medv']]
print(inV_boston.shape, deV_boston.shape) 

X_boston = tf.keras.layers.Input(shape=[13])
# using batchNormalization layer
H_boston = tf.keras.layers.Dense(10)(X_boston)
H_boston = tf.keras.layers.BatchNormalization()(H_boston)
H_boston = tf.keras.layers.Activation('swish')(H_boston)

H_boston = tf.keras.layers.Dense(10)(H_boston)
H_boston = tf.keras.layers.BatchNormalization()(H_boston)
H_boston = tf.keras.layers.Activation('swish')(H_boston)

H_boston = tf.keras.layers.Dense(10)(H_boston)
H_boston = tf.keras.layers.BatchNormalization()(H_boston)
H_boston = tf.keras.layers.Activation('swish')(H_boston)

Y_boston = tf.keras.layers.Dense(1)(H_boston)
model_boston = tf.keras.models.Model(X_boston, Y_boston)
model_boston.compile(loss='mse')

model_boston.summary()

# loss 가 19 ~ 21 에서 batchNormalization 적용하였더니 9 ~ 10 으로 감소
model_boston.fit(inV_boston, deV_boston, epochs=1000, verbose=0)
model_boston.fit(inV_boston, deV_boston, epochs=10)

print(boston.medv.head(5))
model_boston.predict(inV_boston[0:5]) 
model_boston.get_weights() 
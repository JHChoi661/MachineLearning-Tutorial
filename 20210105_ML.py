### 20210105 머신러닝 야학 Tensorflow 실습 ###

import pandas as pd
import tensorflow as tf

##### 범주형 데이터의 분류 : classification
url_iris = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'
iris = pd.read_csv(url_iris)
# 범주형 데이터인 '품종' column을 수치형으로 바꿔주기 위한 onehot-encoding
iris = pd.get_dummies(iris)
print(iris.columns) # '품종'이 3가지 범주를 가지기 때문에 col이 3개로 분리

inV_iris = iris[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
deV_iris = iris[['품종_setosa', '품종_versicolor', '품종_virginica']]
print(inV_iris.shape, deV_iris.shape)

X = tf.keras.layers.Input(shape=[4])
# Activation function 으로 확률로 나타내기 위한 softmax 함수 선택 - 입력 값을 정규화 해줌
Y = tf.keras.layers.Dense(3, activation='softmax')(X)
model = tf.keras.models.Model(X, Y)
# Crossentropy는 y = logx 와 비슷하다. https://blog.naver.com/riverrun17/221902229496 참고
model.compile(loss='categorical_crossentropy', metrics='accuracy')

model.fit(inV_iris, deV_iris, epochs=1000, verbose=0)
model.fit(inV_iris, deV_iris, epochs=10)

model.predict(inV_iris[-5:])

model.get_weights()


##### Hidden Layer 추가
### regression
url_boston = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
boston = pd.read_csv(url_boston)
print(boston.shape)
print(boston.columns)
inV_boston = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', \
        'ptratio', 'b', 'lstat']]
deV_boston = boston[['medv']]
print(inV_boston.shape, deV_boston.shape) 

X_boston = tf.keras.layers.Input(shape=[13])
H_boston = tf.keras.layers.Dense(10, activation='swish')(X_boston)
Y_boston = tf.keras.layers.Dense(1)(H_boston)
model_boston = tf.keras.models.Model(X_boston, Y_boston)
model_boston.compile(loss='mse')

# layer 확인
model_boston.summary()

model_boston.fit(inV_boston, deV_boston, epochs=1000, verbose=0)
model_boston.fit(inV_boston, deV_boston, epochs=10)

print(boston.medv.head(5))
model_boston.predict(inV_boston[0:5]) 
model_boston.get_weights() 

### classification
url_iris = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'
iris = pd.read_csv(url_iris)
iris = pd.get_dummies(iris)
print(iris.columns)

inV_iris = iris[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
deV_iris = iris[['품종_setosa', '품종_versicolor', '품종_virginica']]
print(inV_iris.shape, deV_iris.shape)

X = tf.keras.layers.Input(shape=[4])
H = tf.keras.layers.Dense(8, activation='swish')(X)
H = tf.keras.layers.Dense(8, activation='swish')(H)
H = tf.keras.layers.Dense(8, activation='swish')(H)
Y = tf.keras.layers.Dense(3, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

# layer 확인
model.summary()

model.fit(inV_iris, deV_iris, epochs=1000, verbose=0)
model.fit(inV_iris, deV_iris, epochs=10)

model.predict(inV_iris[-5:])
deV_iris[-5:]
model.get_weights()
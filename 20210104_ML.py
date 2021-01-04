### 20210104 머신러닝 야학 Tensorflow 실습 ###

import pandas as pd
import tensorflow as tf

##### lemonade 판매량 예측 -- 독립변수:1, 종속변수:1
url_lemonade = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
lemonade = pd.read_csv(url_lemonade)
print(lemonade.shape)
print(lemonade.columns)
# 독립변수와 종속변수의 분리
inV_lemonade = lemonade[['온도']]
deV_lemonade = lemonade[['판매량']]

print(lemonade.head(5))

### 모델 생성
X_lemonade = tf.keras.layers.Input(shape=[1])
Y_lemonade = tf.keras.layers.Dense(1)(X_lemonade)
model_lemonade = tf.keras.models.Model(X_lemonade, Y_lemonade)
model_lemonade.compile(loss='mse') #Mean Squared Error
### 모델 학습
model_lemonade.fit(inV_lemonade, deV_lemonade, epochs=10000, verbose=0)
### 데이터로 학습 결과 확인
model_lemonade.predict(inV_lemonade) # 학습 데이터
model_lemonade.predict([5121]) # 테스트 데이터

##### Boston 집 값 예측 -- 독립변수:13, 종속변수:1
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
model_boston.predict(inV_boston[0:5]) # 학습 데이터로 학습 결과 확인
model_boston.get_weights() # 변수 별 가중치 확인
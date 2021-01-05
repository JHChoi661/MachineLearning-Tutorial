### 20210105 머신러닝 야학 데이터 전처리 ###

import pandas as pd

url_iris = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris2.csv'
iris = pd.read_csv(url_iris)


### type casting
# 범주형 '품종' col 이 num data로 인식되어 onehot-encoding 불가
print(iris.head())
print(iris.dtypes)
# '품종'을 범주형으로 casting
iris['품종'] = iris['품종'].astype('category')
iris = pd.get_dummies(iris)
print(iris.head())


### fill NaN values
# NaN value 가 있는 column 찾기
iris.isna().sum()
# NaN value 가 있는 row, index 찾기
iris[iris['꽃잎폭'].isna()].index
print(iris.tail())
# fiilna with mean value
meanV = iris['꽃잎폭'].mean()
iris['꽃잎폭'] = iris['꽃잎폭'].fillna(meanV)
print(iris.tail())
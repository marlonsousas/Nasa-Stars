 __author__ = "Marlon"
# Marlon Sousa
#marlonsousa.medium.com

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Stars.csv")

df.head()


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


df['Color'].value_counts()

df['Color_'] = LabelEncoder().fit_transform(df['Color'])

df.drop('Color', axis=1, inplace=True)

df['Spectral_Class'] = LabelEncoder().fit_transform(df['Spectral_Class'])

x = df[['Temperature', 'L', "R", "A_M", "Spectral_Class", "Color_"]]
y = df[["Type"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.linear_model import LogisticRegression, LinearRegression

lr = LogisticRegression()
lr.fit(x_train, y_train)

predict_lr = lr.predict(x_test)

lrn = LinearRegression()
lrn.fit(x_train, y_train)

predict_lrn = lrn.predict(x_test)

from sklearn.metrics import accuracy_score, r2_score

print("{:.2f}%" .format(accuracy_score(predict_lr, y_test)*100))
print("{:.2f}%" .format(r2_score(predict_lrn, y_test)*100))

predict_lrn.shape

predict_lr.shape

x = df[["Temperature", "L", "R", "A_M", "Spectral_Class", "Color_"]].values
y = df[["Type"]].values

from keras.utils import np_utils

x = df[["Temperature", "L", "R", "A_M", "Spectral_Class", "Color_"]].values
y = df[["Type"]].values

labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(y)
classe_dummy = np_utils.to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, classe_dummy, test_size=0.2)


# Rede Neural

import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential

classificador = Sequential()

classificador.add(Dense(units = 10, activation = 'relu', input_dim = 6))
classificador.add(Dense(units = 10, activation = 'relu'))
classificador.add(Dense(units = 6, activation = 'softmax'))
classificador.compile(optimizer = 'adam', loss="categorical_crossentropy",
                      metrics = ['categorical_accuracy'])


classificador.summary()

classificador.fit(x_train, y_train, batch_size = 10,
                  epochs = 1000, validation_data=(x_test, y_test))


resultado = classificador.evaluate(x_test, y_test)
previsoes = classificador.predict(x_test)

from sklearn.metrics import confusion_matrix

sns.heatmap(previsoes)
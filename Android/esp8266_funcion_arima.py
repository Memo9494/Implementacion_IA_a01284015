import numpy as np
import pandas as pd

minuto = []
temperatura = []

#maxima temperatura
max_temp = 40.0
minima_temp = 20.0
ruido = 0.5

def temperatura_(max_temp, minima_temp, ruido, minuto):
    #número random 0,1,-1
    spike =np.random.randint(-1,1)
    r = np.random.randint(-1,1)

    return minima_temp + (max_temp - minima_temp)*np.cos(355+2*np.pi*minuto/1440) + ruido*r + spike

dias = 1440
for i in range(dias):
    minuto.append(i)
    temperatura.append(temperatura_(max_temp, minima_temp, ruido, i))

df = pd.DataFrame({'minuto':minuto,'temperatura':temperatura})
df.to_csv('temperatura.csv', index=False, sep=';')


#split temperatura ascendente y descendente
tiempoa = []
tiempod = []
temperaturaa = []
temperaturad = []
for i in range(len(minuto)):
    if i < 720:
        tiempoa.append(minuto[i])
        temperaturaa.append(temperatura[i])
    else:
        tiempod.append(minuto[i])
        temperaturad.append(temperatura[i])




import matplotlib.pyplot as plt
# plt.plot(minuto, temperatura)
# plt.show()



# Leer el archivo csv
df = pd.read_csv('/Users/guillermocepeda/C:C++/Implementacion_IA_a01284015/Android/temperatura.csv', sep=';')

print(df.head())

# x = df['minuto'].values
# y = df['temperatura'].values

# Dividir los datos en entrenamiento y prueba x sólo las variables que se consideran importantes length, margin_low y margin_up
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.model_selection import train_test_split
# X = df[['minuto']]
# y = df['temperatura']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #Train the model ETS
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# model = ExponentialSmoothing(y_train, trend='add', seasonal='add', seasonal_periods=1440)
# model_fit = model.fit()
# y_pred = model_fit.predict(start=len(y_train), end=len(y_train)+len(y_test)-1)

# #Plot the results
# plt.plot(y_test.values, label='Real')
# plt.plot(y_pred.values, label='Predicción')
# plt.legend()
# plt.show()

#Train linear regression
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# from sklearn.model_selection import train_test_split
X = df[['minuto']]
y = df['temperatura']

# model.fit(X, y)
# y_pred = model.predict(X)

#Plot the results
# plt.plot(X, y, label='Real')
# plt.plot(X, y_pred, label='Predicción')
# plt.legend()
# plt.show()

#Train ARIMA
# from statsmodels.tsa.arima.model import ARIMA

# model = ARIMA(y, order=(5,1,0))
# model_fit = model.fit()
# y_pred = model_fit.predict(start=len(y), end=len(y)+len(y)-1)


#Plot the results
# plt.plot(y, label='Real')
# plt.plot(y_pred, label='Predicción')
# plt.legend()
# plt.show()


#usa una red neuronal para predecir la temperatura
# from sklearn.neural_network import MLPRegressor

# model = MLPRegressor(hidden_layer_sizes=(10,10,10,10,10,10), max_iter=10000)
# model.fit(X, y)
# y_pred = model.predict(X)

# #Plot the results
# plt.plot(X, y, label='Real')
# plt.plot(X, y_pred, label='Predicción')
# plt.legend()
# plt.show()

#print the model
# print(model)

#Cosenoidal neural network
# from sklearn.neural_network import MLPRegressor

# model = MLPRegressor(hidden_layer_sizes=(100,100,100,100), max_iter=1000)
# model.fit(X, y)
# y_pred = model.predict(X)

#Importar el modelo para usarlo con micropython
# import joblib
# from joblib import dump
# dump(model,'modelo_neuronal.joblib')

# import ujson
# with open('modelo_neuronal.json', 'w') as f:
#     ujson.dump(model, f)
    


# import pickle
# with open('modelo_neuronal.pkl', 'wb') as f:
#     pickle.dump(model, f)

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
feature_columns = [tf.feature_column.numeric_column('minuto', shape=[1])]
Xraw = df[['minuto']]
y = df['temperatura']
#import standard scaler
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense
scaler = StandardScaler()
X = scaler.fit_transform(Xraw)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = keras.models.Sequential([
    Dense(10,activation='relu',input_shape=(1,),kernel_initializer='random_uniform', bias_initializer="zeros"), 
    Dense(10,activation='relu',kernel_initializer='random_uniform', bias_initializer="zeros"),
#    Dense(15,activation='relu',kernel_initializer='random_uniform', bias_initializer="zeros"), 
#    Dense(10,activation='relu',kernel_initializer='random_uniform', bias_initializer="zeros"), 
    Dense(1)
])
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01,momentum=0.09),
    loss='mean_squared_error',
    metrics=['mae']
)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
loss = model.history.history['loss']
val_loss = model.history.history['val_loss']
epochs = range(1, len(loss) + 1)
SKIP = 10
plt.plot(epochs[SKIP:], loss[SKIP:], 'b.', label='Training Loss')
plt.plot(epochs[SKIP:], val_loss[SKIP:], 'r.', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#print the accuracy
loss, accuracy = model.evaluate(X, y)
print("Accuracy = {:.2f}".format(accuracy))


weights = model.get_weights()
print(weights)
file = open("hyper_param.txt", "w+")
content = str(weights)
file.write(content)
file.close()
y_pred = model.predict(X)
#Plot the results
plt.plot(Xraw, y, label='Real')
plt.plot(Xraw, y_pred, label='Predicción')
plt.legend()
plt.show()



# extract the predicted probabilities
p_pred = model.predict(X)
p_pred = p_pred.flatten()
print(p_pred.round(2))
# [1. 0.01 0.91 0.87 0.06 0.95 0.24 0.58 0.78 ...

# extract the predicted class labels
y_pred = np.where(p_pred > 0.5, 1, 0)
print(y_pred)
# import the metrics class
# from sklearn import metrics
# cnf_matrix = metrics.confusion_matrix(y, y_pred)
# cnf_matrix



#Plotear prediccion contra real
# plt.plot(Xraw, y, label='Real')
# plt.plot(Xraw, y_pred, label='Predicción')
# plt.legend()
# plt.show()


#Entrenar una red neuronal para temperatura asendente con tinyml
# import tensorflow as tf
# from tensorflow import keras
# from sklearn.model_selection import train_test_split
# feature_columns = [tf.feature_column.numeric_column('minuto', shape=[1])]
# dfa = pd.DataFrame({'minuto':tiempoa,'temperatura':temperaturaa})
# Xraw = dfa[['minuto']]
# y = dfa['temperatura']
# #import standard scaler
# from sklearn.preprocessing import StandardScaler
# from keras.layers import Dense
# scaler = StandardScaler()
# #split
# X = scaler.fit_transform(Xraw)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# modela = tf.keras.Sequential()
# modela = tf.keras.Sequential([
#     Dense(5, activation='relu', input_shape=(1,)),  # Capa oculta 1 con 10 neuronas y función de activación ReLU
#     Dense(1,activation='relu')  # Capa de salida (una única neurona para la predicción de temperatura)







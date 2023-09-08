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

import matplotlib.pyplot as plt
plt.plot(minuto, temperatura)
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
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(hidden_layer_sizes=(10,10,10,10,10,10), max_iter=10000)
model.fit(X, y)
y_pred = model.predict(X)

#Plot the results
plt.plot(X, y, label='Real')
plt.plot(X, y_pred, label='Predicción')
plt.legend()
# plt.show()

#print the model
print(model)

#Cosenoidal neural network
# from sklearn.neural_network import MLPRegressor

# model = MLPRegressor(hidden_layer_sizes=(100,100,100,100), max_iter=1000)
# model.fit(X, y)
# y_pred = model.predict(X)

#Importar el modelo para usarlo con micropython
import joblib
from joblib import dump
dump(model,'modelo_neuronal.joblib')


# import pickle
# with open('modelo_neuronal.pkl', 'wb') as f:
#     pickle.dump(model, f)







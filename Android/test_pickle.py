import pickle 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importar el modelo
with open('modelo_neuronal.pkl', 'rb') as f:
    modelo_cargado = pickle.load(f)

#array del 1 al 1440
x = np.arange(1,1441)

predicciones = modelo_cargado.predict(x.reshape(-1,1))

#Plot the results
plt.plot(x, predicciones, label='Predicción')
plt.legend()
plt.show()


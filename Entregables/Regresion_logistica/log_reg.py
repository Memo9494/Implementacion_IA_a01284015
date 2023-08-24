import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Función para el cambio en theta0
def dtheta0(X,y,theta0,thetak,hyp):
    return np.sum(hyp-y)/len(X)

#Función para el cambio en thetak
def dthetak(X,y,theta0,thetak,hyp):
    return np.sum((hyp-y)*X)/len(X)

#Función de hipótesis
def h(x,theta0,thetak):
    x_return = []
    for i in range(len(x)):
        dot = 0
        for j in range(len(x[i])):
            dot += x[i][j]*thetak[j]
        x_return.append(dot)
    # print(x_return[0])
    #sum int to array
    x_return = np.array(x_return)
    return np.array(1/(1+np.exp(-(theta0+x_return))))


    # for i in range(len(thetak)):
    #     print(x[:,i])
    # # x_dot = []
    # for i in range(len(x)):
    #     x_dot.append(np.dot(x[i],thetak))
    # return 1/(1+np.exp(-(theta0+x_dot)))

              
#Leer el archivo csv
df = pd.read_csv('/Users/guillermocepeda/C:C++/Implementacion_IA_a01284015/Entregables/Regresion_logistica/fake_bills.csv',sep=';')
print(df.head())
#Separar los datos en entrenamiento y prueba así como limpiar un poco los datos
#drop nan values
df = df.dropna()

shuffle = df.sample(frac=1)
total_rows = df.shape[0]
twenty_percent = int(total_rows*0.2)
for i in df['is_genuine']:
    if i == "True":
        i = 1
    else:
        i = 0
df20 = shuffle.iloc[:twenty_percent,:]
df80 = shuffle.iloc[twenty_percent:,:]

#Convertir los datos a numpy arrays
X_train = np.array(df80[['diagonal','height_left','height_right','margin_low','margin_up','length']])
y_train = np.array(df80['is_genuine'])
X_test = np.array(df20[['diagonal','height_left','height_right','margin_low','margin_up','length']])
y_test = np.array(df20['is_genuine'])


#Inicializar los thetas y deltas de manera aleatoria para las variables
theta0 = 0.1
thetak = [0.2,0.3,0.4,0.5,0.6,0.1]
delta0 = 0
deltak = [0,0,0,0,0,0]
alpha = 0.001
iterations = 50000
n = len(X_train)


#Realizamos el descenso de gradiente para encontrar los thetas

for i in range(iterations):
    # print(theta0,thetak)
    hyp = h(X_train,theta0,thetak)
    delta0 = dtheta0(X_train,y_train,theta0,thetak,hyp)
    for j in range(len(thetak)):
        deltak[j] = dthetak(X_train[:,0],y_train,theta0,thetak[j],hyp)
    theta0 = theta0 - alpha*delta0
    for j in range(len(thetak)):
        thetak[j] = thetak[j] - alpha*deltak[j]


#Aquí tengo un error, mis thetas se indeterminan, me ayudaría mucho sí encuentra un error en mi código/lógica y le agradezo mucho de antemano
#predicciones con mi hipótesis
hyp = h(X_test,theta0,thetak)
print(hyp)

if hyp[0] > 0.5:
    hyp[0] = 1
else:
    hyp[0] = 0


Falso_positivo = 0
Falso_negativo = 0
Verdadero_positivo = 0
Verdadero_negativo = 0
for i in range(len(hyp)):
    if hyp[i] == 1 and y_test[i] == 1:
        Verdadero_positivo += 1
    elif hyp[i] == 1 and y_test[i] == 0:
        Falso_positivo += 1
    elif hyp[i] == 0 and y_test[i] == 1:
        Falso_negativo += 1
    elif hyp[i] == 0 and y_test[i] == 0:
        Verdadero_negativo += 1

print("Falso positivo: ",Falso_positivo)
print("Falso negativo: ",Falso_negativo)
print("Verdadero positivo: ",Verdadero_positivo)
print("Verdadero negativo: ",Verdadero_negativo)

#Exactitud
print("Exactitud: ",(Verdadero_positivo+Verdadero_negativo)/(Verdadero_positivo+Verdadero_negativo+Falso_positivo+Falso_negativo))
#Precisión
print("Precisión: ",(Verdadero_positivo)/(Verdadero_positivo+Falso_positivo))
#Exhaustividad
print("Exhaustividad: ",(Verdadero_positivo)/(Verdadero_positivo+Falso_negativo))
#F1
print("F1: ",(2*Verdadero_positivo)/(2*Verdadero_positivo+Falso_positivo+Falso_negativo))







# print(theta0,thetak)

from sklearn.neighbors import KNeighborsClassifier  # Clasificador de Vecinos más Cercanos
from sklearn.svm import SVC  # Clasificador de Vectores de Soporte
from sklearn.linear_model import LogisticRegression  # Clasificador de Regresión Logística
from sklearn.tree import DecisionTreeClassifier  # Clasificador de Árbol de Decisiones
from sklearn.naive_bayes import GaussianNB  # Clasificador Bayesiano Ingenuo Gaussiano
from sklearn.ensemble import RandomForestClassifier  # Clasificador de Bosque Aleatorio
from sklearn.ensemble import GradientBoostingClassifier  # Clasificador de Aumento de Gradiente
from sklearn.neural_network import MLPClassifier  # Clasificador de Perceptrón Multicapa

# Descripción y características de cada modelo:
# DecisionTreeClassifier: Construye un árbol de decisión que divide recursivamente el espacio de características en regiones más puras.
#                         Es fácilmente interpretable, pero puede ser propenso al sobreajuste si no se controla adecuadamente.

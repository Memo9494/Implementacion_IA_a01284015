import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Función para el cambio en theta0
def dtheta0(X,y,theta0,thetak):
    xi = np.zeros(len(X))
    for i in range(len(X)):
        xi[i] = np.dot(X[i],thetak)
    return np.sum(h(xi,theta0)-y)/len(X)

#Función para el cambio en thetak
def dthetak(X,y,theta0,thetak):
    X *= thetak
    return np.sum((h(X,theta0)-y)*X)/len(X)

#Función de hipótesis
def h(x,theta0):
    return 1/(1+np.exp(-(theta0+x))) 

              
#Leer el archivo csv
df = pd.read_csv('/Users/guillermocepeda/C:C++/Implementacion_IA_a01284015/Entregables/Regresion_logistica/fake_bills.csv',sep=';')
print(df.head())
#Separar los datos en entrenamiento y prueba así como limpiar un poco los datos
shuffle = df.sample(frac=1)
df.dropna()
total_rows = df.shape[0]
twenty_percent = int(total_rows*0.2)
for i in df['is_genuine']:
    if i == "True":
        i = 1
    else:
        i = 0
df20 = shuffle.iloc[:twenty_percent,:]
df80 = shuffle.iloc[twenty_percent:,:]

X_train = np.array(df80[['diagonal','height_left','height_right','margin_low','margin_up','length']])
y_train = np.array(df80['is_genuine'])
X_test = np.array(df20[['diagonal','height_left','height_right','margin_low','margin_up','length']])
y_test = np.array(df20['is_genuine'])

theta0 = np.random.rand(1)
thetak = np.random.rand(6)
delta0 = np.random.rand(1)
deltak = np.random.rand(6)
alpha = 0.001
iterations = 1000
n = len(X_train)


for i in range(iterations):
    print(theta0,thetak)
    delta0 = dtheta0(X_train,y_train,theta0,thetak)
    for j in range(len(thetak)):
        deltak[j] = dthetak(X_train[:,0],y_train,theta0,thetak[j])
    theta0 = theta0 - alpha*delta0
    for j in range(len(thetak)):
        thetak[j] = thetak[j] - alpha*deltak[j]

print(theta0,thetak)

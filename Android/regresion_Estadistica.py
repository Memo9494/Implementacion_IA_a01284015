import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Importar datos
data = pd.read_csv('/Users/guillermocepeda/C:C++/Implementacion_IA_a01284015/Android/Estatura-peso_HyM.csv')

print(data.head())

#Cmabiar el sexo a 0 y 1
data['Sexo'] = data['Sexo'].map({'Hombre': 0, 'Mujer': 1})

b0 = 1
b1 = 1
b2 = 1
#Variable independiente = estatura + sexo
X1 = data['Estatura']
X2 = data['Sexo']
Y = data['Peso']

#iMplementar regresion lineal
def h(x1,x2,b0,b1,b2):
    return b0 + b1*x1 + b2*x2

def dtheta0(X1,X2,Y,b0,b1,b2,hyp):
    return np.sum(hyp-Y)/len(X1)

def dthetak(X1,X2,Y,b0,b1,b2,hyp):
    return np.sum((hyp-Y)*X1)/len(X1)

def dthetak2(X1,X2,Y,b0,b1,b2,hyp):
    return np.sum((hyp-Y)*X2)/len(X1)

X1 = np.array(X1)
X2 = np.array(X2)
Y = np.array(Y)

iterations = 10000
alpha = 0.001
for i in range(iterations):
    hyp = h(X1,X2,b0,b1,b2)
    b0 = b0 - alpha*dtheta0(X1,X2,Y,b0,b1,b2,hyp)
    b1 = b1 - alpha*dthetak(X1,X2,Y,b0,b1,b2,hyp)
    b2 = b2 - alpha*dthetak2(X1,X2,Y,b0,b1,b2,hyp)

print(b0,b1,b2)

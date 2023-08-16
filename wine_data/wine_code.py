import pandas as pd
import numpy as np

#valor esperado
x = [0.2,0.25,0.2,0.15,0.1,0.05,0.05]
esperanza_x = 0
for i in range(len(x)):
    esperanza_x += x[i]*(i)
print("Valor esperado: ", esperanza_x)

#Se va a cobrar $100x^2 por multa, cual es el valor esperado del pago
def multa(x):
    return 100*x**2

#valor esperado de la multa
esperanza_multa = 0
for i in range(len(x)):
    esperanza_multa += multa(i)*x[i]
print("Valor esperado de la multa: ", esperanza_multa)

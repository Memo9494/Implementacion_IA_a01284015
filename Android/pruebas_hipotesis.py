#Enlatados
#pESOS DE 21 LATAS DE DURAZNOS
pesos = [11,11.6,10.9,12,11.5,12,11.2,10.5,12.2,11.8,12.1,11.6,11.7,11.6,11.2,12,11.4,10.8,11.8,10.9,11.4]

#haremos una prueba de hipótesis de media 11.7 con confianza de 0.98

from scipy import stats as st
import numpy as np
import math
import matplotlib.pyplot as plt

#nuestra hipótesis alternativa es que la media es diferente de 11.7
#nuestra hipótesis nula es que la media es igual a 11.7
H0 = 11.7
confianza = 0.98

a = (1 - confianza)/2
xbarra = np.mean(pesos)
s = np.std(pesos)
sigma = s/math.sqrt(len(pesos))
Zasterisco = (xbarra-H0)/sigma
print("Z_asterisco: ",Zasterisco)

Z0 = st.norm.ppf(a)
print("Z0 : ",Z0)

#Pruena de hipótesis
if Zasterisco < Z0:
    print("Rechazamos la hipótesis nula")
else:
    print("No rechazamos la hipótesis nula")

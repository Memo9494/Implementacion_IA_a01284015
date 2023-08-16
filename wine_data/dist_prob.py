# Instrucciones
# 1. Graficar una distribución Normal con media 
#  = 10, y desviación estándar  
#  = 2
import numpy as np
import matplotlib.pyplot as plt


def normal_dist(x,d,m):
    return ((1/(d*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-m)/d)**2))

#linspace de -100 a 100 con 1000 puntos
x = np.linspace(-100,100,1000)
y = normal_dist(x,2,10)
plt.plot(x,y)
plt.show()


# Sugerencia. Adapte el código siguiente: 

# miu = 0 
# sigma = 1
# x = seq(miu - 4*sigma, miu + 4*sigma, 0.01)
# y = dnorm(x,miu, sigma)
# plot(x,y, type = "l", col = "red", main = "Normal(0,1)")

# 2. Graficar una distribución T Student con grados de libertad  
#   = 12 

# Sugerencia. Adapte el código siguiente: 

# gl = 5  # Grados de libertad
# sigma = sqrt(gl/(gl-2))
# x = seq( -4*sigma, 4*sigma, 0.01)
# y = dt(x,gl)
# plot(x,y, type = "l", col = "blue", main = "T Student con gl = 5")

# 3.  Gráfique la distribución Chi-cuadrada con 8 grados de libertad.

# Sugerencia. Adapte el código siguiente: 

# gl = 10
# sigma = sqrt(2*gl)
# x = seq( 0, miu + 8*sigma, 0.01)
# y = dchisq(x,gl)
# plot(x,y, type = "l", col = "green", main = "Chi2 con gl = 10")

# 4. Graficar una distribución F con v1 = 9, v2 = 13

# Sugerencia. Adapte el código siguiente:

# v1 = 6
# v2 = 10
# sigma = sqrt(2)*v2*sqrt(v2+v1-2)/(sqrt(v2-4)*(v2-2)*sqrt(v1))
# x = seq( 0, miu + 8*sigma, 0.01)
# y = df(x,v1, v2)
# plot(x,y, type = "l", col = "red", main = "F con v1 = 6, v2 = 10")

# 5.  Si Z es una variable aleatoria que se distribuye normalmente con media 0 y desviación estándar 1, hallar los procedimientos de: 

# a) P(Z >0.7) = 0.2419637
# b) P(Z < 0.7) = 0.7580363
# c) P(Z = 0.7)  = 0

# Sugerencia. Utilice la función pnorm, por ejemplo P(Z < 2.1) = pnorm(2.1)

# 6.  Cuando lo que se quiere es hallar el valor de Z dada el área a la izquierda bajo la curva se usa qnorm(área izq). Hallar el valor de Z que tiene al 45% de los demás valores inferiores a ese valor. 

 

# 7.  Hallar el procedimiento para verificar los siguientes resultados si se sabe que X se distribuye normalmente con una media de 100 y desviación estándar de 7.
# P(X < 87) = 0.031645
# P(X > 87) = 0.968354
# P(87 < X < 110) = 0.89179

# Sugerencia. Utilice la función pnorm(x, miu, sigma) de R. 

# 8.  Hallar el procedimiento para verificar los siguientes resultados si se sabe que X se distribuye T Student con gl= 10, hallar: 
# P(X <0.5) = 0.6860532
# P(X > 1.5)  = 0.082253
# La t que sólo el 5% son inferiores a ella.  (t = -1.812461)

# Sugerencia. Utilice   pt(x, gl)    y qt(área izq, gl)  



# 9. Hallar el procedimiento para verificar los siguientes resultados si se sabe que X se distribuye Chi-cuadrada con gl = 6, hallar
# P(X2 < 3) = 0.1911532
# P(X2 > 2) = 0.9196986
# El valor x de chi que sólo el 5% de los demás valores de x es mayor a ese valor ( Resp. 12.59159) 

# Sugerencia. Utilice pchisq(x, gl) y qchisq(área izq., gl) 

 

# 10. Hallar el procedimiento para verificar los siguientes resultados si se sabe que X se distribuye F con v1 = 8, v2 = 10, hallar
# P(X < 2) = 0.8492264
# P(X > 3) = 0.05351256
# El valor de x que sólo el 25% de los demás valores es inferior a él. (Resp. 0.6131229)

 

# 11. Resolver el siguiente problema: Una compañía de reparación de fotocopiadoras encuentra, revisando sus expedientes, que el tiempo invertido en realizar un servicio, se comporta como una variable normal con media de 65 minutos y desviación estándar de 20 minutos. Calculal la proporción de servicios que se hacen en menos de 60 minutos. Resultado en porcentaje con dos decimales, ejemplo 91.32%.

# [R. 40.12%]

# Sugerencia. Use la función de R pnorm(x, miu, sigma)
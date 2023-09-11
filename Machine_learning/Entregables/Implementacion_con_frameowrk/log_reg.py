
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


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

# Leer el archivo csv
df = pd.read_csv('fake_bills.csv', sep=';')
# print(df.head())

# Convertir las etiquetas a valores numéricos
df['is_genuine'] = df['is_genuine'].replace(['False', 'True'], [0, 1])

# Eliminar filas con valores NaN
df = df.dropna()

# Dividir los datos en entrenamiento y prueba x sólo las variables que se consideran importantes length, margin_low y margin_up
X = df[['length', 'margin_low', 'margin_up']]

# X = df[['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']]
y = df['is_genuine']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo Gradient Boosting
GB = GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000)
GB.fit(X_train, y_train)
y_pred = GB.predict(X_test)

# Calcular la matriz de confusión
#Imprimimos en la terminal los resultados de GB para ver que tan relevantes son sus variables y sus resultados
cm = confusion_matrix(y_test, y_pred)
score = GB.score(X_test, y_test)
print(score, accuracy_score(y_test, y_pred), "score")
print(cm)
#Gradient Boosting tiene un score de 1.0, lo cual es muy bueno, pero esto puede ser debido a que el modelo está sobreajustado

# Visualizar la importancia de las características
feat_imp = pd.Series(GB.feature_importances_, ['margin_low', 'margin_up', 'length']).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Importancia de las variables')
plt.ylabel('Importancia de las variables')
plt.show()
#Este plot nos indica que las únicas variables de importancia son lenght, margin_low y margin_up


from sklearn.preprocessing import MinMaxScaler
#Utilizamos el método de MinMaxScaler para normalizar los datos, este se basa en la siguiente ecuación: x_normalized = (x - x_min) / (x_max - x_min)

# Inicializa el escalador
scaler = MinMaxScaler()

# Transformar los datos de entrenamiento y prueba
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Pasar los datos de X_train_scaled y X_test_scaled a un dataframe para poder utilizarlos en la función de hipótesis 
#Ademas de solo utilizar las variables signnificntes
X_train_scaled = pd.DataFrame(X_train_scaled)
X_test_scaled = pd.DataFrame(X_test_scaled)
X_train_scaled.columns = ['length', 'margin_low', 'margin_up']
X_test_scaled.columns = ['length', 'margin_low', 'margin_up']

print(X_train_scaled.head())

#Pasar los datos de entrenamiento y testeo a un array para poder utilizarlos en la función de hipótesis
X_train = np.array(X_train_scaled)
y_train = np.array(y_train)
X_test = np.array(X_test_scaled)
y_test = np.array(y_test)


#Implementacion de la regresión logística con el modelo de sklearn

from sklearn.linear_model import LogisticRegression
#Se crea la instancia del modelo
Modelo1 = LogisticRegression()
#Se entrena el modelo

Modelo1.fit(X_train, y_train)

#Se realiza la prediccion
y_pred = Modelo1.predict(X_test)
print(y_pred)

#Se crea una matriz de confusión para ver los resultados del modelo
confusion_matrix_sk = confusion_matrix(y_test, y_pred)
plt.clf()
plt.imshow(confusion_matrix_sk, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
plt.title('Confusion Matrix - Test Data - Logistic Regression')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(confusion_matrix_sk[i][j]))
plt.show()
print("Score prueba: ",Modelo1.score(X_test, y_test))
print("Exactitud prueba: ",accuracy_score(y_test, y_pred))
#Para conjunto de entrenamiento
y_pred = Modelo1.predict(X_train)
print("score entrenamiento: ",Modelo1.score(X_train, y_train))
print("Exactitud entrenamiento: ",accuracy_score(y_train, y_pred))


#Se puede apreciar que el modelo es excelente, ya que tiene un score de 1.0 y una exactitud de 1.0

#Hagamos cambios en los metaparametros para ver si el modelo sigue siendo bueno con cambios de iteraciones
modelo10 = LogisticRegression(max_iter=5,solver='lbfgs')
#Se entrena el modelo
modelo10.fit(X_train, y_train)
#Se realiza la prediccion
y_pred = modelo10.predict(X_test)
#Score
print("Score prueba: ",modelo10.score(X_test, y_test))
print("Exactitud prueba: ",accuracy_score(y_test, y_pred))
#Para conjunto de entrenamiento
y_pred = modelo10.predict(X_train)
print("score entrenamiento: ",modelo10.score(X_train, y_train))
print("Exactitud entrenamiento: ",accuracy_score(y_train, y_pred))

modelo50 = LogisticRegression(max_iter=50)
#Se entrena el modelo
modelo50.fit(X_train, y_train)
#Se realiza la prediccion
y_pred = modelo50.predict(X_test)
#Score
print("Score prueba: ",modelo50.score(X_test, y_test))
print("Exactitud prueba: ",accuracy_score(y_test, y_pred))
#Para conjunto de entrenamiento
y_pred = modelo10.predict(X_train)
print("score entrenamiento: ",modelo50.score(X_train, y_train))
print("Exactitud entrenamiento: ",accuracy_score(y_train, y_pred))


modelo100 = LogisticRegression(max_iter=100)
#Se entrena el modelo
modelo100.fit(X_train, y_train)
#Se realiza la prediccion
y_pred = modelo100.predict(X_test)
#Score
print("Score prueba: ",modelo100.score(X_test, y_test))
print("Exactitud prueba: ",accuracy_score(y_test, y_pred))
#Para conjunto de entrenamiento
y_pred = modelo100.predict(X_train)
print("score entrenamiento: ",modelo100.score(X_train, y_train))
print("Exactitud entrenamiento: ",accuracy_score(y_train, y_pred))

#Los resultados dan una exactitud y score muy parecidos sin importar el numero de iteraciones, por lo que podemos concluir que los datos son linealmente separables y facilmente clasificables, asi como que no hay mucho ruido significativo

#Ahora hagamos cambios en el solver con el mismo numero de 10 iteraciones

modelo10 = LogisticRegression(max_iter=10,solver='liblinear')
#Se entrena el modelo
modelo10.fit(X_train, y_train)
#Se realiza la prediccion
y_pred = modelo10.predict(X_test)
#Score
print("Score prueba: ",modelo10.score(X_test, y_test))
print("Exactitud prueba: ",accuracy_score(y_test, y_pred))
#Para conjunto de entrenamiento
y_pred = modelo10.predict(X_train)
print("score entrenamiento: ",modelo10.score(X_train, y_train))
print("Exactitud entrenamiento: ",accuracy_score(y_train, y_pred))

modelo10 = LogisticRegression(max_iter=10,solver='sag')
#Se entrena el modelo
modelo10.fit(X_train, y_train)
#Se realiza la prediccion
y_pred = modelo10.predict(X_test)
#Score
print("Score prueba: ",modelo10.score(X_test, y_test))
print("Exactitud prueba: ",accuracy_score(y_test, y_pred))
#Para conjunto de entrenamiento
y_pred = modelo10.predict(X_train)
print("score entrenamiento: ",modelo10.score(X_train, y_train))
print("Exactitud entrenamiento: ",accuracy_score(y_train, y_pred))

modelo10 = LogisticRegression(max_iter=10,solver='saga')
#Se entrena el modelo
modelo10.fit(X_train, y_train)
#Se realiza la prediccion
y_pred = modelo10.predict(X_test)
#Score
print("Score prueba: ",modelo10.score(X_test, y_test))
print("Exactitud prueba: ",accuracy_score(y_test, y_pred))
#Para conjunto de entrenamiento
y_pred = modelo10.predict(X_train)
print("score entrenamiento: ",modelo10.score(X_train, y_train))
print("Exactitud entrenamiento: ",accuracy_score(y_train, y_pred))

#Parece que cambiar el solver no afecta mucho al modelo, ya que los resultados son muy parecidos

#Ahora hagamos cambios del learning rate

modelo10 = LogisticRegression(max_iter=10,solver='lbfgs',C=0.1)
#Se entrena el modelo
modelo10.fit(X_train, y_train)
#Se realiza la prediccion
y_pred = modelo10.predict(X_test)
#Score
print("Score prueba C=0.1: ",modelo10.score(X_test, y_test))
print("Exactitud prueba C=0.1: ",accuracy_score(y_test, y_pred))
#Para conjunto de entrenamiento
y_pred = modelo10.predict(X_train)
print("score entrenamiento C=0.1: ",modelo10.score(X_train, y_train))
print("Exactitud entrenamiento C=0.1: ",accuracy_score(y_train, y_pred))

modelo10 = LogisticRegression(max_iter=10,solver='lbfgs',C=0.01)
#Se entrena el modelo
modelo10.fit(X_train, y_train)
#Se realiza la prediccion
y_pred = modelo10.predict(X_test)
#Score
print("Score prueba C=0.01: ",modelo10.score(X_test, y_test))
print("Exactitud prueba C=0.01: ",accuracy_score(y_test, y_pred))
#Para conjunto de entrenamiento
y_pred = modelo10.predict(X_train)
print("score entrenamiento C=0.01: ",modelo10.score(X_train, y_train))
print("Exactitud entrenamiento C=0.01: ",accuracy_score(y_train, y_pred))

modelo10 = LogisticRegression(max_iter=10,solver='lbfgs',C=0.001)
#Se entrena el modelo
modelo10.fit(X_train, y_train)
#Se realiza la prediccion
y_pred = modelo10.predict(X_test)
#Score
print("Score prueba C=0.001: ",modelo10.score(X_test, y_test))
print("Exactitud prueba C=0.001: ",accuracy_score(y_test, y_pred))
#Para conjunto de entrenamiento
y_pred = modelo10.predict(X_train)
print("score entrenamiento C=0.001: ",modelo10.score(X_train, y_train))
print("Exactitud entrenamiento C=0.001: ",accuracy_score(y_train, y_pred))


#Graficar rendimiento del modelo con diferentes valores de C
#Se crea una lista con los valores de C
C = [0.1,0.01,0.001]
#Se crea una lista vacia para guardar los scores
scores = []
#Se crea un ciclo para entrenar el modelo con los diferentes valores de C
for i in C:
    modelo10 = LogisticRegression(max_iter=10,solver='lbfgs',C=i)
    #Se entrena el modelo
    modelo10.fit(X_train, y_train)
    #Se realiza la prediccion
    y_pred = modelo10.predict(X_test)
    #Se guarda el score en la lista
    scores.append(modelo10.score(X_test, y_test))

#Se grafica el rendimiento del modelo con diferentes 
plt.plot(C,scores)
plt.xlabel('C')
plt.ylabel('Score')
plt.title('Rendimiento del modelo con diferentes valores de C')
plt.show()


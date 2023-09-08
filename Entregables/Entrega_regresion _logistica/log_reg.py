
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
df = pd.read_csv('/Users/guillermocepeda/C:C++/Implementacion_IA_a01284015/Entregables/Entrega_regresion _logistica/fake_bills.csv', sep=';')
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
feat_imp = pd.Series(GB.feature_importances_, ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']).sort_values(ascending=False)
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
X_train_scaled = pd.DataFrame(X_train_scaled)
X_test_scaled = pd.DataFrame(X_test_scaled)
X_train_scaled.columns = ['length', 'margin_low', 'margin_up']
X_test_scaled.columns = ['length', 'margin_low', 'margin_up']

print(X_train_scaled.head())






#Inicializar los thetas y deltas de manera aleatoria para las variables
theta0 = 0.1
thetak = [0.2,0.3,0.4]
delta0 = 0
deltak = [0,0,0]
alpha = 0.01
iterations = 10000
n = len(X_train)

X_train = np.array(X_train_scaled)
y_train = np.array(y_train)
X_test = np.array(X_test_scaled)
y_test = np.array(y_test)

#Realizamos el descenso de gradiente para encontrar los thetas
for i in range(iterations):
    #Se crea la hipótesis
    hyp = h(X_train,theta0,thetak)
    #se actualiza el cambio en theta0 y thetak
    delta0 = dtheta0(X_train,y_train,theta0,thetak,hyp)
    for j in range(len(thetak)):
        deltak[j] = dthetak(X_train[:,j],y_train,theta0,thetak[j],hyp)
    #se actualizan los theta0 y thetak
    theta0 = theta0 - alpha*delta0
    for j in range(len(thetak)):
        thetak[j] = thetak[j] - alpha*deltak[j]


#predicciones con mi hipótesis
hyp = h(X_test,theta0,thetak)
predicted_labels = (hyp >= 0.5).astype(int)

print("Theta0: ",theta0)
print("Thetak: ",thetak)
print("Hyp: ",hyp)
print("Predicted labels: ",predicted_labels)
print("Real labels: ",y_test)


print(hyp)

for i in range(len(hyp)):
    if hyp[i] >= 0.5:
        hyp[i] = 1
    else:
        hyp[i] = 0


# Se corren algunas predicciones para validar la salida del modelo, usando datos diferentes a los de entrenamiento

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

#Se corren algunas predicciones para validar la salida del modelo, usando datos diferentes a los de entrenamiento


#Hasta aquí se realizó el algoritmo con éxito sin uso de un framework, pero se decidió utilizar el framework de sklearn para comparar resultados
from sklearn.linear_model import LogisticRegression
LogisticRegression = LogisticRegression()
LogisticRegression.fit(X_train, y_train)
y_pred = LogisticRegression.predict(X_test)
print(y_pred)
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
print("Score: ",LogisticRegression.score(X_test, y_test))
print("Exactitud: ",accuracy_score(y_test, y_pred))


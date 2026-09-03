"""
KNN (K-Vecinos Mas Cercanos) implementado desde cero, sin framework.
Clasifica el tipo de garantia de un credito automotriz.

KNN no "entrena": memoriza los datos. Para predecir mide la distancia del
punto nuevo a todos los de entrenamiento, toma los k mas cercanos y estos
votan; gana la clase con mas votos.

Ejecutar: python3 main.py     (solo requiere matplotlib para las graficas)
"""

import csv
import math
import os
import random
import matplotlib.pyplot as plt


# -----------------------------
# CONFIGURACION
# -----------------------------

# Ruta relativa a este archivo, para poder correrlo desde cualquier carpeta.
RUTA =  "interest_ledger.csv"
K = 5


# -----------------------------
# CARGAR DATOS
# -----------------------------

def cargar_datos():
    X = []
    y = []

    with open(RUTA, "r") as archivo:
        for fila in csv.DictReader(archivo):
            # El CSV trae espacios raros (\xa0) pegados a la clase.
            clase = fila["collateral"].replace("\xa0", " ").strip().upper()
            if clase == "":
                continue  # sin etiqueta no sirve para entrenar ni para medir

            principal = float(fila["principal_amount"] or 0)
            interes = float(fila["interest_amount"] or 0)
            year_model = float(fila["year_model"] or 0)
            fecha = fila["month_date"] or ""

            if len(fecha) >= 7:
                year = int(fecha[:4])
                mes = int(fecha[5:7])
            else:
                year = 0
                mes = 0

            # Tasa mensual: cuanto interes genera cada peso prestado.
            if principal > 0:
                tasa = interes / principal
            else:
                tasa = 0

            if year > 0 and year_model > 0:
                antiguedad = year - year_model
            else:
                antiguedad = 0

            X.append([principal, interes, tasa, antiguedad, mes])
            y.append(clase)

    return X, y


# -----------------------------
# DIVIDIR TRAIN / TEST
# -----------------------------

def dividir_datos(X, y):
    indices = list(range(len(X)))

    random.seed(42)  # semilla fija para que el resultado sea reproducible
    random.shuffle(indices)

    corte = int(len(indices) * 0.75)

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for i in indices[:corte]:
        X_train.append(X[i])
        y_train.append(y[i])

    for i in indices[corte:]:
        X_test.append(X[i])
        y_test.append(y[i])

    return X_train, y_train, X_test, y_test


# -----------------------------
# NORMALIZAR
# -----------------------------
# Para que el valor del vehiculo no domine la distancia sobre el resto de las variables, se normalizan 
# solo se calcula en el train

def calcular_media_desviacion(X):
    medias = []
    desviaciones = []

    for columna in range(len(X[0])):
        valores = []
        for fila in X:
            valores.append(fila[columna])

        media = sum(valores) / len(valores)

        suma = 0
        for valor in valores:
            suma += (valor - media) ** 2
        desviacion = math.sqrt(suma / len(valores))

        if desviacion == 0:
            desviacion = 1  # evita dividir entre cero si la columna es constante

        medias.append(media)
        desviaciones.append(desviacion)

    return medias, desviaciones


def normalizar(X, medias, desviaciones):
    resultado = []

    for fila in X:
        nueva_fila = []
        for i in range(len(fila)):
            nueva_fila.append((fila[i] - medias[i]) / desviaciones[i])
        resultado.append(nueva_fila)

    return resultado


# -----------------------------
# KNN
# -----------------------------

def distancia_euclidiana(a, b):
    # Raiz de la suma de las diferencias al cuadrado.
    suma = 0
    for i in range(len(a)):
        suma += (a[i] - b[i]) ** 2
    return math.sqrt(suma)


def predecir_knn(X_train, y_train, punto, k):
    distancias = []

    # Distancia del punto nuevo a cada punto de entrenamiento.
    for i in range(len(X_train)):
        distancias.append((distancia_euclidiana(X_train[i], punto), y_train[i]))

    distancias.sort()
    vecinos = distancias[:k]  # los k mas cercanos

    # Cada vecino aporta un voto a su clase.
    votos = {}
    for clase in vecinos:
        if clase in votos:
            votos[clase] += 1
        else:
            votos[clase] = 1

    mejor_clase = None
    mayor_votos = 0
    for clase in votos:
        if votos[clase] > mayor_votos:
            mayor_votos = votos[clase]
            mejor_clase = clase

    return mejor_clase


def predecir_todos(X_train, y_train, X_test, k):
    predicciones = []
    for punto in X_test:
        predicciones.append(predecir_knn(X_train, y_train, punto, k))
    return predicciones


# -----------------------------
# METRICAS
# -----------------------------

def accuracy(y_real, y_pred):
    correctos = 0
    for i in range(len(y_real)):
        if y_real[i] == y_pred[i]:
            correctos += 1
    return correctos / len(y_real)


def matriz_confusion(y_real, y_pred):
    # matriz[i][j] = casos de clase real i que se predijeron como j.
    # La diagonal son los aciertos, lo de fuera son los errores.
    clases = sorted(list(set(y_real)))

    matriz = []
    for i in range(len(clases)):
        matriz.append([0] * len(clases))

    for i in range(len(y_real)):
        fila = clases.index(y_real[i])
        columna = clases.index(y_pred[i])
        matriz[fila][columna] += 1

    return clases, matriz


def calcular_metricas(clases, matriz):
    # Las clases estan desbalanceadas, asi que el accuracy solo no basta:
    # se saca precision, recall y F1 de cada clase por separado.
    precisiones = []
    recalls = []
    f1_scores = []

    for i in range(len(clases)):
        TP = matriz[i][i]              # aciertos de esta clase
        FP = 0                         
        FN = 0                         
        for fila in range(len(clases)):
            if fila != i:
                FP += matriz[fila][i]

        for columna in range(len(clases)):
            if columna != i:
                FN += matriz[i][columna]

        if TP + FP > 0:
            precision = TP / (TP + FP)  # de lo que predije, cuanto acerte
        else:
            precision = 0

        if TP + FN > 0:
            recall = TP / (TP + FN)     # de lo que habia, cuanto encontre
        else:
            recall = 0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)  # balance de ambas
        else:
            f1 = 0

        precisiones.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        print("\nClase:", clases[i])
        print("Precision:", round(precision, 3))
        print("Recall:", round(recall, 3))
        print("F1:", round(f1, 3))

    return precisiones, recalls, f1_scores


# -----------------------------
# GRAFICAS
# -----------------------------

def graficar_matriz(clases, matriz):
    plt.figure(figsize=(7, 5))
    plt.imshow(matriz)
    plt.title("Matriz de Confusion")
    plt.xlabel("Prediccion")
    plt.ylabel("Clase real")
    plt.xticks(range(len(clases)), clases, rotation=45)
    plt.yticks(range(len(clases)), clases)

    for i in range(len(clases)):
        for j in range(len(clases)):
            plt.text(j, i, matriz[i][j], ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.show()


def graficar_metricas(clases, precision, recall, f1):
    posiciones = list(range(len(clases)))

    plt.figure(figsize=(8, 5))
    plt.plot(posiciones, precision, marker="o", label="Precision")
    plt.plot(posiciones, recall, marker="o", label="Recall")
    plt.plot(posiciones, f1, marker="o", label="F1")
    plt.xticks(posiciones, clases, rotation=45)
    plt.ylim(0, 1)
    plt.title("Metricas por clase")
    plt.ylabel("Valor")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# PRINCIPAL
# -----------------------------

X, y = cargar_datos()
X_train, y_train, X_test, y_test = dividir_datos(X, y)

# El escalador se ajusta con train y esos mismos valores se aplican a test.
medias, desviaciones = calcular_media_desviacion(X_train)
X_train = normalizar(X_train, medias, desviaciones)
X_test = normalizar(X_test, medias, desviaciones)

predicciones = predecir_todos(X_train, y_train, X_test, K)

print("Total de datos:", len(X))
print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))
print("\nAccuracy:", round(accuracy(y_test, predicciones) * 100, 2), "%")

clases, matriz = matriz_confusion(y_test, predicciones)
precision, recall, f1 = calcular_metricas(clases, matriz)

graficar_matriz(clases, matriz)
graficar_metricas(clases, precision, recall, f1)

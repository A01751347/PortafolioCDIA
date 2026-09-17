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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# CONFIGURACION
# -----------------------------

# Rutas relativas a este archivo, para poder correrlo desde cualquier carpeta.
CARPETA = os.path.dirname(os.path.abspath(__file__))
RUTA = os.path.join(CARPETA, "interest_ledger.csv")
FIGURAS = os.path.join(CARPETA, "figuras")

K = 5
VALORES_K = [1, 3, 5, 7, 9, 11, 15, 21]
NOMBRES = ["principal", "interes", "tasa", "antiguedad", "mes"]


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
                continue  # sin etiqueta no sirve ni para entrenar ni para medir

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
                tasa = 0.0

            if year > 0 and year_model > 0:
                antiguedad = year - year_model
            else:
                antiguedad = 0.0

            X.append([principal, interes, tasa, antiguedad, float(mes)])
            y.append(clase)

    return X, y


# -----------------------------
# DIVIDIR TRAIN / TEST
# -----------------------------

def dividir_datos(X, y):
    # Se barajan los indices y no las listas para que X y y sigan emparejados.
    indices = list(range(len(X)))

    random.seed(42)  # semilla fija para que el resultado sea reproducible
    random.shuffle(indices)

    corte = int(len(indices) * 0.75)

    X_train = [X[i] for i in indices[:corte]]
    y_train = [y[i] for i in indices[:corte]]
    X_test = [X[i] for i in indices[corte:]]
    y_test = [y[i] for i in indices[corte:]]

    return X_train, y_train, X_test, y_test


# -----------------------------
# NORMALIZAR
# -----------------------------
# El monto del credito vale cientos de miles y la tasa vale milesimas, asi que
# sin normalizar la distancia seria practicamente solo el monto. Se estandariza
# con z = (x - media) / desviacion, calculando media y desviacion SOLO en train
# para no meter informacion del conjunto de prueba.

def calcular_media_desviacion(X):
    medias = []
    desviaciones = []

    for columna in range(len(X[0])):
        valores = [fila[columna] for fila in X]

        media = sum(valores) / len(valores)

        suma = 0.0
        for valor in valores:
            suma += (valor - media) ** 2
        desviacion = math.sqrt(suma / len(valores))

        if desviacion == 0:
            desviacion = 1.0  # evita dividir entre cero si la columna es constante

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
    suma = 0.0
    for i in range(len(a)):
        suma += (a[i] - b[i]) ** 2
    return math.sqrt(suma)


def vecinos_mas_cercanos(X_train, y_train, punto, k):
    distancias = []

    # Distancia del punto nuevo a cada punto de entrenamiento.
    for i in range(len(X_train)):
        distancias.append((distancia_euclidiana(X_train[i], punto), y_train[i]))

    distancias.sort()

    return [clase for _, clase in distancias[:k]]  # los k mas cercanos


def votar(clases_vecinas):
    # Cada vecino aporta un voto a su clase.
    votos = {}
    for clase in clases_vecinas:
        votos[clase] = votos.get(clase, 0) + 1

    # Se recorre en orden de cercania: si hay empate gana el vecino mas cercano.
    mejor_clase = None
    mayor_votos = -1
    for clase in clases_vecinas:
        if votos[clase] > mayor_votos:
            mayor_votos = votos[clase]
            mejor_clase = clase

    return mejor_clase


def predecir_knn(X_train, y_train, punto, k):
    return votar(vecinos_mas_cercanos(X_train, y_train, punto, k))


def vecinos_de_todos(X_train, y_train, X_test, k_maximo):
    # Guardar de una vez los k_maximo vecinos de cada punto permite probar todos
    # los valores de k repitiendo solo la votacion, que es instantanea.
    tabla = []

    for i, punto in enumerate(X_test):
        tabla.append(vecinos_mas_cercanos(X_train, y_train, punto, k_maximo))
        if (i + 1) % 500 == 0:
            print("   ...", i + 1, "de", len(X_test))

    return tabla


# -----------------------------
# METRICAS
# -----------------------------

def accuracy(y_real, y_pred):
    correctos = 0
    for i in range(len(y_real)):
        if y_real[i] == y_pred[i]:
            correctos += 1
    return correctos / len(y_real)


def matriz_confusion(y_real, y_pred, clases):
    # matriz[i][j] = casos de clase real i que se predijeron como j.
    # La diagonal son los aciertos, lo de fuera son los errores.
    indice = {clase: i for i, clase in enumerate(clases)}
    matriz = [[0] * len(clases) for _ in range(len(clases))]

    for i in range(len(y_real)):
        matriz[indice[y_real[i]]][indice[y_pred[i]]] += 1

    return matriz


def calcular_metricas(clases, matriz, imprimir=True):
    # Las clases estan desbalanceadas, asi que el accuracy solo no basta:
    # se saca precision, recall y F1 de cada clase por separado.
    precisiones = []
    recalls = []
    f1_scores = []

    for i in range(len(clases)):
        TP = matriz[i][i]

        FP = 0
        for fila in range(len(clases)):
            if fila != i:
                FP += matriz[fila][i]

        FN = 0
        for columna in range(len(clases)):
            if columna != i:
                FN += matriz[i][columna]

        if TP + FP > 0:
            precision = TP / (TP + FP)  # de lo que predije, cuanto acerte
        else:
            precision = 0.0

        if TP + FN > 0:
            recall = TP / (TP + FN)  # de lo que habia, cuanto encontre
        else:
            recall = 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)  # balance de ambas
        else:
            f1 = 0.0

        precisiones.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        if imprimir:
            print("   {:<12} precision {:.3f}   recall {:.3f}   F1 {:.3f}   (n={})".format(
                clases[i], precision, recall, f1, TP + FN))

    return precisiones, recalls, f1_scores


def f1_macro(f1_scores):
    # Promedio simple: cada clase pesa igual sin importar cuantos casos tenga.
    return sum(f1_scores) / len(f1_scores)


# -----------------------------
# GRAFICAS
# -----------------------------

def guardar(nombre):
    if not os.path.isdir(FIGURAS):
        os.makedirs(FIGURAS)
    plt.savefig(os.path.join(FIGURAS, nombre), dpi=140)
    plt.close()
    print("   figura guardada: figuras/" + nombre)


def graficar_distribucion(y_train, y_test, clases):
    n_train = [sum(1 for v in y_train if v == c) for c in clases]
    n_test = [sum(1 for v in y_test if v == c) for c in clases]
    x = list(range(len(clases)))

    plt.figure(figsize=(8, 5))
    plt.bar([p - 0.19 for p in x], n_train, 0.38, label="Entrenamiento")
    plt.bar([p + 0.19 for p in x], n_test, 0.38, label="Prueba")
    plt.xticks(x, clases, rotation=45, ha="right")
    plt.ylabel("Registros")
    plt.title("Distribucion de clases")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    guardar("distribucion_clases.png")


def graficar_matriz(clases, matriz):
    # Se normaliza por fila para poder comparar clases de tamaños muy distintos.
    normalizada = []
    for fila in matriz:
        total = sum(fila)
        normalizada.append([v / total if total else 0 for v in fila])

    plt.figure(figsize=(7.5, 6))
    plt.imshow(normalizada, cmap="Blues", vmin=0, vmax=1)
    plt.title("Matriz de Confusion (KNN, k={})".format(K))
    plt.xlabel("Prediccion")
    plt.ylabel("Clase real")
    plt.xticks(range(len(clases)), clases, rotation=45, ha="right")
    plt.yticks(range(len(clases)), clases)

    for i in range(len(clases)):
        for j in range(len(clases)):
            color = "white" if normalizada[i][j] > 0.5 else "black"
            plt.text(j, i, "{}\n{:.0%}".format(matriz[i][j], normalizada[i][j]),
                     ha="center", va="center", fontsize=8, color=color)

    plt.colorbar()
    plt.tight_layout()
    guardar("matriz_confusion.png")


def graficar_metricas(clases, precision, recall, f1):
    x = list(range(len(clases)))

    plt.figure(figsize=(8, 5))
    plt.bar([p - 0.27 for p in x], precision, 0.27, label="Precision")
    plt.bar(x, recall, 0.27, label="Recall")
    plt.bar([p + 0.27 for p in x], f1, 0.27, label="F1")
    plt.xticks(x, clases, rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.title("Metricas por clase (k={})".format(K))
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    guardar("metricas_por_clase.png")


def graficar_barrido_k(resultados):
    ks = [r[0] for r in resultados]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, [r[1] for r in resultados], marker="o", label="Accuracy")
    plt.plot(ks, [r[2] for r in resultados], marker="s", label="F1 macro")
    plt.axvline(K, color="gray", linestyle="--", alpha=0.7, label="k elegido")
    plt.xticks(ks)
    plt.xlabel("k (numero de vecinos)")
    plt.ylabel("Valor en el conjunto de prueba")
    plt.title("Efecto del numero de vecinos")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    guardar("barrido_k.png")


# -----------------------------
# PREDICCIONES NUEVAS
# -----------------------------

def predecir_casos_nuevos(X_train, y_train, medias, desviaciones):
    casos = [
        ("Credito grande, auto del año", 350000.0, 1500.0, 2026, 2025, 2),
        ("Credito muy grande, auto nuevo", 1200000.0, 9000.0, 2026, 2026, 5),
        ("Credito chico, auto con 4 años", 90000.0, 400.0, 2025, 2021, 8),
    ]

    print("\n" + "=" * 70)
    print(" PREDICCIONES DE CASOS NUEVOS")
    print("=" * 70)

    for descripcion, principal, interes, year, year_model, mes in casos:
        tasa = interes / principal if principal > 0 else 0.0
        crudo = [principal, interes, tasa, float(year - year_model), float(mes)]

        # El caso nuevo se normaliza con la media y desviacion del entrenamiento.
        punto = normalizar([crudo], medias, desviaciones)[0]
        vecinos = vecinos_mas_cercanos(X_train, y_train, punto, K)

        print("\n", descripcion)
        print("   principal ${:,.2f} | interes ${:,.2f} | antiguedad {} | mes {}".format(
            principal, interes, year - year_model, mes))
        print("   PREDICCION:", votar(vecinos))

        votos = {}
        for clase in vecinos:
            votos[clase] = votos.get(clase, 0) + 1
        print("   votos:", ", ".join("{} {}/{}".format(c, n, K)
                                     for c, n in sorted(votos.items())))


# -----------------------------
# PRINCIPAL
# -----------------------------

X, y = cargar_datos()
X_train, y_train, X_test, y_test = dividir_datos(X, y)
clases = sorted(set(y))

print("Total de datos:", len(X))
print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))

print("\nRegistros por clase:")
for clase in clases:
    n_tr = sum(1 for v in y_train if v == clase)
    n_te = sum(1 for v in y_test if v == clase)
    print("   {:<12} train {:>5}   test {:>5}".format(clase, n_tr, n_te))

# El escalador se ajusta con train y esos mismos valores se aplican a test.
medias, desviaciones = calcular_media_desviacion(X_train)

print("\nMedia y desviacion (calculadas solo con entrenamiento):")
for i in range(len(NOMBRES)):
    print("   {:<12} media {:>14,.4f}   desviacion {:>14,.4f}".format(
        NOMBRES[i], medias[i], desviaciones[i]))

X_train_n = normalizar(X_train, medias, desviaciones)
X_test_n = normalizar(X_test, medias, desviaciones)

print("\nCalculando distancias...")
tabla_vecinos = vecinos_de_todos(X_train_n, y_train, X_test_n, max(VALORES_K))

print("\n" + "=" * 70)
print(" COMPARACION DE VALORES DE k")
print("=" * 70)
print(" {:>4}   {:>10}   {:>10}".format("k", "accuracy", "F1 macro"))

resultados_k = []
for k in VALORES_K:
    predicciones_k = [votar(vecinos[:k]) for vecinos in tabla_vecinos]
    matriz_k = matriz_confusion(y_test, predicciones_k, clases)
    _, _, f1_k = calcular_metricas(clases, matriz_k, imprimir=False)

    acc = accuracy(y_test, predicciones_k)
    resultados_k.append((k, acc, f1_macro(f1_k)))

    marca = "  <-- elegido" if k == K else ""
    print(" {:>4}   {:>9.2f}%   {:>10.3f}{}".format(k, acc * 100, f1_macro(f1_k), marca))

predicciones = [votar(vecinos[:K]) for vecinos in tabla_vecinos]

print("\n" + "=" * 70)
print(" RESULTADOS CON k =", K)
print("=" * 70)
print("\nAccuracy:", round(accuracy(y_test, predicciones) * 100, 2), "%")

matriz = matriz_confusion(y_test, predicciones, clases)

print("\nMatriz de confusion (filas = real, columnas = prediccion):")
print("   {:<12}".format("") + "".join("{:>11}".format(c[:10]) for c in clases))
for i in range(len(clases)):
    print("   {:<12}".format(clases[i]) + "".join("{:>11}".format(v) for v in matriz[i]))

print("\nMetricas por clase:")
precision, recall, f1 = calcular_metricas(clases, matriz)
print("\nF1 macro:", round(f1_macro(f1), 3))

predecir_casos_nuevos(X_train_n, y_train, medias, desviaciones)

print("\n" + "=" * 70)
print(" GRAFICAS")
print("=" * 70)
graficar_distribucion(y_train, y_test, clases)
graficar_matriz(clases, matriz)
graficar_metricas(clases, precision, recall, f1)
graficar_barrido_k(resultados_k)

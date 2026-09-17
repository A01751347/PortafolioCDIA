"""
================================================================================
 KNN (K-Vecinos Mas Cercanos) IMPLEMENTADO DESDE CERO, SIN FRAMEWORK
================================================================================

 Modulo 2 - Portafolio de Implementacion
 Santiago Serrano Montalvo - A01751347

 PROBLEMA
 --------
 Clasificar el tipo de garantia (`collateral`) de un credito automotriz a
 partir de los datos financieros del pago mensual.

 COMO FUNCIONA KNN
 -----------------
 KNN es un algoritmo "perezoso": no ajusta parametros durante el
 entrenamiento, simplemente memoriza los datos. Todo el trabajo ocurre al
 momento de predecir:

   1. Se recibe un punto nuevo x.
   2. Se mide la distancia euclidiana de x a CADA punto de entrenamiento.
   3. Se ordenan esas distancias de menor a mayor.
   4. Se toman los k puntos mas cercanos (los "vecinos").
   5. Cada vecino emite un voto por su propia clase.
   6. La clase con mas votos es la prediccion.

 La distancia euclidiana entre dos vectores a y b de n dimensiones es:

       d(a, b) = sqrt( (a1-b1)^2 + (a2-b2)^2 + ... + (an-bn)^2 )

 RESTRICCION DE LA ACTIVIDAD
 ---------------------------
 No se usa ninguna biblioteca de aprendizaje maquina ni de estadistica
 avanzada. Solo la libreria estandar de Python (csv, math, os, random) y
 matplotlib, que se usa UNICAMENTE para dibujar las graficas de resultados.
 La media, la desviacion estandar, la distancia, la votacion, la matriz de
 confusion, el accuracy, la precision, el recall y el F1 estan programados
 a mano en este archivo.

 EJECUCION
 ---------
       python3 main.py

================================================================================
"""

import csv
import math
import os
import random

import matplotlib
matplotlib.use("Agg")          # backend sin ventana: permite correr en consola pura
import matplotlib.pyplot as plt


# ==============================================================================
# CONFIGURACION
# ==============================================================================

# Las rutas se construyen a partir de la ubicacion de este archivo para que el
# programa funcione sin importar desde que carpeta se invoque.
CARPETA = os.path.dirname(os.path.abspath(__file__))
RUTA_CSV = os.path.join(CARPETA, "interest_ledger.csv")
CARPETA_FIGURAS = os.path.join(CARPETA, "figuras")

K = 5                          # numero de vecinos que votan
PORCENTAJE_ENTRENAMIENTO = 0.75
SEMILLA = 42                   # semilla fija -> resultados reproducibles

# Valores de k que se comparan en el experimento de la seccion 6 del reporte.
VALORES_K = [1, 3, 5, 7, 9, 11, 15, 21]

NOMBRES_VARIABLES = ["principal", "interes", "tasa", "antiguedad", "mes"]


# ==============================================================================
# 1. CARGA Y PREPARACION DE LOS DATOS
# ==============================================================================

def cargar_datos(ruta=RUTA_CSV):
    """Lee el CSV y construye la matriz de variables X y el vector de clases y.

    A partir de las columnas crudas se derivan dos variables nuevas:
      - tasa:       interes / principal  (que tan caro sale cada peso prestado)
      - antiguedad: year del pago - year_model  (años del vehiculo al pagar)

    Devuelve:
        X -> lista de listas, cada fila son las 5 variables de un registro
        y -> lista de cadenas, la clase real de cada registro
    """
    X = []
    y = []

    with open(ruta, "r") as archivo:
        for fila in csv.DictReader(archivo):
            # El CSV trae espacios duros (\xa0) pegados al nombre de la clase.
            clase = fila["collateral"].replace("\xa0", " ").strip().upper()
            if clase == "":
                continue                      # sin etiqueta no sirve ni para entrenar ni para medir

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

            # Tasa mensual efectiva. Se protege la division entre cero.
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


def dividir_datos(X, y, porcentaje=PORCENTAJE_ENTRENAMIENTO, semilla=SEMILLA):
    """Baraja los indices con una semilla fija y corta en entrenamiento / prueba.

    Se barajan los INDICES y no las listas para que X y y sigan emparejados.
    La semilla fija garantiza que la misma particion se reproduzca siempre.
    """
    indices = list(range(len(X)))

    random.seed(semilla)
    random.shuffle(indices)

    corte = int(len(indices) * porcentaje)

    X_train = [X[i] for i in indices[:corte]]
    y_train = [y[i] for i in indices[:corte]]
    X_test = [X[i] for i in indices[corte:]]
    y_test = [y[i] for i in indices[corte:]]

    return X_train, y_train, X_test, y_test


# ==============================================================================
# 2. NORMALIZACION (estandarizacion z-score, calculada a mano)
# ==============================================================================
#
# KNN se basa en distancias, asi que una variable con numeros grandes domina
# el calculo. Aqui `principal` vale cientos de miles y `tasa` vale ~0.005: sin
# normalizar, la distancia seria practicamente solo el monto del credito.
#
# Se estandariza cada columna con  z = (x - media) / desviacion.
#
# IMPORTANTE: la media y la desviacion se calculan SOLO con entrenamiento y
# despues se aplican tal cual a prueba. Si se calcularan con todo el dataset
# habria fuga de informacion (data leakage) del conjunto de prueba.
# ==============================================================================

def calcular_media_desviacion(X):
    """Devuelve la media y la desviacion estandar poblacional de cada columna."""
    medias = []
    desviaciones = []

    for columna in range(len(X[0])):
        valores = [fila[columna] for fila in X]

        media = sum(valores) / len(valores)

        # Varianza = promedio de las diferencias al cuadrado respecto a la media.
        suma = 0.0
        for valor in valores:
            suma += (valor - media) ** 2
        desviacion = math.sqrt(suma / len(valores))

        if desviacion == 0:
            desviacion = 1.0       # columna constante: evita dividir entre cero

        medias.append(media)
        desviaciones.append(desviacion)

    return medias, desviaciones


def normalizar(X, medias, desviaciones):
    """Aplica z = (x - media) / desviacion a cada valor de cada fila."""
    resultado = []

    for fila in X:
        nueva_fila = []
        for i in range(len(fila)):
            nueva_fila.append((fila[i] - medias[i]) / desviaciones[i])
        resultado.append(nueva_fila)

    return resultado


# ==============================================================================
# 3. EL ALGORITMO KNN
# ==============================================================================

def distancia_euclidiana(a, b):
    """Raiz cuadrada de la suma de las diferencias al cuadrado."""
    suma = 0.0
    for i in range(len(a)):
        diferencia = a[i] - b[i]
        suma += diferencia * diferencia
    return math.sqrt(suma)


def vecinos_mas_cercanos(X_train, y_train, punto, k):
    """Devuelve la lista de las clases de los k puntos mas cercanos a `punto`.

    Se arma una lista de pares (distancia, clase), se ordena de menor a mayor
    distancia y se recortan los primeros k elementos.
    """
    distancias = []
    for i in range(len(X_train)):
        distancias.append((distancia_euclidiana(X_train[i], punto), y_train[i]))

    distancias.sort()                       # ordena por distancia (primer elemento del par)

    return [clase for _, clase in distancias[:k]]


def votar(clases_vecinas):
    """Cuenta los votos y devuelve la clase ganadora.

    Cada vecino aporta exactamente un voto a SU clase. En caso de empate gana
    la clase que aparece antes en la lista, es decir la del vecino mas cercano,
    que es el criterio de desempate habitual en KNN.
    """
    votos = {}
    for clase in clases_vecinas:
        if clase in votos:
            votos[clase] += 1
        else:
            votos[clase] = 1

    mejor_clase = None
    mayor_votos = -1
    for clase in clases_vecinas:             # se recorre en orden de cercania
        if votos[clase] > mayor_votos:
            mayor_votos = votos[clase]
            mejor_clase = clase

    return mejor_clase


def predecir_knn(X_train, y_train, punto, k):
    """Prediccion de un solo punto: busca los k vecinos y los hace votar."""
    return votar(vecinos_mas_cercanos(X_train, y_train, punto, k))


def predecir_todos(X_train, y_train, X_test, k):
    """Prediccion de todo un conjunto de puntos."""
    return [predecir_knn(X_train, y_train, punto, k) for punto in X_test]


def vecinos_de_todos(X_train, y_train, X_test, k_maximo):
    """Precalcula, para cada punto de prueba, las clases de sus k_maximo vecinos.

    El costo real de KNN esta en calcular y ordenar las distancias. Guardando
    de una sola vez la lista ordenada de los k_maximo vecinos se puede evaluar
    cualquier k <= k_maximo con solo repetir la votacion, que es instantanea.
    Asi la comparacion de 8 valores de k cuesta lo mismo que una sola corrida.
    """
    tabla = []
    total = len(X_test)

    for indice, punto in enumerate(X_test):
        tabla.append(vecinos_mas_cercanos(X_train, y_train, punto, k_maximo))

        if (indice + 1) % 500 == 0:
            print("   ... {}/{} puntos de prueba procesados".format(indice + 1, total))

    return tabla


# ==============================================================================
# 4. METRICAS DE EVALUACION (programadas a mano)
# ==============================================================================

def accuracy(y_real, y_pred):
    """Proporcion de predicciones correctas sobre el total."""
    correctos = 0
    for i in range(len(y_real)):
        if y_real[i] == y_pred[i]:
            correctos += 1
    return correctos / len(y_real)


def matriz_confusion(y_real, y_pred, clases=None):
    """Construye la matriz de confusion.

    matriz[i][j] = cuantos casos cuya clase REAL es clases[i] fueron
    PREDICHOS como clases[j]. La diagonal son los aciertos y todo lo que
    queda fuera de la diagonal son errores.
    """
    if clases is None:
        clases = sorted(set(y_real) | set(y_pred))

    indice_de = {clase: i for i, clase in enumerate(clases)}

    matriz = [[0] * len(clases) for _ in range(len(clases))]

    for i in range(len(y_real)):
        fila = indice_de[y_real[i]]
        columna = indice_de[y_pred[i]]
        matriz[fila][columna] += 1

    return clases, matriz


def calcular_metricas(clases, matriz, imprimir=True):
    """Precision, recall y F1 de cada clase a partir de la matriz de confusion.

    Para la clase i:
        TP = matriz[i][i]                    aciertos de la clase
        FP = suma de la COLUMNA i sin TP     otras clases predichas como i
        FN = suma de la FILA i sin TP        casos de i predichos como otra cosa

        precision = TP / (TP + FP)   de lo que predije como i, cuanto era i
        recall    = TP / (TP + FN)   de todos los i que habia, cuantos encontre
        F1        = media armonica de precision y recall

    El dataset esta muy desbalanceado (NUEVO tiene 4498 casos y DEMO solo 229),
    por eso el accuracy global no basta y se reportan las metricas por clase.
    """
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

        precision = TP / (TP + FP) if TP + FP > 0 else 0.0
        recall = TP / (TP + FN) if TP + FN > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        precisiones.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        if imprimir:
            soporte = TP + FN
            print("   {:<12} precision {:.3f}   recall {:.3f}   F1 {:.3f}   (n={})".format(
                clases[i], precision, recall, f1, soporte))

    return precisiones, recalls, f1_scores


def f1_macro(f1_scores):
    """Promedio simple del F1 de cada clase.

    Se elige el F1 macro como metrica principal porque trata a todas las clases
    con el mismo peso: si el modelo ignorara por completo a DEMO o a REFACC el
    accuracy casi no se movería, pero el F1 macro se desplomaria.
    """
    return sum(f1_scores) / len(f1_scores)


# ==============================================================================
# 5. PRUEBAS DE CORRECTITUD DEL ALGORITMO
# ==============================================================================
#
# Antes de confiar en los resultados sobre el dataset real se verifica que cada
# pieza del algoritmo devuelva exactamente lo que se espera en casos pequeños
# resueltos a mano. Si alguna prueba falla el programa se detiene.
# ==============================================================================

def probar_implementacion():
    print("=" * 78)
    print(" PRUEBAS DE CORRECTITUD DE LA IMPLEMENTACION")
    print("=" * 78)

    # --- Prueba 1: distancia euclidiana contra el triangulo 3-4-5 -------------
    d = distancia_euclidiana([0, 0], [3, 4])
    assert abs(d - 5.0) < 1e-9, "distancia_euclidiana incorrecta"
    print(" [OK] distancia_euclidiana([0,0],[3,4]) = {:.1f}  (esperado 5.0)".format(d))

    # --- Prueba 2: media y desviacion contra un calculo manual ---------------
    # Datos: 2, 4, 4, 4, 5, 5, 7, 9  ->  media 5, desviacion poblacional 2
    medias, desviaciones = calcular_media_desviacion([[2], [4], [4], [4], [5], [5], [7], [9]])
    assert abs(medias[0] - 5.0) < 1e-9 and abs(desviaciones[0] - 2.0) < 1e-9
    print(" [OK] media = {:.1f} y desviacion = {:.1f}  (esperado 5.0 y 2.0)".format(
        medias[0], desviaciones[0]))

    # --- Prueba 3: la votacion respeta la mayoria -----------------------------
    # 3 votos para B contra 2 para A: debe ganar B aunque A este mas cerca.
    ganador = votar(["A", "B", "B", "A", "B"])
    assert ganador == "B", "la votacion no respeta la mayoria"
    print(" [OK] votar(['A','B','B','A','B']) = '{}'  (esperado 'B')".format(ganador))

    # --- Prueba 4: empate -> gana el vecino mas cercano ----------------------
    ganador = votar(["A", "B", "B", "A"])
    assert ganador == "A", "el desempate no usa al vecino mas cercano"
    print(" [OK] votar(['A','B','B','A']) = '{}'  (empate, gana el mas cercano)".format(ganador))

    # --- Prueba 5: KNN completo sobre un problema de juguete -----------------
    # Dos grupos bien separados en la recta. El punto 0.9 esta rodeado de ROJO.
    Xj = [[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]]
    yj = ["ROJO", "ROJO", "ROJO", "AZUL", "AZUL", "AZUL"]
    assert predecir_knn(Xj, yj, [0.9], 3) == "ROJO"
    assert predecir_knn(Xj, yj, [11.2], 3) == "AZUL"
    print(" [OK] KNN de juguete: 0.9 -> ROJO y 11.2 -> AZUL  (grupos separados)")

    # --- Prueba 6: con k=1 el modelo memoriza su propio entrenamiento --------
    # Es la propiedad basica de KNN: el vecino mas cercano de un punto del
    # entrenamiento es el mismo, asi que el accuracy de train con k=1 es 100%.
    predicciones = predecir_todos(Xj, yj, Xj, 1)
    assert accuracy(yj, predicciones) == 1.0
    print(" [OK] k=1 reproduce el 100% del entrenamiento  (memorizacion)")

    # --- Prueba 7: matriz de confusion y metricas contra numeros a mano ------
    # Real:  A A A B B      Predicho:  A A B B B
    # Clase A -> TP=2, FN=1, FP=0  -> precision 1.000, recall 0.667, F1 0.800
    clases, matriz = matriz_confusion(["A", "A", "A", "B", "B"],
                                      ["A", "A", "B", "B", "B"])
    assert matriz == [[2, 1], [0, 2]], "matriz de confusion incorrecta"
    p, r, f = calcular_metricas(clases, matriz, imprimir=False)
    assert abs(p[0] - 1.0) < 1e-9 and abs(r[0] - 2 / 3) < 1e-9 and abs(f[0] - 0.8) < 1e-9
    print(" [OK] matriz de confusion = [[2,1],[0,2]] y F1(A) = {:.3f}  (esperado 0.800)".format(f[0]))

    print("\n Las 7 pruebas pasaron: el algoritmo esta implementado correctamente.\n")


# ==============================================================================
# 6. GRAFICAS
# ==============================================================================

def guardar(nombre):
    """Guarda la figura actual dentro de la carpeta figuras/."""
    if not os.path.isdir(CARPETA_FIGURAS):
        os.makedirs(CARPETA_FIGURAS)
    ruta = os.path.join(CARPETA_FIGURAS, nombre)
    plt.savefig(ruta, dpi=140)
    plt.close()
    print("   figura guardada:", os.path.relpath(ruta, CARPETA))


def graficar_matriz(clases, matriz, titulo, nombre):
    """Dibuja la matriz de confusion normalizada por fila (% de cada clase real)."""
    normalizada = []
    for fila in matriz:
        total = sum(fila)
        if total == 0:
            normalizada.append([0] * len(fila))
        else:
            normalizada.append([valor / total for valor in fila])

    plt.figure(figsize=(7.5, 6))
    plt.imshow(normalizada, cmap="Blues", vmin=0, vmax=1)
    plt.title(titulo)
    plt.xlabel("Prediccion")
    plt.ylabel("Clase real")
    plt.xticks(range(len(clases)), clases, rotation=45, ha="right")
    plt.yticks(range(len(clases)), clases)

    for i in range(len(clases)):
        for j in range(len(clases)):
            color = "white" if normalizada[i][j] > 0.5 else "black"
            plt.text(j, i, "{}\n{:.0%}".format(matriz[i][j], normalizada[i][j]),
                     ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(label="proporcion de la fila")
    plt.tight_layout()
    guardar(nombre)


def graficar_metricas(clases, precision, recall, f1):
    """Compara precision, recall y F1 de cada clase en una sola grafica."""
    posiciones = list(range(len(clases)))
    ancho = 0.27

    plt.figure(figsize=(8, 5))
    plt.bar([p - ancho for p in posiciones], precision, ancho, label="Precision")
    plt.bar(posiciones, recall, ancho, label="Recall")
    plt.bar([p + ancho for p in posiciones], f1, ancho, label="F1")
    plt.xticks(posiciones, clases, rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.title("Metricas por clase (KNN sin framework, k={})".format(K))
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    guardar("metricas_por_clase.png")


def graficar_barrido_k(resultados):
    """Accuracy y F1 macro en funcion de k."""
    ks = [r["k"] for r in resultados]
    accs = [r["accuracy"] for r in resultados]
    f1s = [r["f1_macro"] for r in resultados]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, accs, marker="o", label="Accuracy")
    plt.plot(ks, f1s, marker="s", label="F1 macro")
    plt.axvline(K, color="gray", linestyle="--", alpha=0.7, label="k elegido = {}".format(K))
    plt.xticks(ks)
    plt.xlabel("k (numero de vecinos)")
    plt.ylabel("Valor sobre el conjunto de prueba")
    plt.title("Efecto del numero de vecinos")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    guardar("barrido_k.png")


def graficar_distribucion(y_train, y_test, clases):
    """Muestra que la particion conserva la proporcion de cada clase."""
    def cuenta(lista):
        return [sum(1 for v in lista if v == c) for c in clases]

    n_train = cuenta(y_train)
    n_test = cuenta(y_test)
    posiciones = list(range(len(clases)))
    ancho = 0.38

    plt.figure(figsize=(8, 5))
    plt.bar([p - ancho / 2 for p in posiciones], n_train, ancho, label="Entrenamiento")
    plt.bar([p + ancho / 2 for p in posiciones], n_test, ancho, label="Prueba")
    plt.xticks(posiciones, clases, rotation=45, ha="right")
    plt.ylabel("Numero de registros")
    plt.title("Distribucion de clases (dataset desbalanceado)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    guardar("distribucion_clases.png")


# ==============================================================================
# 7. PREDICCIONES SOBRE CASOS NUEVOS
# ==============================================================================

def predecir_casos_nuevos(X_train, y_train, medias, desviaciones):
    """Usa el modelo ya construido para clasificar creditos que no estan en el CSV.

    Es la demostracion de que el programa puede hacer predicciones reales:
    se le dan los datos crudos de un credito y devuelve el tipo de garantia
    junto con el reparto de votos de los k vecinos.
    """
    casos = [
        ("Credito grande, auto del año",      350000.0, 1500.0, 2026, 2025, 2),
        ("Credito muy grande, auto nuevo",   1200000.0, 9000.0, 2026, 2026, 5),
        ("Credito chico, auto con 4 años",     90000.0,  400.0, 2025, 2021, 8),
    ]

    print("=" * 78)
    print(" PREDICCIONES SOBRE CASOS NUEVOS (datos que no estan en el dataset)")
    print("=" * 78)

    for descripcion, principal, interes, year, year_model, mes in casos:
        tasa = interes / principal if principal > 0 else 0.0
        antiguedad = float(year - year_model)
        crudo = [principal, interes, tasa, antiguedad, float(mes)]

        # El caso nuevo se normaliza con la MISMA media y desviacion del entrenamiento.
        punto = normalizar([crudo], medias, desviaciones)[0]

        vecinos = vecinos_mas_cercanos(X_train, y_train, punto, K)
        prediccion = votar(vecinos)

        print("\n " + descripcion)
        print("   principal ${:>12,.2f} | interes ${:>9,.2f} | tasa {:.5f}".format(
            principal, interes, tasa))
        print("   antiguedad {} años | mes {}".format(int(antiguedad), mes))
        print("   --> PREDICCION: {}".format(prediccion))

        votos = {}
        for clase in vecinos:
            votos[clase] = votos.get(clase, 0) + 1
        reparto = ", ".join("{} {}/{}".format(c, n, K) for c, n in sorted(votos.items()))
        print("       votos de los {} vecinos: {}".format(K, reparto))

    print()


# ==============================================================================
# PROGRAMA PRINCIPAL
# ==============================================================================

def main():
    probar_implementacion()

    # ---------------------------------------------------------------- datos --
    print("=" * 78)
    print(" DATOS")
    print("=" * 78)

    X, y = cargar_datos()
    X_train, y_train, X_test, y_test = dividir_datos(X, y)

    clases_totales = sorted(set(y))

    print(" Archivo: interest_ledger.csv")
    print(" Variables usadas: " + ", ".join(NOMBRES_VARIABLES))
    print(" Registros totales : {}".format(len(X)))
    print(" Entrenamiento     : {}  ({:.0%})".format(len(X_train), len(X_train) / len(X)))
    print(" Prueba            : {}  ({:.0%})".format(len(X_test), len(X_test) / len(X)))
    print("\n Registros por clase:")
    for clase in clases_totales:
        n_tr = sum(1 for v in y_train if v == clase)
        n_te = sum(1 for v in y_test if v == clase)
        print("   {:<12} total {:>5}   train {:>5}   test {:>5}".format(
            clase, n_tr + n_te, n_tr, n_te))

    # -------------------------------------------------------- normalizacion --
    medias, desviaciones = calcular_media_desviacion(X_train)

    print("\n Normalizacion z-score ajustada SOLO con entrenamiento:")
    for i in range(len(NOMBRES_VARIABLES)):
        print("   {:<12} media {:>14,.4f}   desviacion {:>14,.4f}".format(
            NOMBRES_VARIABLES[i], medias[i], desviaciones[i]))

    X_train_n = normalizar(X_train, medias, desviaciones)
    X_test_n = normalizar(X_test, medias, desviaciones)

    # --------------------------------------------------- vecinos y barrido k --
    print("\n" + "=" * 78)
    print(" CALCULO DE VECINOS")
    print("=" * 78)
    print(" Midiendo la distancia de cada uno de los {} puntos de prueba".format(len(X_test_n)))
    print(" contra los {} puntos de entrenamiento...".format(len(X_train_n)))

    k_maximo = max(VALORES_K + [K])
    tabla_vecinos = vecinos_de_todos(X_train_n, y_train, X_test_n, k_maximo)

    print("\n" + "=" * 78)
    print(" COMPARACION DE VALORES DE k")
    print("=" * 78)
    print(" {:>4}   {:>10}   {:>10}".format("k", "accuracy", "F1 macro"))

    resultados_k = []
    for k in VALORES_K:
        predicciones_k = [votar(vecinos[:k]) for vecinos in tabla_vecinos]
        clases_k, matriz_k = matriz_confusion(y_test, predicciones_k, clases_totales)
        _, _, f1_k = calcular_metricas(clases_k, matriz_k, imprimir=False)

        acc = accuracy(y_test, predicciones_k)
        macro = f1_macro(f1_k)
        resultados_k.append({"k": k, "accuracy": acc, "f1_macro": macro})

        marca = "  <-- elegido" if k == K else ""
        print(" {:>4}   {:>9.2f}%   {:>10.3f}{}".format(k, acc * 100, macro, marca))

    # ------------------------------------------------- evaluacion del modelo --
    predicciones = [votar(vecinos[:K]) for vecinos in tabla_vecinos]

    print("\n" + "=" * 78)
    print(" RESULTADOS DEL MODELO FINAL (k = {})".format(K))
    print("=" * 78)
    print(" Accuracy : {:.2f}%".format(accuracy(y_test, predicciones) * 100))

    clases, matriz = matriz_confusion(y_test, predicciones, clases_totales)

    print("\n Matriz de confusion (filas = clase real, columnas = prediccion):")
    print("   {:<12}".format("") + "".join("{:>11}".format(c[:10]) for c in clases))
    for i in range(len(clases)):
        print("   {:<12}".format(clases[i]) + "".join("{:>11}".format(v) for v in matriz[i]))

    print("\n Metricas por clase:")
    precision, recall, f1 = calcular_metricas(clases, matriz)
    print("\n F1 macro : {:.3f}".format(f1_macro(f1)))

    # --------------------------------------------------------- predicciones --
    print()
    predecir_casos_nuevos(X_train_n, y_train, medias, desviaciones)

    # ------------------------------------------------------------- graficas --
    print("=" * 78)
    print(" GRAFICAS")
    print("=" * 78)
    graficar_distribucion(y_train, y_test, clases_totales)
    graficar_matriz(clases, matriz,
                    "Matriz de confusion - KNN sin framework (k={})".format(K),
                    "matriz_confusion.png")
    graficar_metricas(clases, precision, recall, f1)
    graficar_barrido_k(resultados_k)
    print("\n Listo.")


if __name__ == "__main__":
    main()

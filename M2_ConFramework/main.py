"""
Random Forest con scikit-learn.
Clasifica el tipo de garantia de un credito automotriz.

Es el mismo problema y las mismas variables de la entrega sin framework, para
poder comparar el KNN hecho a mano contra un modelo configurado con framework.

Ejecutar: python3 main.py
"""

import os

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report, balanced_accuracy_score)


# -----------------------------
# CONFIGURACION
# -----------------------------

CARPETA = os.path.dirname(os.path.abspath(__file__))
RUTA = os.path.join(CARPETA, "interest_ledger.csv")
FIGURAS = os.path.join(CARPETA, "figuras")

# Las mismas cinco variables que use en la entrega sin framework.
COLUMNAS = ["principal_amount", "interest_amount", "tasa", "antiguedad", "mes"]

SEMILLA = 42
PORCENTAJE_TEST = 0.25

# Resultados del KNN de la entrega anterior, para comparar al final.
KNN_ACCURACY = 81.51
KNN_F1_MACRO = 0.739


# -----------------------------
# CARGAR DATOS
# -----------------------------

def cargar_datos():
    datos = pd.read_csv(RUTA)

    # El CSV trae espacios raros (\xa0) pegados a la clase.
    datos["collateral"] = (datos["collateral"]
                           .str.replace("\xa0", " ", regex=False)
                           .str.strip().str.upper())

    datos["year"] = datos["month_date"].str[:4].astype(int)
    datos["mes"] = datos["month_date"].str[5:7].astype(int)

    # Tasa mensual: cuanto interes genera cada peso prestado.
    datos["tasa"] = 0.0
    con_principal = datos["principal_amount"] > 0
    datos.loc[con_principal, "tasa"] = (datos.loc[con_principal, "interest_amount"] /
                                        datos.loc[con_principal, "principal_amount"])

    # Años que tenia el vehiculo al momento del pago.
    datos["antiguedad"] = datos["year"] - datos["year_model"]

    return datos[COLUMNAS], datos["collateral"]


# -----------------------------
# EL MODELO
# -----------------------------
# Random Forest no necesita normalizar: cada arbol parte las variables por
# umbrales y un corte funciona igual sin importar la escala de la columna.

def crear_modelo(config):
    return RandomForestClassifier(
        n_estimators=config["n_estimators"],    # numero de arboles del bosque
        max_depth=config["max_depth"],          # hasta donde crece cada arbol
        min_samples_leaf=config["min_samples_leaf"],  # minimo de datos por hoja
        max_features="sqrt",                    # cada corte ve solo algunas columnas
        class_weight=config["class_weight"],    # peso de cada clase
        random_state=SEMILLA,
        n_jobs=-1,                              # usa todos los nucleos
    )


# -----------------------------
# SELECCION DE HIPERPARAMETROS
# -----------------------------
# Se elige con validacion cruzada SOBRE EL ENTRENAMIENTO. El conjunto de prueba
# no se toca: si se usara para elegir, su resultado dejaria de ser una
# estimacion honesta del desempeño con datos nuevos.

CONFIGURACIONES = [
    {"n_estimators": 100, "max_depth": 3,    "min_samples_leaf": 1,  "class_weight": None},
    {"n_estimators": 100, "max_depth": 5,    "min_samples_leaf": 1,  "class_weight": None},
    {"n_estimators": 100, "max_depth": 10,   "min_samples_leaf": 1,  "class_weight": None},
    {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1,  "class_weight": None},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1,  "class_weight": None},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 5,  "class_weight": None},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 20, "class_weight": None},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1,  "class_weight": "balanced"},
]


def comparar_configuraciones(X_train, y_train):
    # Se compara con F1 macro y no con accuracy porque las clases estan muy
    # desbalanceadas: un modelo que ignorara DEMO perderia 2 puntos de accuracy
    # pero hundiria el F1 macro, que promedia las 5 clases por igual.
    particion = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

    print("\n" + "=" * 80)
    print(" VALIDACION CRUZADA (5 partes) SOBRE ENTRENAMIENTO")
    print("=" * 80)
    print(" {:>8} {:>10} {:>10} {:>11} {:>14} {:>9}".format(
        "arboles", "max_depth", "min_leaf", "pesos", "F1 macro", "desv"))

    resultados = []
    for config in CONFIGURACIONES:
        marcas = cross_val_score(crear_modelo(config), X_train, y_train,
                                 cv=particion, scoring="f1_macro", n_jobs=-1)

        resultados.append({"config": config, "media": marcas.mean(), "desv": marcas.std()})

        print(" {:>8} {:>10} {:>10} {:>11} {:>14.4f} {:>9.4f}".format(
            config["n_estimators"], str(config["max_depth"]),
            config["min_samples_leaf"], str(config["class_weight"]),
            marcas.mean(), marcas.std()))

    mejor = max(resultados, key=lambda r: r["media"])
    print("\n Mejor configuracion:", mejor["config"])

    return resultados, mejor["config"]


# -----------------------------
# METRICAS
# -----------------------------

def evaluar(modelo, X, y, nombre, clases, detalle=True):
    y_pred = modelo.predict(X)

    acc = accuracy_score(y, y_pred)
    f1m = f1_score(y, y_pred, average="macro")

    print("\n" + "-" * 80)
    print(" {}  (n = {})".format(nombre, len(y)))
    print("-" * 80)
    print("   Accuracy           :", round(acc * 100, 2), "%")
    print("   Accuracy balanceado:", round(balanced_accuracy_score(y, y_pred) * 100, 2), "%")
    print("   F1 macro           :", round(f1m, 4))
    print("   F1 ponderado       :", round(f1_score(y, y_pred, average="weighted"), 4))

    matriz = confusion_matrix(y, y_pred, labels=clases)

    if detalle:
        # Las filas son la clase real y las columnas la predicha.
        print("\n   Matriz de confusion:")
        print("     {:<12}".format("") + "".join("{:>11}".format(c[:10]) for c in clases))
        for i, clase in enumerate(clases):
            print("     {:<12}".format(clase) + "".join("{:>11}".format(v) for v in matriz[i]))

        print("\n   Reporte por clase:")
        print(classification_report(y, y_pred, labels=clases, digits=3, zero_division=0))

    return {"accuracy": acc, "f1_macro": f1m, "matriz": matriz}


def linea_base(X_train, y_train, X_test, y_test):
    # Modelo tonto que siempre predice la clase mayoritaria. Sirve para saber
    # desde donde se parte: si el bosque no lo supera, no aprendio nada.
    tonto = DummyClassifier(strategy="most_frequent")
    tonto.fit(X_train, y_train)
    y_pred = tonto.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n" + "=" * 80)
    print(" LINEA BASE: siempre la clase mayoritaria")
    print("=" * 80)
    print("   Accuracy:", round(acc * 100, 2), "% | F1 macro:", round(f1m, 4))

    return {"accuracy": acc, "f1_macro": f1m}


# -----------------------------
# PREDICCIONES NUEVAS
# -----------------------------

def predecir_casos_nuevos(modelo):
    casos = pd.DataFrame([
        [350000.0,  1500.0, 1500.0 / 350000.0,  1, 2],
        [1200000.0, 9000.0, 9000.0 / 1200000.0, 0, 5],
        [90000.0,    400.0,  400.0 / 90000.0,   4, 8],
        [45000.0,    260.0,  260.0 / 45000.0,   2, 11],
    ], columns=COLUMNAS)

    descripciones = ["Credito mediano, vehiculo de un año",
                     "Credito muy grande, vehiculo del año",
                     "Credito chico, vehiculo con 4 años",
                     "Credito muy chico, vehiculo con 2 años"]

    predicciones = modelo.predict(casos)
    probabilidades = modelo.predict_proba(casos)

    print("\n" + "=" * 80)
    print(" PREDICCIONES DE CASOS NUEVOS")
    print("=" * 80)

    for i in range(len(casos)):
        print("\n", descripciones[i])
        print("   principal ${:,.2f} | interes ${:,.2f} | antiguedad {} | mes {}".format(
            casos["principal_amount"][i], casos["interest_amount"][i],
            casos["antiguedad"][i], casos["mes"][i]))
        print("   PREDICCION:", predicciones[i])

        # predict_proba es la proporcion de arboles que voto por cada clase.
        orden = np.argsort(probabilidades[i])[::-1]
        print("   confianza:", "  ".join(
            "{} {:.1f}%".format(modelo.classes_[j], probabilidades[i][j] * 100)
            for j in orden if probabilidades[i][j] > 0.005))


# -----------------------------
# GRAFICAS
# -----------------------------

def guardar(nombre):
    if not os.path.isdir(FIGURAS):
        os.makedirs(FIGURAS)
    plt.savefig(os.path.join(FIGURAS, nombre), dpi=140)
    plt.close()
    print("   figura guardada: figuras/" + nombre)


def graficar_matriz(clases, matriz):
    matriz = np.asarray(matriz, dtype=float)
    normalizada = matriz / np.clip(matriz.sum(axis=1, keepdims=True), 1, None)

    plt.figure(figsize=(7.5, 6))
    plt.imshow(normalizada, cmap="Blues", vmin=0, vmax=1)
    plt.title("Matriz de Confusion (Random Forest)")
    plt.xlabel("Prediccion")
    plt.ylabel("Clase real")
    plt.xticks(range(len(clases)), clases, rotation=45, ha="right")
    plt.yticks(range(len(clases)), clases)

    for i in range(len(clases)):
        for j in range(len(clases)):
            color = "white" if normalizada[i, j] > 0.5 else "black"
            plt.text(j, i, "{:.0f}\n{:.0%}".format(matriz[i, j], normalizada[i, j]),
                     ha="center", va="center", fontsize=8, color=color)

    plt.colorbar()
    plt.tight_layout()
    guardar("matriz_confusion.png")


def graficar_importancias(modelo):
    # El modelo guarda que tanto ayudo cada columna a separar las clases.
    importancias = modelo.feature_importances_
    desviaciones = np.std([a.feature_importances_ for a in modelo.estimators_], axis=0)
    orden = np.argsort(importancias)[::-1]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(COLUMNAS)), importancias[orden], yerr=desviaciones[orden], capsize=4)
    plt.xticks(range(len(COLUMNAS)), [COLUMNAS[i] for i in orden], rotation=30, ha="right")
    plt.ylabel("Importancia")
    plt.title("Importancia de cada variable")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    guardar("importancias.png")


def graficar_configuraciones(resultados):
    etiquetas = ["{}a d={} l={}{}".format(
        r["config"]["n_estimators"], r["config"]["max_depth"],
        r["config"]["min_samples_leaf"], " bal" if r["config"]["class_weight"] else "")
        for r in resultados]
    medias = [r["media"] for r in resultados]

    plt.figure(figsize=(10, 5))
    colores = ["tab:orange" if m == max(medias) else "tab:blue" for m in medias]
    plt.bar(range(len(medias)), medias, yerr=[r["desv"] for r in resultados],
            capsize=4, color=colores)
    plt.xticks(range(len(medias)), etiquetas, rotation=35, ha="right", fontsize=8)
    plt.ylabel("F1 macro en validacion cruzada")
    plt.ylim(min(medias) - 0.05, max(medias) + 0.03)
    plt.title("Comparacion de configuraciones")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    guardar("configuraciones.png")


def graficar_comparacion(base, rf):
    modelos = ["Linea base", "KNN sin framework", "Random Forest"]
    accs = [base["accuracy"] * 100, KNN_ACCURACY, rf["accuracy"] * 100]
    f1s = [base["f1_macro"], KNN_F1_MACRO, rf["f1_macro"]]

    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.5))

    for eje, valores, titulo, tope in [(ejes[0], accs, "Accuracy (%)", 105),
                                       (ejes[1], f1s, "F1 macro", 1.05)]:
        eje.bar(modelos, valores, color=["gray", "tab:blue", "tab:green"])
        eje.set_ylabel(titulo)
        eje.set_title(titulo + " en prueba")
        eje.set_ylim(0, tope)
        eje.grid(axis="y", alpha=0.3)
        for i, v in enumerate(valores):
            eje.text(i, v + tope * 0.015, round(v, 3), ha="center", fontsize=9)
        eje.tick_params(axis="x", labelrotation=15)

    plt.tight_layout()
    guardar("comparacion_modelos.png")


# -----------------------------
# PRINCIPAL
# -----------------------------

X, y = cargar_datos()

# stratify reparte las clases en la misma proporcion en train y en test, si no
# las clases chicas (DEMO, REFACC) podrian quedar casi todas de un solo lado.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=PORCENTAJE_TEST, random_state=SEMILLA, stratify=y)

clases = sorted(y.unique())

print("Total de datos:", len(X))
print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))

print("\nDatos por clase:")
for clase in clases:
    n_tr = int((y_train == clase).sum())
    n_te = int((y_test == clase).sum())
    print("   {:<12} train {:>5}   test {:>5}   ({:.1f}% del total)".format(
        clase, n_tr, n_te, 100 * (n_tr + n_te) / len(y)))

base = linea_base(X_train, y_train, X_test, y_test)
resultados_cv, mejor_config = comparar_configuraciones(X_train, y_train)

modelo = crear_modelo(mejor_config)
modelo.fit(X_train, y_train)

print("\n" + "=" * 80)
print(" MODELO FINAL")
print("=" * 80)
print(" Configuracion:", mejor_config)
print(" Profundidad media de los arboles:",
      round(np.mean([a.get_depth() for a in modelo.estimators_]), 1))
print(" Hojas por arbol:", round(np.mean([a.get_n_leaves() for a in modelo.estimators_])))

entrenamiento = evaluar(modelo, X_train, y_train, "ENTRENAMIENTO", clases, detalle=False)
prueba = evaluar(modelo, X_test, y_test, "PRUEBA", clases)

print(" Brecha entrenamiento - prueba en F1 macro:",
      round(entrenamiento["f1_macro"] - prueba["f1_macro"], 4))
print(" (esta brecha se analiza a fondo en la entrega M2_Analisis)")

print("\nImportancia de cada variable:")
for i in np.argsort(modelo.feature_importances_)[::-1]:
    print("   {:<20} {:.4f}".format(COLUMNAS[i], modelo.feature_importances_[i]))

predecir_casos_nuevos(modelo)

print("\n" + "=" * 80)
print(" GRAFICAS")
print("=" * 80)
graficar_matriz(clases, prueba["matriz"])
graficar_importancias(modelo)
graficar_configuraciones(resultados_cv)
graficar_comparacion(base, prueba)

print("\n" + "=" * 80)
print(" RESUMEN")
print("=" * 80)
print(" {:<26} {:>10} {:>10}".format("modelo", "accuracy", "F1 macro"))
print(" {:<26} {:>9.2f}% {:>10.4f}".format("Linea base", base["accuracy"] * 100, base["f1_macro"]))
print(" {:<26} {:>9.2f}% {:>10.4f}".format("KNN sin framework", KNN_ACCURACY, KNN_F1_MACRO))
print(" {:<26} {:>9.2f}% {:>10.4f}".format("Random Forest", prueba["accuracy"] * 100, prueba["f1_macro"]))

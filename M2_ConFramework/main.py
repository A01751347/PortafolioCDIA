"""
================================================================================
 RANDOM FOREST CON FRAMEWORK (scikit-learn)
================================================================================

 Modulo 2 - Portafolio de Implementacion
 Santiago Serrano Montalvo - A01751347

 PROBLEMA
 --------
 El mismo de la entrega sin framework: clasificar el tipo de garantia
 (`collateral`) de un credito automotriz. Usar el mismo problema y las mismas
 variables permite comparar de forma justa el KNN programado a mano contra un
 modelo configurado con un framework profesional.

 POR QUE RANDOM FOREST
 ---------------------
 El KNN dejo ver dos problemas del dataset: las clases NUEVO y FLOTILLA se
 traslapan en el espacio de variables, y las clases DEMO y REFACC son muy
 pequeñas. Random Forest ataca los dos:

   * Es un ENSAMBLE de arboles de decision. Cada arbol se entrena sobre una
     muestra bootstrap distinta (muestreo con reemplazo) y en cada corte solo
     considera un subconjunto aleatorio de variables (max_features="sqrt").
     Esa doble aleatoriedad hace que los arboles se equivoquen en lugares
     diferentes, y al promediar sus votos los errores se cancelan: la varianza
     del ensamble baja mucho respecto a la de un arbol individual.

   * Los arboles cortan cada variable por separado con umbrales, asi que no
     necesitan normalizacion y pueden construir fronteras de decision con
     "escalones" que una frontera basada en distancia (KNN) no puede formar.

 QUE SE DEMUESTRA AQUI SOBRE EL FRAMEWORK
 ----------------------------------------
   * train_test_split con `stratify` para conservar la proporcion de clases.
   * StratifiedKFold + cross_val_score para elegir hiperparametros SIN tocar
     el conjunto de prueba.
   * Configuracion explicita del estimador (n_estimators, max_depth,
     min_samples_leaf, max_features, class_weight).
   * Evaluacion con confusion_matrix y classification_report.
   * predict y predict_proba para generar predicciones nuevas en consola.

 EJECUCION
 ---------
       python3 main.py

================================================================================
"""

import os

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # backend sin ventana: corre en consola pura
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report, balanced_accuracy_score)


# ==============================================================================
# CONFIGURACION
# ==============================================================================

CARPETA = os.path.dirname(os.path.abspath(__file__))
RUTA_CSV = os.path.join(CARPETA, "interest_ledger.csv")
CARPETA_FIGURAS = os.path.join(CARPETA, "figuras")

# Exactamente las mismas 5 variables que se usaron en la entrega sin framework.
COLUMNAS = ["principal_amount", "interest_amount", "tasa", "antiguedad", "mes"]

SEMILLA = 42
PORCENTAJE_TEST = 0.25


# ==============================================================================
# 1. CARGA Y PREPARACION DE LOS DATOS
# ==============================================================================

def cargar_datos(ruta=RUTA_CSV):
    """Lee el CSV con pandas y deriva las variables que alimentan al modelo."""
    datos = pd.read_csv(ruta)

    # El CSV trae espacios duros (\xa0) pegados al nombre de la clase.
    datos["collateral"] = (datos["collateral"]
                           .str.replace("\xa0", " ", regex=False)
                           .str.strip()
                           .str.upper())

    # De la fecha del pago solo interesan el año y el mes.
    datos["year"] = datos["month_date"].str[:4].astype(int)
    datos["mes"] = datos["month_date"].str[5:7].astype(int)

    # Tasa mensual efectiva: cuanto interes genera cada peso prestado.
    datos["tasa"] = 0.0
    con_principal = datos["principal_amount"] > 0
    datos.loc[con_principal, "tasa"] = (datos.loc[con_principal, "interest_amount"] /
                                        datos.loc[con_principal, "principal_amount"])

    # Años que tenia el vehiculo al momento del pago.
    datos["antiguedad"] = datos["year"] - datos["year_model"]

    return datos[COLUMNAS], datos["collateral"]


# ==============================================================================
# 2. BUSQUEDA DE HIPERPARAMETROS CON VALIDACION CRUZADA
# ==============================================================================
#
# La seleccion del modelo se hace con validacion cruzada estratificada de 5
# particiones SOBRE EL CONJUNTO DE ENTRENAMIENTO. El conjunto de prueba no se
# toca hasta el final: si se usara para elegir hiperparametros, su resultado
# dejaria de ser una estimacion honesta del desempeño con datos nuevos.
# ==============================================================================

CONFIGURACIONES = [
    # Modelos poco flexibles (arboles cortos) -> se espera sesgo alto.
    {"n_estimators": 100, "max_depth": 3,    "min_samples_leaf": 1,  "class_weight": None},
    {"n_estimators": 100, "max_depth": 5,    "min_samples_leaf": 1,  "class_weight": None},
    {"n_estimators": 100, "max_depth": 10,   "min_samples_leaf": 1,  "class_weight": None},
    # Arboles sin limite de profundidad -> maxima flexibilidad.
    {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1,  "class_weight": None},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1,  "class_weight": None},
    # Regularizacion por tamaño minimo de hoja.
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 5,  "class_weight": None},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 20, "class_weight": None},
    # Reponderacion de clases para atacar el desbalance.
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1,  "class_weight": "balanced"},
]


def construir_modelo(config):
    """Crea un RandomForestClassifier con la configuracion indicada.

    Significado de cada hiperparametro:
      n_estimators     numero de arboles del bosque. Mas arboles = voto mas
                       estable; nunca provoca sobreajuste, solo cuesta tiempo.
      max_depth        profundidad maxima de cada arbol. Es el control
                       principal de flexibilidad: poca profundidad = sesgo alto.
      min_samples_leaf minimo de registros que debe quedar en cada hoja. Subirlo
                       impide que un arbol memorice casos individuales.
      max_features     cuantas variables considera cada corte. "sqrt" toma
                       sqrt(5)~2 de las 5 y es lo que decorrelaciona los arboles.
      class_weight     peso de cada clase en el criterio de division.
                       "balanced" compensa el desbalance del dataset.
    """
    return RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features="sqrt",
        class_weight=config["class_weight"],
        random_state=SEMILLA,
        n_jobs=-1,
    )


def comparar_configuraciones(X_train, y_train):
    """Evalua cada configuracion con validacion cruzada y devuelve los resultados.

    La metrica de seleccion es el F1 MACRO, no el accuracy. Justificacion:
    el dataset esta muy desbalanceado (NUEVO 42.5% contra DEMO 2.2%), asi que
    un modelo que ignorara por completo a DEMO perderia apenas ~2 puntos de
    accuracy pero hundiria el F1 macro, que promedia el F1 de las 5 clases por
    igual. Para el negocio interesa clasificar bien TODAS las garantias, no
    solo las mayoritarias.
    """
    particion = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

    print("\n" + "=" * 86)
    print(" SELECCION DE HIPERPARAMETROS - validacion cruzada estratificada de 5 particiones")
    print(" (solo sobre el conjunto de entrenamiento; el de prueba no se toca)")
    print("=" * 86)
    print(" {:>7} {:>10} {:>10} {:>10}   {:>16} {:>10}".format(
        "arboles", "max_depth", "min_leaf", "pesos", "F1 macro (CV)", "desv"))

    resultados = []
    for config in CONFIGURACIONES:
        modelo = construir_modelo(config)
        marcas = cross_val_score(modelo, X_train, y_train, cv=particion,
                                 scoring="f1_macro", n_jobs=-1)

        resultados.append({"config": config,
                           "media": marcas.mean(),
                           "desv": marcas.std()})

        print(" {:>7} {:>10} {:>10} {:>10}   {:>16.4f} {:>10.4f}".format(
            config["n_estimators"],
            str(config["max_depth"]),
            config["min_samples_leaf"],
            str(config["class_weight"]),
            marcas.mean(), marcas.std()))

    mejor = max(resultados, key=lambda r: r["media"])
    print("\n Mejor configuracion por F1 macro en validacion cruzada:")
    print("  ", mejor["config"])

    return resultados, mejor["config"]


# ==============================================================================
# 3. EVALUACION
# ==============================================================================

def evaluar(modelo, X, y, nombre, clases, detalle=True):
    """Calcula y muestra las metricas del modelo sobre un conjunto de datos."""
    y_pred = modelo.predict(X)

    acc = accuracy_score(y, y_pred)
    bal = balanced_accuracy_score(y, y_pred)
    f1m = f1_score(y, y_pred, average="macro")
    f1w = f1_score(y, y_pred, average="weighted")

    print("\n" + "-" * 86)
    print(" {}  (n = {})".format(nombre, len(y)))
    print("-" * 86)
    print("   Accuracy          : {:.2f}%".format(acc * 100))
    print("   Accuracy balanceado: {:.2f}%   (promedio del recall de cada clase)".format(bal * 100))
    print("   F1 macro          : {:.4f}   <-- metrica de seleccion".format(f1m))
    print("   F1 ponderado      : {:.4f}".format(f1w))

    matriz = confusion_matrix(y, y_pred, labels=clases)

    if detalle:
        print("\n   Matriz de confusion (filas = real, columnas = prediccion):")
        print("     {:<12}".format("") + "".join("{:>11}".format(c[:10]) for c in clases))
        for i, clase in enumerate(clases):
            print("     {:<12}".format(clase) + "".join("{:>11}".format(v) for v in matriz[i]))

        print("\n   Reporte por clase:")
        print(classification_report(y, y_pred, labels=clases, digits=3, zero_division=0))

    return {"accuracy": acc, "balanced": bal, "f1_macro": f1m,
            "f1_weighted": f1w, "matriz": matriz, "y_pred": y_pred}


def linea_base(X_train, y_train, X_test, y_test):
    """Modelo tonto de referencia: siempre predice la clase mas frecuente.

    Sirve para dimensionar los resultados. Si el Random Forest no superara
    claramente a esta linea base, no estaria aprendiendo nada util.
    """
    tonto = DummyClassifier(strategy="most_frequent")
    tonto.fit(X_train, y_train)
    y_pred = tonto.predict(X_test)

    print("\n" + "=" * 86)
    print(" LINEA BASE - clasificador que siempre predice la clase mayoritaria")
    print("=" * 86)
    print("   Accuracy : {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    print("   F1 macro : {:.4f}".format(f1_score(y_test, y_pred, average='macro', zero_division=0)))
    print("   Cualquier modelo util tiene que superar claramente estos numeros.")

    return {"accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0)}


# ==============================================================================
# 4. PREDICCIONES SOBRE CASOS NUEVOS
# ==============================================================================

def predecir_casos_nuevos(modelo):
    """Clasifica creditos inventados que no aparecen en el dataset.

    Ademas de la clase predicha se imprime predict_proba, que es la proporcion
    de arboles del bosque que votaron por cada clase: es la confianza del
    ensamble y permite distinguir una prediccion segura de una dudosa.
    """
    casos = pd.DataFrame([
        # principal,  interes,   tasa,                 antiguedad, mes
        [350000.0,    1500.0,    1500.0 / 350000.0,    1,          2],
        [1200000.0,   9000.0,    9000.0 / 1200000.0,   0,          5],
        [90000.0,     400.0,     400.0 / 90000.0,      4,          8],
        [45000.0,     260.0,     260.0 / 45000.0,      2,         11],
    ], columns=COLUMNAS)

    descripciones = [
        "Credito mediano, vehiculo de un año",
        "Credito muy grande, vehiculo del año",
        "Credito chico, vehiculo con 4 años",
        "Credito muy chico, vehiculo con 2 años",
    ]

    predicciones = modelo.predict(casos)
    probabilidades = modelo.predict_proba(casos)

    print("\n" + "=" * 86)
    print(" PREDICCIONES SOBRE CASOS NUEVOS (no estan en el dataset)")
    print("=" * 86)

    for i in range(len(casos)):
        print("\n {}".format(descripciones[i]))
        print("   principal ${:>12,.2f} | interes ${:>9,.2f} | tasa {:.5f} | antiguedad {} | mes {}".format(
            casos["principal_amount"][i], casos["interest_amount"][i],
            casos["tasa"][i], casos["antiguedad"][i], casos["mes"][i]))
        print("   --> PREDICCION: {}".format(predicciones[i]))

        orden = np.argsort(probabilidades[i])[::-1]
        reparto = "  ".join("{} {:.1f}%".format(modelo.classes_[j], probabilidades[i][j] * 100)
                            for j in orden if probabilidades[i][j] > 0.005)
        print("       confianza del bosque: {}".format(reparto))

    print()


# ==============================================================================
# 5. GRAFICAS
# ==============================================================================

def guardar(nombre):
    if not os.path.isdir(CARPETA_FIGURAS):
        os.makedirs(CARPETA_FIGURAS)
    ruta = os.path.join(CARPETA_FIGURAS, nombre)
    plt.savefig(ruta, dpi=140)
    plt.close()
    print("   figura guardada:", os.path.relpath(ruta, CARPETA))


def graficar_matriz(clases, matriz, titulo, nombre):
    """Matriz de confusion normalizada por fila."""
    matriz = np.asarray(matriz, dtype=float)
    normalizada = matriz / np.clip(matriz.sum(axis=1, keepdims=True), 1, None)

    plt.figure(figsize=(7.5, 6))
    plt.imshow(normalizada, cmap="Blues", vmin=0, vmax=1)
    plt.title(titulo)
    plt.xlabel("Prediccion")
    plt.ylabel("Clase real")
    plt.xticks(range(len(clases)), clases, rotation=45, ha="right")
    plt.yticks(range(len(clases)), clases)

    for i in range(len(clases)):
        for j in range(len(clases)):
            color = "white" if normalizada[i, j] > 0.5 else "black"
            plt.text(j, i, "{:.0f}\n{:.0%}".format(matriz[i, j], normalizada[i, j]),
                     ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(label="proporcion de la fila")
    plt.tight_layout()
    guardar(nombre)


def graficar_importancias(modelo):
    """Importancia de cada variable con la desviacion entre arboles."""
    importancias = modelo.feature_importances_
    desviaciones = np.std([arbol.feature_importances_ for arbol in modelo.estimators_], axis=0)
    orden = np.argsort(importancias)[::-1]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(COLUMNAS)), importancias[orden], yerr=desviaciones[orden], capsize=4)
    plt.xticks(range(len(COLUMNAS)), [COLUMNAS[i] for i in orden], rotation=30, ha="right")
    plt.ylabel("Reduccion media de impureza")
    plt.title("Importancia de cada variable (barras = desviacion entre arboles)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    guardar("importancias.png")


def graficar_configuraciones(resultados):
    """F1 macro de cada configuracion probada en validacion cruzada."""
    etiquetas = ["{}a d={} l={}{}".format(
        r["config"]["n_estimators"],
        r["config"]["max_depth"],
        r["config"]["min_samples_leaf"],
        " bal" if r["config"]["class_weight"] else "") for r in resultados]
    medias = [r["media"] for r in resultados]
    desvs = [r["desv"] for r in resultados]

    plt.figure(figsize=(10, 5))
    colores = ["tab:orange" if m == max(medias) else "tab:blue" for m in medias]
    plt.bar(range(len(medias)), medias, yerr=desvs, capsize=4, color=colores)
    plt.xticks(range(len(medias)), etiquetas, rotation=35, ha="right", fontsize=8)
    plt.ylabel("F1 macro en validacion cruzada")
    plt.ylim(min(medias) - 0.05, max(medias) + 0.03)
    plt.title("Comparacion de configuraciones (naranja = ganadora)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    guardar("configuraciones.png")


def graficar_comparacion(base, rf_test, knn_f1, knn_acc):
    """Compara la linea base, el KNN sin framework y el Random Forest."""
    modelos = ["Linea base\n(mayoritaria)", "KNN sin framework\n(k=5)", "Random Forest\n(sklearn)"]
    accs = [base["accuracy"] * 100, knn_acc, rf_test["accuracy"] * 100]
    f1s = [base["f1_macro"], knn_f1, rf_test["f1_macro"]]

    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.5))

    ejes[0].bar(modelos, accs, color=["gray", "tab:blue", "tab:green"])
    ejes[0].set_ylabel("Accuracy (%)")
    ejes[0].set_title("Accuracy en el conjunto de prueba")
    for i, v in enumerate(accs):
        ejes[0].text(i, v + 1, "{:.2f}%".format(v), ha="center", fontsize=9)
    ejes[0].set_ylim(0, 105)
    ejes[0].grid(axis="y", alpha=0.3)

    ejes[1].bar(modelos, f1s, color=["gray", "tab:blue", "tab:green"])
    ejes[1].set_ylabel("F1 macro")
    ejes[1].set_title("F1 macro en el conjunto de prueba")
    for i, v in enumerate(f1s):
        ejes[1].text(i, v + 0.02, "{:.3f}".format(v), ha="center", fontsize=9)
    ejes[1].set_ylim(0, 1.05)
    ejes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    guardar("comparacion_modelos.png")


# ==============================================================================
# PROGRAMA PRINCIPAL
# ==============================================================================

# Resultados del KNN sin framework (entrega anterior, misma particion 75/25).
KNN_ACCURACY = 81.51
KNN_F1_MACRO = 0.739


def main():
    # ---------------------------------------------------------------- datos --
    X, y = cargar_datos()

    # stratify reparte cada clase en la misma proporcion entre train y test.
    # Sin esto las clases chicas (DEMO con 229 casos, REFACC con 259) podrian
    # quedar casi todas de un solo lado por puro azar.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=PORCENTAJE_TEST, random_state=SEMILLA, stratify=y)

    clases = sorted(y.unique())

    print("=" * 86)
    print(" DATOS")
    print("=" * 86)
    print(" Archivo          : interest_ledger.csv")
    print(" Variables        : " + ", ".join(COLUMNAS))
    print(" Registros totales: {}".format(len(X)))
    print(" Entrenamiento    : {}  ({:.0%})".format(len(X_train), 1 - PORCENTAJE_TEST))
    print(" Prueba           : {}  ({:.0%})".format(len(X_test), PORCENTAJE_TEST))
    print("\n Registros por clase (el dataset esta muy desbalanceado):")
    for clase in clases:
        n_tr = int((y_train == clase).sum())
        n_te = int((y_test == clase).sum())
        print("   {:<12} total {:>5} ({:>5.1f}%)   train {:>5}   test {:>5}".format(
            clase, n_tr + n_te, 100 * (n_tr + n_te) / len(y), n_tr, n_te))

    # ------------------------------------------------------------ linea base --
    base = linea_base(X_train, y_train, X_test, y_test)

    # ------------------------------------------------- seleccion de modelo ---
    resultados_cv, mejor_config = comparar_configuraciones(X_train, y_train)

    # ------------------------------------------------------- modelo final ----
    print("\n" + "=" * 86)
    print(" ENTRENAMIENTO DEL MODELO FINAL")
    print("=" * 86)
    print(" Configuracion:", mejor_config)

    modelo = construir_modelo(mejor_config)
    modelo.fit(X_train, y_train)
    print(" Bosque entrenado con {} arboles.".format(len(modelo.estimators_)))

    profundidades = [arbol.get_depth() for arbol in modelo.estimators_]
    hojas = [arbol.get_n_leaves() for arbol in modelo.estimators_]
    print(" Profundidad media de los arboles: {:.1f}  (max {})".format(
        np.mean(profundidades), max(profundidades)))
    print(" Hojas por arbol (promedio)      : {:.0f}".format(np.mean(hojas)))

    # ------------------------------------------------------------ metricas ---
    print("\n" + "=" * 86)
    print(" EVALUACION")
    print("=" * 86)
    entrenamiento = evaluar(modelo, X_train, y_train, "CONJUNTO DE ENTRENAMIENTO",
                            clases, detalle=False)
    prueba = evaluar(modelo, X_test, y_test, "CONJUNTO DE PRUEBA", clases, detalle=True)

    print(" Brecha entrenamiento - prueba en F1 macro: {:.4f}".format(
        entrenamiento["f1_macro"] - prueba["f1_macro"]))
    print(" (esta brecha se analiza a fondo en la entrega M2_Analisis)")

    # --------------------------------------------------------- importancias --
    print("\n Importancia de cada variable:")
    orden = np.argsort(modelo.feature_importances_)[::-1]
    for i in orden:
        print("   {:<20} {:.4f}".format(COLUMNAS[i], modelo.feature_importances_[i]))

    # --------------------------------------------------------- predicciones --
    predecir_casos_nuevos(modelo)

    # ------------------------------------------------------------- graficas --
    print("=" * 86)
    print(" GRAFICAS")
    print("=" * 86)
    graficar_matriz(clases, prueba["matriz"],
                    "Matriz de confusion - Random Forest (prueba)", "matriz_confusion.png")
    graficar_importancias(modelo)
    graficar_configuraciones(resultados_cv)
    graficar_comparacion(base, prueba, KNN_F1_MACRO, KNN_ACCURACY)

    # -------------------------------------------------------------- resumen --
    print("\n" + "=" * 86)
    print(" RESUMEN")
    print("=" * 86)
    print(" {:<28} {:>12} {:>12}".format("modelo", "accuracy", "F1 macro"))
    print(" {:<28} {:>11.2f}% {:>12.4f}".format("Linea base (mayoritaria)",
                                                base["accuracy"] * 100, base["f1_macro"]))
    print(" {:<28} {:>11.2f}% {:>12.4f}".format("KNN sin framework (k=5)",
                                                KNN_ACCURACY, KNN_F1_MACRO))
    print(" {:<28} {:>11.2f}% {:>12.4f}".format("Random Forest (sklearn)",
                                                prueba["accuracy"] * 100, prueba["f1_macro"]))
    print("\n Listo.")


if __name__ == "__main__":
    main()

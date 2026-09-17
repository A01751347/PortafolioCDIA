"""
================================================================================
 ANALISIS Y REPORTE SOBRE EL DESEMPEÑO DEL MODELO
================================================================================

 Modulo 2 - Portafolio de Analisis
 Santiago Serrano Montalvo - A01751347

 OBJETIVO
 --------
 Tomar una de las dos implementaciones del portafolio (se elige el RANDOM
 FOREST de la entrega con framework) y analizar a fondo su desempeño:

   1. Separar y evaluar con conjunto de ENTRENAMIENTO, VALIDACION y PRUEBA.
   2. Diagnosticar el grado de SESGO (bias):      bajo / medio / alto
   3. Diagnosticar el grado de VARIANZA:          bajo / medio / alto
   4. Diagnosticar el nivel de AJUSTE:            underfitting / fitting / overfitting
   5. Aplicar tecnicas de REGULARIZACION y documentar la mejora.

 POR QUE TRES CONJUNTOS Y NO DOS
 -------------------------------
   ENTRENAMIENTO (60%)  el modelo ajusta sus parametros aqui.
   VALIDACION    (20%)  se usa para ELEGIR hiperparametros y para diagnosticar
                        sesgo y varianza. El modelo nunca se entrena con estos
                        datos, pero si se toman decisiones mirandolos, asi que
                        su resultado queda optimista.
   PRUEBA        (20%)  se abre UNA SOLA VEZ al final. Como ninguna decision se
                        tomo mirandolo, es la unica estimacion honesta del
                        desempeño con datos nuevos.

 Si solo hubiera train/test y se eligieran los hiperparametros mirando test,
 el resultado de test dejaria de ser una estimacion valida: se estaria
 sobreajustando al conjunto de prueba.

 EJECUCION
 ---------
       python3 main.py

 Genera todas las graficas del reporte dentro de la carpeta figuras/.

================================================================================
"""

import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report


# ==============================================================================
# CONFIGURACION
# ==============================================================================

CARPETA = os.path.dirname(os.path.abspath(__file__))
RUTA_CSV = os.path.join(CARPETA, "interest_ledger.csv")
CARPETA_FIGURAS = os.path.join(CARPETA, "figuras")

COLUMNAS = ["principal_amount", "interest_amount", "tasa", "antiguedad", "mes"]
SEMILLA = 42

# 60 / 20 / 20
FRACCION_PRUEBA = 0.20
FRACCION_VALIDACION = 0.25      # 25% del 80% restante = 20% del total

# ------------------------------------------------------------------------------
# UMBRALES DEL DIAGNOSTICO
# ------------------------------------------------------------------------------
# El diagnostico de sesgo y varianza se hace con reglas explicitas y no "a ojo",
# para que sea reproducible y auditable.
#
# SESGO: se mide con el error del modelo SOBRE SU PROPIO ENTRENAMIENTO. Un
#        modelo con sesgo alto es tan rigido que ni siquiera puede ajustar los
#        datos que ya vio.
# VARIANZA: se mide con la BRECHA entre entrenamiento y validacion. Si el
#        modelo va excelente en lo que memorizo y mal en datos nuevos, su
#        prediccion depende demasiado de la muestra concreta que le toco.
# ------------------------------------------------------------------------------

UMBRAL_SESGO_BAJO = 0.95        # F1 macro en entrenamiento por encima de esto -> sesgo bajo
UMBRAL_SESGO_MEDIO = 0.80       # entre 0.80 y 0.95 -> sesgo medio; abajo -> sesgo alto

UMBRAL_VARIANZA_BAJA = 0.03     # brecha train-val menor a esto -> varianza baja
UMBRAL_VARIANZA_MEDIA = 0.10    # entre 0.03 y 0.10 -> media; arriba -> alta


# ==============================================================================
# UTILIDADES
# ==============================================================================

def guardar(nombre):
    if not os.path.isdir(CARPETA_FIGURAS):
        os.makedirs(CARPETA_FIGURAS)
    ruta = os.path.join(CARPETA_FIGURAS, nombre)
    plt.savefig(ruta, dpi=140)
    plt.close()
    print("   figura guardada: figuras/" + nombre)


def titulo(texto):
    print("\n" + "=" * 86)
    print(" " + texto)
    print("=" * 86)


# ==============================================================================
# 1. DATOS Y PARTICION EN TRES CONJUNTOS
# ==============================================================================

def cargar_datos(ruta=RUTA_CSV):
    """Lee el CSV y deriva las variables. Identico a la entrega con framework."""
    datos = pd.read_csv(ruta)

    datos["collateral"] = (datos["collateral"]
                           .str.replace("\xa0", " ", regex=False)
                           .str.strip().str.upper())

    datos["year"] = datos["month_date"].str[:4].astype(int)
    datos["mes"] = datos["month_date"].str[5:7].astype(int)

    datos["tasa"] = 0.0
    con_principal = datos["principal_amount"] > 0
    datos.loc[con_principal, "tasa"] = (datos.loc[con_principal, "interest_amount"] /
                                        datos.loc[con_principal, "principal_amount"])

    datos["antiguedad"] = datos["year"] - datos["year_model"]

    return datos[COLUMNAS], datos["collateral"], datos["vin"]


def partir_en_tres(X, y):
    """Divide en 60% entrenamiento / 20% validacion / 20% prueba, estratificado.

    Se hace en dos pasos porque train_test_split solo parte en dos:
      paso 1:  todo        -> (80% temporal, 20% prueba)
      paso 2:  80% temporal -> (60% entrenamiento, 20% validacion)

    `stratify` conserva la proporcion de cada clase en los tres conjuntos, lo
    cual es indispensable aqui porque DEMO y REFACC representan ~2% cada una.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=FRACCION_PRUEBA, random_state=SEMILLA, stratify=y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=FRACCION_VALIDACION, random_state=SEMILLA, stratify=y_temp)

    return X_train, y_train, X_val, y_val, X_test, y_test


# ==============================================================================
# 2. MEDICION
# ==============================================================================

def medir(modelo, X, y):
    """Devuelve accuracy y F1 macro del modelo sobre un conjunto."""
    y_pred = modelo.predict(X)
    return {"accuracy": accuracy_score(y, y_pred),
            "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0)}


def construir(**kwargs):
    """Random Forest con los valores por defecto del proyecto mas los cambios dados."""
    parametros = dict(n_estimators=300, max_depth=None, min_samples_leaf=1,
                      min_samples_split=2, max_features="sqrt", ccp_alpha=0.0,
                      max_samples=None, class_weight=None,
                      random_state=SEMILLA, n_jobs=-1)
    parametros.update(kwargs)
    return RandomForestClassifier(**parametros)


def diagnosticar(f1_train, f1_val):
    """Aplica las reglas de la cabecera y devuelve el diagnostico en texto."""
    brecha = f1_train - f1_val

    if f1_train >= UMBRAL_SESGO_BAJO:
        sesgo = "BAJO"
    elif f1_train >= UMBRAL_SESGO_MEDIO:
        sesgo = "MEDIO"
    else:
        sesgo = "ALTO"

    if brecha < UMBRAL_VARIANZA_BAJA:
        varianza = "BAJA"
    elif brecha < UMBRAL_VARIANZA_MEDIA:
        varianza = "MEDIA"
    else:
        varianza = "ALTA"

    # El nivel de ajuste se deduce de la combinacion de los dos anteriores.
    if sesgo == "ALTO":
        ajuste = "UNDERFITTING"
    elif varianza == "ALTA":
        ajuste = "OVERFITTING"
    elif varianza == "MEDIA":
        ajuste = "FITTING con sobreajuste leve"
    else:
        ajuste = "FITTING"

    return {"sesgo": sesgo, "varianza": varianza, "ajuste": ajuste, "brecha": brecha}


def imprimir_diagnostico(nombre, f1_train, f1_val, d):
    print("\n   {}".format(nombre))
    print("     F1 macro entrenamiento : {:.4f}".format(f1_train))
    print("     F1 macro validacion    : {:.4f}".format(f1_val))
    print("     Brecha (train - val)   : {:.4f}".format(d["brecha"]))
    print("     -> SESGO    : {}   (regla: F1 train >= {} bajo, >= {} medio)".format(
        d["sesgo"], UMBRAL_SESGO_BAJO, UMBRAL_SESGO_MEDIO))
    print("     -> VARIANZA : {}   (regla: brecha < {} baja, < {} media)".format(
        d["varianza"], UMBRAL_VARIANZA_BAJA, UMBRAL_VARIANZA_MEDIA))
    print("     -> AJUSTE   : {}".format(d["ajuste"]))


# ==============================================================================
# 3. CURVA DE APRENDIZAJE
# ==============================================================================

def curva_de_aprendizaje(X_train, y_train, X_val, y_val, X_test, y_test, parametros, etiqueta):
    """Entrena el mismo modelo con fracciones crecientes del entrenamiento.

    COMO SE LEE:
      * Si la curva de entrenamiento se queda BAJA -> el modelo no tiene
        capacidad suficiente: SESGO ALTO.
      * Si la curva de entrenamiento va muy ALTA y la de validacion queda muy
        por debajo, con una separacion que NO se cierra -> VARIANZA ALTA.
      * Si las dos convergen juntas y altas -> buen ajuste.
      * Si la curva de validacion sigue SUBIENDO al final, conseguir mas datos
        todavia ayudaria. Si ya se aplano, mas datos no resuelven nada.
    """
    fracciones = [0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.0]

    f1_train, f1_val, f1_test, tamanos = [], [], [], []

    print("\n   {:>8} {:>14} {:>14} {:>14} {:>10}".format(
        "n train", "F1 train", "F1 validacion", "F1 prueba", "brecha"))

    for fraccion in fracciones:
        n = int(len(X_train) * fraccion)

        if fraccion < 1.0:
            # Submuestra estratificada: conserva la proporcion de clases.
            X_sub, _, y_sub, _ = train_test_split(
                X_train, y_train, train_size=n, random_state=SEMILLA, stratify=y_train)
        else:
            X_sub, y_sub = X_train, y_train

        modelo = construir(**parametros)
        modelo.fit(X_sub, y_sub)

        m_tr = medir(modelo, X_sub, y_sub)
        m_va = medir(modelo, X_val, y_val)
        m_te = medir(modelo, X_test, y_test)

        tamanos.append(len(X_sub))
        f1_train.append(m_tr["f1_macro"])
        f1_val.append(m_va["f1_macro"])
        f1_test.append(m_te["f1_macro"])

        print("   {:>8} {:>14.4f} {:>14.4f} {:>14.4f} {:>10.4f}".format(
            len(X_sub), m_tr["f1_macro"], m_va["f1_macro"], m_te["f1_macro"],
            m_tr["f1_macro"] - m_va["f1_macro"]))

    return {"tamanos": tamanos, "train": f1_train, "val": f1_val,
            "test": f1_test, "etiqueta": etiqueta}


def graficar_curvas_aprendizaje(curvas, nombre):
    """Dibuja una o dos curvas de aprendizaje lado a lado."""
    fig, ejes = plt.subplots(1, len(curvas), figsize=(6.2 * len(curvas), 5), squeeze=False)

    for eje, curva in zip(ejes[0], curvas):
        eje.plot(curva["tamanos"], curva["train"], marker="o", color="tab:blue",
                 label="Entrenamiento")
        eje.plot(curva["tamanos"], curva["val"], marker="s", color="tab:orange",
                 label="Validacion")
        eje.plot(curva["tamanos"], curva["test"], marker="^", color="tab:green",
                 linestyle="--", label="Prueba")

        eje.fill_between(curva["tamanos"], curva["train"], curva["val"],
                         color="tab:red", alpha=0.12)

        brecha = curva["train"][-1] - curva["val"][-1]
        eje.annotate("brecha final = {:.3f}".format(brecha),
                     xy=(curva["tamanos"][-1], (curva["train"][-1] + curva["val"][-1]) / 2),
                     xytext=(-150, 0), textcoords="offset points",
                     fontsize=9, color="tab:red",
                     arrowprops=dict(arrowstyle="->", color="tab:red"))

        eje.set_xlabel("Registros de entrenamiento usados")
        eje.set_ylabel("F1 macro")
        eje.set_title(curva["etiqueta"])
        eje.set_ylim(0.3, 1.02)
        eje.grid(alpha=0.3)
        eje.legend(loc="lower right")

    plt.tight_layout()
    guardar(nombre)


# ==============================================================================
# 4. CURVAS DE VALIDACION (efecto de cada hiperparametro)
# ==============================================================================

def curva_de_validacion(X_train, y_train, X_val, y_val, nombre_parametro, valores, base=None):
    """Mide F1 de entrenamiento y validacion variando UN hiperparametro.

    Es la herramienta directa para ver el nivel de ajuste: a la izquierda del
    grafico (modelo rigido) domina el sesgo, a la derecha (modelo flexible)
    domina la varianza, y el mejor punto de validacion marca el equilibrio.
    """
    base = base or {}
    f1_train, f1_val = [], []

    print("\n   Variando {}:".format(nombre_parametro))
    print("   {:>14} {:>14} {:>14} {:>10}".format("valor", "F1 train", "F1 val", "brecha"))

    for valor in valores:
        parametros = dict(base)
        parametros[nombre_parametro] = valor

        modelo = construir(**parametros)
        modelo.fit(X_train, y_train)

        tr = medir(modelo, X_train, y_train)["f1_macro"]
        va = medir(modelo, X_val, y_val)["f1_macro"]

        f1_train.append(tr)
        f1_val.append(va)

        print("   {:>14} {:>14.4f} {:>14.4f} {:>10.4f}".format(str(valor), tr, va, tr - va))

    return {"parametro": nombre_parametro, "valores": valores,
            "train": f1_train, "val": f1_val}


def graficar_curvas_validacion(curvas, nombre):
    """Rejilla con la curva de validacion de cada hiperparametro."""
    columnas = 2
    filas = (len(curvas) + columnas - 1) // columnas
    fig, ejes = plt.subplots(filas, columnas, figsize=(6.2 * columnas, 4.6 * filas))
    ejes = np.array(ejes).reshape(-1)

    for eje, curva in zip(ejes, curvas):
        # Los valores pueden no ser numericos (None, "sqrt"): se usa el indice.
        x = list(range(len(curva["valores"])))

        eje.plot(x, curva["train"], marker="o", color="tab:blue", label="Entrenamiento")
        eje.plot(x, curva["val"], marker="s", color="tab:orange", label="Validacion")
        eje.fill_between(x, curva["train"], curva["val"], color="tab:red", alpha=0.12)

        mejor = int(np.argmax(curva["val"]))
        eje.axvline(mejor, color="tab:green", linestyle="--", alpha=0.8)
        eje.plot([mejor], [curva["val"][mejor]], marker="*", markersize=16, color="tab:green")

        eje.set_xticks(x)
        eje.set_xticklabels([str(v) for v in curva["valores"]], rotation=25, fontsize=8)
        eje.set_xlabel(curva["parametro"] + "   (menos flexible  ->  mas flexible)")
        eje.set_ylabel("F1 macro")
        eje.set_title("Curva de validacion: " + curva["parametro"])
        eje.set_ylim(0.3, 1.02)
        eje.grid(alpha=0.3)
        eje.legend(loc="lower right", fontsize=8)

    for eje in ejes[len(curvas):]:
        eje.axis("off")

    plt.tight_layout()
    guardar(nombre)


# ==============================================================================
# 5. BUSQUEDA DE LA MEJOR COMBINACION DE REGULARIZACION
# ==============================================================================

# Tecnicas de regularizacion disponibles en un Random Forest y que hace cada una:
#
#   max_depth         corta el arbol a una profundidad maxima. Impide que siga
#                     partiendo hasta aislar casos individuales.
#   min_samples_leaf  obliga a que cada hoja conserve un minimo de registros.
#                     Una hoja con 1 registro es memorizacion pura.
#   max_features      cuantas variables puede mirar cada corte. Bajarlo
#                     decorrelaciona mas los arboles y reduce la varianza del
#                     promedio (es la regularizacion caracteristica del bosque).
#   ccp_alpha         poda por costo-complejidad: elimina las ramas cuya mejora
#                     de impureza no compensa el costo de tener mas hojas.
#                     Es el analogo de una penalizacion L1 sobre el arbol.
#   max_samples       fraccion de registros que recibe cada arbol en su muestra
#                     bootstrap. Bajarlo aumenta la diversidad del ensamble.
#   n_estimators      mas arboles promedian mas y reducen varianza sin subir sesgo.
#   class_weight      reponderacion para compensar el desbalance de clases.

REJILLA = []
for max_features in [2, 3, 4, 5]:
    for max_depth in [None, 25, 16, 12]:
        for min_samples_leaf in [1, 2, 4]:
            for ccp_alpha in [0.0, 0.0002]:
                for class_weight in [None, "balanced_subsample"]:
                    REJILLA.append({"max_depth": max_depth,
                                    "min_samples_leaf": min_samples_leaf,
                                    "ccp_alpha": ccp_alpha,
                                    "class_weight": class_weight,
                                    "max_features": max_features,
                                    "n_estimators": 300})

# Tope de brecha train-val para la variante "conservadora": entre las
# configuraciones cuya varianza ya es MEDIA o BAJA, se toma la de mejor
# validacion. Sirve para mostrar el costo real de exigir poca varianza.
TOPE_BRECHA_CONSERVADORA = UMBRAL_VARIANZA_MEDIA


def buscar_regularizacion(X_train, y_train, X_val, y_val):
    """Prueba la rejilla completa y devuelve todas las combinaciones evaluadas.

    El criterio de seleccion es el F1 MACRO EN VALIDACION. Se elige el F1 macro
    y no el accuracy porque el dataset esta muy desbalanceado: la clase NUEVO
    es el 42.5% de los datos y DEMO apenas el 2.2%, asi que un modelo que
    abandonara por completo a DEMO perderia ~2 puntos de accuracy pero hundiria
    el F1 macro, que promedia el F1 de las 5 clases con el mismo peso.
    """
    resultados = []
    inicio = time.time()

    print("\n   Evaluando {} combinaciones sobre el conjunto de validacion...".format(len(REJILLA)))

    for i, parametros in enumerate(REJILLA):
        modelo = construir(**parametros)
        modelo.fit(X_train, y_train)

        tr = medir(modelo, X_train, y_train)["f1_macro"]
        va = medir(modelo, X_val, y_val)["f1_macro"]

        resultados.append({"parametros": parametros, "train": tr, "val": va,
                           "brecha": tr - va})

        if (i + 1) % 15 == 0:
            print("      {}/{} combinaciones ({:.0f}s)".format(
                i + 1, len(REJILLA), time.time() - inicio))

    resultados.sort(key=lambda r: r["val"], reverse=True)

    print("\n   Las 12 mejores combinaciones por F1 macro en VALIDACION:")
    encabezado = "   {:>9} {:>10} {:>9} {:>10} {:>20} {:>9} {:>9} {:>8}"
    print(encabezado.format("max_feat", "max_depth", "min_leaf", "ccp_alpha",
                            "class_weight", "F1 train", "F1 val", "brecha"))
    for r in resultados[:12]:
        p = r["parametros"]
        print("   {:>9} {:>10} {:>9} {:>10} {:>20} {:>9.4f} {:>9.4f} {:>8.4f}".format(
            str(p["max_features"]), str(p["max_depth"]), p["min_samples_leaf"],
            p["ccp_alpha"], str(p["class_weight"]), r["train"], r["val"], r["brecha"]))

    # Variante conservadora: la mejor validacion entre las que NO tienen
    # varianza alta. Documenta cuanto desempeño cuesta cerrar la brecha.
    conservadoras = [r for r in resultados if r["brecha"] <= TOPE_BRECHA_CONSERVADORA]
    if conservadoras:
        mejor_cons = max(conservadoras, key=lambda r: r["val"])
        print("\n   Mejor configuracion con brecha <= {} (varianza no alta):".format(
            TOPE_BRECHA_CONSERVADORA))
        p = mejor_cons["parametros"]
        print("      max_features={} max_depth={} min_leaf={} ccp_alpha={} pesos={}".format(
            p["max_features"], p["max_depth"], p["min_samples_leaf"],
            p["ccp_alpha"], p["class_weight"]))
        print("      F1 train {:.4f} | F1 val {:.4f} | brecha {:.4f}".format(
            mejor_cons["train"], mejor_cons["val"], mejor_cons["brecha"]))
    else:
        mejor_cons = None

    return resultados, mejor_cons


def graficar_frontera(resultados, nombre):
    """Dispersa todas las combinaciones: brecha (varianza) contra F1 de validacion."""
    brechas = [r["brecha"] for r in resultados]
    vals = [r["val"] for r in resultados]

    plt.figure(figsize=(8.5, 5.5))
    plt.scatter(brechas, vals, c=[r["train"] for r in resultados],
                cmap="viridis", s=45, alpha=0.85, edgecolor="k", linewidth=0.3)
    plt.colorbar(label="F1 macro en entrenamiento")

    mejor = resultados[0]
    plt.scatter([mejor["brecha"]], [mejor["val"]], marker="*", s=420,
                color="red", edgecolor="k", zorder=5, label="Configuracion elegida")

    plt.axvline(UMBRAL_VARIANZA_BAJA, color="gray", linestyle=":", alpha=0.7)
    plt.axvline(UMBRAL_VARIANZA_MEDIA, color="gray", linestyle="--", alpha=0.7)
    plt.text(UMBRAL_VARIANZA_BAJA, min(vals), " varianza\n baja", fontsize=8, color="gray")
    plt.text(UMBRAL_VARIANZA_MEDIA, min(vals), " varianza\n alta ->", fontsize=8, color="gray")

    plt.xlabel("Brecha entrenamiento - validacion   (proxy de la varianza)")
    plt.ylabel("F1 macro en validacion")
    plt.title("Las {} configuraciones probadas: varianza contra desempeño".format(len(resultados)))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    guardar(nombre)


# ==============================================================================
# 6. GRAFICAS DE COMPARACION ANTES / DESPUES
# ==============================================================================

def graficar_antes_despues(antes, despues, nombre):
    """Barras comparativas del modelo base contra el modelo regularizado."""
    conjuntos = ["Entrenamiento", "Validacion", "Prueba"]
    v_antes = [antes["train"]["f1_macro"], antes["val"]["f1_macro"], antes["test"]["f1_macro"]]
    v_despues = [despues["train"]["f1_macro"], despues["val"]["f1_macro"], despues["test"]["f1_macro"]]

    x = np.arange(len(conjuntos))
    ancho = 0.36

    fig, ejes = plt.subplots(1, 2, figsize=(13, 5))

    # --- panel izquierdo: F1 macro en los tres conjuntos ---
    b1 = ejes[0].bar(x - ancho / 2, v_antes, ancho, label="ANTES (sin regularizar)",
                     color="tab:red", alpha=0.85)
    b2 = ejes[0].bar(x + ancho / 2, v_despues, ancho, label="DESPUES (regularizado)",
                     color="tab:green", alpha=0.85)
    for barras in (b1, b2):
        for barra in barras:
            ejes[0].text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.012,
                         "{:.3f}".format(barra.get_height()), ha="center", fontsize=9)
    ejes[0].set_xticks(x)
    ejes[0].set_xticklabels(conjuntos)
    ejes[0].set_ylabel("F1 macro")
    ejes[0].set_ylim(0, 1.12)
    ejes[0].set_title("Desempeño antes y despues de regularizar")
    ejes[0].legend()
    ejes[0].grid(axis="y", alpha=0.3)

    # --- panel derecho: la brecha, que es lo que mide la varianza ---
    brechas = [antes["train"]["f1_macro"] - antes["val"]["f1_macro"],
               despues["train"]["f1_macro"] - despues["val"]["f1_macro"]]
    barras = ejes[1].bar(["ANTES", "DESPUES"], brechas, color=["tab:red", "tab:green"],
                         alpha=0.85, width=0.5)
    for barra in barras:
        ejes[1].text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.004,
                     "{:.4f}".format(barra.get_height()), ha="center", fontsize=10)
    ejes[1].axhline(UMBRAL_VARIANZA_MEDIA, color="darkred", linestyle="--",
                    label="umbral varianza ALTA ({})".format(UMBRAL_VARIANZA_MEDIA))
    ejes[1].axhline(UMBRAL_VARIANZA_BAJA, color="darkgreen", linestyle=":",
                    label="umbral varianza BAJA ({})".format(UMBRAL_VARIANZA_BAJA))
    ejes[1].set_ylabel("Brecha entrenamiento - validacion")
    ejes[1].set_title("Varianza antes y despues de regularizar")
    ejes[1].legend(fontsize=8)
    ejes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    guardar(nombre)


def graficar_matrices_lado_a_lado(clases, matriz_antes, matriz_despues, nombre):
    """Las dos matrices de confusion de prueba, normalizadas por fila."""
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5.8))

    for eje, matriz, subtitulo in [(ejes[0], matriz_antes, "ANTES - sin regularizar"),
                                   (ejes[1], matriz_despues, "DESPUES - regularizado")]:
        matriz = np.asarray(matriz, dtype=float)
        norm = matriz / np.clip(matriz.sum(axis=1, keepdims=True), 1, None)

        imagen = eje.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        eje.set_title(subtitulo)
        eje.set_xlabel("Prediccion")
        eje.set_ylabel("Clase real")
        eje.set_xticks(range(len(clases)))
        eje.set_xticklabels(clases, rotation=45, ha="right")
        eje.set_yticks(range(len(clases)))
        eje.set_yticklabels(clases)

        for i in range(len(clases)):
            for j in range(len(clases)):
                color = "white" if norm[i, j] > 0.5 else "black"
                eje.text(j, i, "{:.0f}\n{:.0%}".format(matriz[i, j], norm[i, j]),
                         ha="center", va="center", fontsize=8, color=color)

        fig.colorbar(imagen, ax=eje, fraction=0.046)

    plt.suptitle("Matriz de confusion en el conjunto de PRUEBA", fontsize=13)
    plt.tight_layout()
    guardar(nombre)


def graficar_f1_por_clase(clases, f1_antes, f1_despues, nombre):
    """F1 de cada clase antes y despues, para ver a quien beneficio la regularizacion."""
    x = np.arange(len(clases))
    ancho = 0.36

    plt.figure(figsize=(9.5, 5))
    plt.bar(x - ancho / 2, f1_antes, ancho, label="ANTES", color="tab:red", alpha=0.85)
    plt.bar(x + ancho / 2, f1_despues, ancho, label="DESPUES", color="tab:green", alpha=0.85)

    for i in range(len(clases)):
        delta = f1_despues[i] - f1_antes[i]
        plt.text(x[i], max(f1_antes[i], f1_despues[i]) + 0.03,
                 "{:+.3f}".format(delta), ha="center", fontsize=9,
                 color="darkgreen" if delta >= 0 else "darkred")

    plt.xticks(x, clases, rotation=30, ha="right")
    plt.ylabel("F1 de la clase (conjunto de prueba)")
    plt.ylim(0, 1.15)
    plt.title("Efecto de la regularizacion clase por clase")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    guardar(nombre)


def graficar_fuga(resultados_fuga, nombre):
    """Compara la particion aleatoria contra la particion agrupada por vehiculo."""
    etiquetas = ["Particion ALEATORIA\n(80% de los VIN de prueba\ntambien estan en train)",
                 "Particion POR VEHICULO\n(ningun VIN compartido)"]
    valores = [resultados_fuga["aleatoria"], resultados_fuga["agrupada"]]

    plt.figure(figsize=(8, 5))
    barras = plt.bar(etiquetas, valores, color=["tab:orange", "tab:blue"], width=0.5)
    for barra in barras:
        plt.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.012,
                 "{:.3f}".format(barra.get_height()), ha="center", fontsize=11)

    caida = resultados_fuga["aleatoria"] - resultados_fuga["agrupada"]
    plt.title("F1 macro en prueba segun como se parten los datos\n"
              "(la particion aleatoria sobreestima el desempeño en {:.3f})".format(caida))
    plt.ylabel("F1 macro")
    plt.ylim(0, max(valores) * 1.25)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    guardar(nombre)


# ==============================================================================
# 7. PREDICCIONES CON EL MODELO FINAL
# ==============================================================================

def predecir_casos_nuevos(modelo):
    casos = pd.DataFrame([
        [350000.0,  1500.0, 1500.0 / 350000.0,  1, 2],
        [1200000.0, 9000.0, 9000.0 / 1200000.0, 0, 5],
        [90000.0,    400.0,  400.0 / 90000.0,   4, 8],
    ], columns=COLUMNAS)

    descripciones = ["Credito mediano, vehiculo de un año",
                     "Credito muy grande, vehiculo del año",
                     "Credito chico, vehiculo con 4 años"]

    predicciones = modelo.predict(casos)
    probabilidades = modelo.predict_proba(casos)

    titulo("PREDICCIONES CON EL MODELO FINAL REGULARIZADO")

    for i in range(len(casos)):
        print("\n {}".format(descripciones[i]))
        print("   principal ${:>12,.2f} | interes ${:>9,.2f} | antiguedad {} | mes {}".format(
            casos["principal_amount"][i], casos["interest_amount"][i],
            casos["antiguedad"][i], casos["mes"][i]))
        print("   --> PREDICCION: {}".format(predicciones[i]))

        orden = np.argsort(probabilidades[i])[::-1]
        reparto = "  ".join("{} {:.1f}%".format(modelo.classes_[j], probabilidades[i][j] * 100)
                            for j in orden if probabilidades[i][j] > 0.005)
        print("       confianza: {}".format(reparto))


# ==============================================================================
# PROGRAMA PRINCIPAL
# ==============================================================================

def main():
    # ==========================================================================
    titulo("PARTE 1 - PARTICION EN ENTRENAMIENTO / VALIDACION / PRUEBA")
    # ==========================================================================

    X, y, vin = cargar_datos()
    X_train, y_train, X_val, y_val, X_test, y_test = partir_en_tres(X, y)
    clases = sorted(y.unique())

    print(" Dataset          : interest_ledger.csv ({} registros)".format(len(X)))
    print(" Modelo analizado : Random Forest (entrega con framework)")
    print(" Metrica principal: F1 macro")
    print("\n {:<16} {:>8} {:>10}".format("conjunto", "n", "% del total"))
    for nombre, conjunto in [("Entrenamiento", y_train), ("Validacion", y_val), ("Prueba", y_test)]:
        print(" {:<16} {:>8} {:>9.0f}%".format(nombre, len(conjunto), 100 * len(conjunto) / len(y)))

    print("\n Proporcion de cada clase en los tres conjuntos (stratify la conserva):")
    print(" {:<12} {:>14} {:>14} {:>14}".format("clase", "entrenamiento", "validacion", "prueba"))
    for clase in clases:
        print(" {:<12} {:>13.1f}% {:>13.1f}% {:>13.1f}%".format(
            clase,
            100 * (y_train == clase).mean(),
            100 * (y_val == clase).mean(),
            100 * (y_test == clase).mean()))

    # ==========================================================================
    titulo("PARTE 2 - MODELO BASE SIN REGULARIZAR")
    # ==========================================================================

    parametros_base = {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1,
                       "max_features": "sqrt", "ccp_alpha": 0.0, "class_weight": None}

    print(" Configuracion de partida (arboles sin ninguna restriccion):")
    for clave, valor in parametros_base.items():
        print("   {:<20} {}".format(clave, valor))

    modelo_base = construir(**parametros_base)
    modelo_base.fit(X_train, y_train)

    profundidades = [a.get_depth() for a in modelo_base.estimators_]
    hojas = [a.get_n_leaves() for a in modelo_base.estimators_]
    print("\n Estructura del bosque resultante:")
    print("   profundidad media {:.1f} (maxima {})".format(np.mean(profundidades), max(profundidades)))
    print("   hojas por arbol   {:.0f}  sobre {} registros de entrenamiento".format(
        np.mean(hojas), len(X_train)))
    print("   -> {:.1f} registros por hoja en promedio: el bosque esta memorizando".format(
        len(X_train) / np.mean(hojas)))

    antes = {"train": medir(modelo_base, X_train, y_train),
             "val": medir(modelo_base, X_val, y_val),
             "test": medir(modelo_base, X_test, y_test)}

    print("\n {:<16} {:>12} {:>12}".format("conjunto", "accuracy", "F1 macro"))
    for nombre in ["train", "val", "test"]:
        print(" {:<16} {:>11.2f}% {:>12.4f}".format(
            nombre, antes[nombre]["accuracy"] * 100, antes[nombre]["f1_macro"]))

    d_antes = diagnosticar(antes["train"]["f1_macro"], antes["val"]["f1_macro"])
    imprimir_diagnostico("DIAGNOSTICO DEL MODELO BASE", antes["train"]["f1_macro"],
                         antes["val"]["f1_macro"], d_antes)

    # ==========================================================================
    titulo("PARTE 3 - CURVA DE APRENDIZAJE DEL MODELO BASE")
    # ==========================================================================

    curva_base = curva_de_aprendizaje(X_train, y_train, X_val, y_val, X_test, y_test,
                                      parametros_base, "ANTES: Random Forest sin regularizar")

    # ==========================================================================
    titulo("PARTE 4 - CURVAS DE VALIDACION: EFECTO DE CADA HIPERPARAMETRO")
    # ==========================================================================

    print(" Cada curva varia un solo hiperparametro dejando los demas en su valor base.")
    print(" El eje x va del modelo MENOS flexible (izquierda) al MAS flexible (derecha).")

    curvas_validacion = [
        curva_de_validacion(X_train, y_train, X_val, y_val,
                            "max_depth", [2, 4, 6, 8, 12, 16, 20, 25, None], parametros_base),
        curva_de_validacion(X_train, y_train, X_val, y_val,
                            "min_samples_leaf", [50, 25, 10, 5, 3, 2, 1], parametros_base),
        curva_de_validacion(X_train, y_train, X_val, y_val,
                            "ccp_alpha", [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0], parametros_base),
        curva_de_validacion(X_train, y_train, X_val, y_val,
                            "max_features", [1, 2, 3, 4, 5], parametros_base),
    ]

    graficar_curvas_validacion(curvas_validacion, "curvas_validacion.png")

    # ==========================================================================
    titulo("PARTE 5 - BUSQUEDA DE LA MEJOR REGULARIZACION")
    # ==========================================================================

    resultados, mejor_conservador = buscar_regularizacion(X_train, y_train, X_val, y_val)
    mejor = resultados[0]
    parametros_mejor = mejor["parametros"]

    graficar_frontera(resultados, "frontera_varianza_desempeno.png")

    # ==========================================================================
    titulo("PARTE 6 - MODELO REGULARIZADO")
    # ==========================================================================

    print(" Configuracion elegida por F1 macro en VALIDACION:")
    for clave, valor in parametros_mejor.items():
        cambio = ""
        if clave in parametros_base and parametros_base[clave] != valor:
            cambio = "   <-- cambio (antes {})".format(parametros_base[clave])
        print("   {:<20} {}{}".format(clave, valor, cambio))

    modelo_reg = construir(**parametros_mejor)
    modelo_reg.fit(X_train, y_train)

    profundidades = [a.get_depth() for a in modelo_reg.estimators_]
    hojas = [a.get_n_leaves() for a in modelo_reg.estimators_]
    print("\n Estructura del bosque regularizado:")
    print("   profundidad media {:.1f} (maxima {})".format(np.mean(profundidades), max(profundidades)))
    print("   hojas por arbol   {:.0f}".format(np.mean(hojas)))
    print("   -> {:.1f} registros por hoja en promedio".format(len(X_train) / np.mean(hojas)))

    despues = {"train": medir(modelo_reg, X_train, y_train),
               "val": medir(modelo_reg, X_val, y_val),
               "test": medir(modelo_reg, X_test, y_test)}

    d_despues = diagnosticar(despues["train"]["f1_macro"], despues["val"]["f1_macro"])
    imprimir_diagnostico("DIAGNOSTICO DEL MODELO REGULARIZADO",
                         despues["train"]["f1_macro"], despues["val"]["f1_macro"], d_despues)

    # --- variante conservadora: la mejor entre las que ya NO tienen varianza alta
    if mejor_conservador is not None:
        print("\n " + "-" * 84)
        print(" VARIANTE CONSERVADORA (regularizada hasta bajar la varianza a MEDIA)")
        print(" " + "-" * 84)
        print(" Configuracion:", {k: v for k, v in mejor_conservador["parametros"].items()
                                  if k != "n_estimators"})

        modelo_cons = construir(**mejor_conservador["parametros"])
        modelo_cons.fit(X_train, y_train)
        conservador = {"train": medir(modelo_cons, X_train, y_train),
                       "val": medir(modelo_cons, X_val, y_val),
                       "test": medir(modelo_cons, X_test, y_test)}
        d_cons = diagnosticar(conservador["train"]["f1_macro"], conservador["val"]["f1_macro"])
        imprimir_diagnostico("DIAGNOSTICO DE LA VARIANTE CONSERVADORA",
                             conservador["train"]["f1_macro"],
                             conservador["val"]["f1_macro"], d_cons)
        print("\n     Lectura: forzar la brecha por debajo de {} si logra el diagnostico de".format(
            TOPE_BRECHA_CONSERVADORA))
        print("     varianza {}, pero cuesta {:.4f} de F1 macro en PRUEBA frente al modelo".format(
            d_cons["varianza"], despues["test"]["f1_macro"] - conservador["test"]["f1_macro"]))
        print("     elegido. Es el intercambio sesgo-varianza en numeros: el conjunto de")
        print("     prueba premia al modelo flexible aunque su brecha sea mayor.")
    else:
        conservador, d_cons = None, None

    curva_reg = curva_de_aprendizaje(X_train, y_train, X_val, y_val, X_test, y_test,
                                     parametros_mejor, "DESPUES: Random Forest regularizado")

    graficar_curvas_aprendizaje([curva_base, curva_reg], "curvas_aprendizaje.png")

    # ==========================================================================
    titulo("PARTE 7 - COMPARACION ANTES / DESPUES")
    # ==========================================================================

    print(" {:<24} {:>14} {:>14} {:>12}".format("", "ANTES", "DESPUES", "cambio"))
    filas = [
        ("F1 macro entrenamiento", antes["train"]["f1_macro"], despues["train"]["f1_macro"]),
        ("F1 macro validacion", antes["val"]["f1_macro"], despues["val"]["f1_macro"]),
        ("F1 macro PRUEBA", antes["test"]["f1_macro"], despues["test"]["f1_macro"]),
        ("Accuracy PRUEBA", antes["test"]["accuracy"], despues["test"]["accuracy"]),
        ("Brecha train - val", d_antes["brecha"], d_despues["brecha"]),
    ]
    for nombre, a, b in filas:
        print(" {:<24} {:>14.4f} {:>14.4f} {:>+12.4f}".format(nombre, a, b, b - a))

    print("\n {:<24} {:>14} {:>14}".format("", "ANTES", "DESPUES"))
    print(" {:<24} {:>14} {:>14}".format("Sesgo", d_antes["sesgo"], d_despues["sesgo"]))
    print(" {:<24} {:>14} {:>14}".format("Varianza", d_antes["varianza"], d_despues["varianza"]))
    print(" {:<24} {:>14} {:>14}".format("Ajuste", d_antes["ajuste"], d_despues["ajuste"]))

    # --- evaluacion final detallada en PRUEBA --------------------------------
    print("\n Reporte por clase en el conjunto de PRUEBA (modelo regularizado):")
    y_pred_reg = modelo_reg.predict(X_test)
    y_pred_base = modelo_base.predict(X_test)
    print(classification_report(y_test, y_pred_reg, labels=clases, digits=3, zero_division=0))

    matriz_antes = confusion_matrix(y_test, y_pred_base, labels=clases)
    matriz_despues = confusion_matrix(y_test, y_pred_reg, labels=clases)

    f1_antes = f1_score(y_test, y_pred_base, labels=clases, average=None, zero_division=0)
    f1_despues = f1_score(y_test, y_pred_reg, labels=clases, average=None, zero_division=0)

    print(" F1 por clase en prueba:")
    print(" {:<12} {:>10} {:>10} {:>10}".format("clase", "ANTES", "DESPUES", "cambio"))
    for i, clase in enumerate(clases):
        print(" {:<12} {:>10.3f} {:>10.3f} {:>+10.3f}".format(
            clase, f1_antes[i], f1_despues[i], f1_despues[i] - f1_antes[i]))

    graficar_antes_despues(antes, despues, "antes_despues.png")
    graficar_matrices_lado_a_lado(clases, matriz_antes, matriz_despues, "matrices_antes_despues.png")
    graficar_f1_por_clase(clases, f1_antes, f1_despues, "f1_por_clase.png")

    # ==========================================================================
    titulo("PARTE 8 - VERIFICACION EXTRA: FUGA DE DATOS POR VEHICULO REPETIDO")
    # ==========================================================================

    # Cada fila del CSV es el pago mensual de UN vehiculo, y el mismo vehiculo
    # (mismo VIN) aparece varias veces. Con una particion aleatoria, pagos del
    # mismo coche caen a los dos lados y el bosque puede reconocerlo en vez de
    # aprender la regla. Para medirlo se reparte por VIN con GroupShuffleSplit.

    vin_train = set(vin.loc[X_train.index])
    compartidos = vin.loc[X_test.index].isin(vin_train)
    print(" Con la particion ALEATORIA usada arriba:")
    print("   filas de prueba cuyo vehiculo tambien aparece en entrenamiento: {} de {} ({:.1%})".format(
        compartidos.sum(), len(compartidos), compartidos.mean()))

    particion = GroupShuffleSplit(n_splits=1, test_size=FRACCION_PRUEBA, random_state=SEMILLA)
    idx_temp, idx_test = next(particion.split(X, y, groups=vin))

    X_g_temp, y_g_temp = X.iloc[idx_temp], y.iloc[idx_temp]
    X_g_test, y_g_test = X.iloc[idx_test], y.iloc[idx_test]

    # Se reentrenan AMBOS modelos (base y regularizado) sobre la particion agrupada
    # para ver si la regularizacion sigue siendo util cuando desaparece la fuga.
    modelo_g_base = construir(**parametros_base)
    modelo_g_base.fit(X_g_temp, y_g_temp)
    f1_g_base = medir(modelo_g_base, X_g_test, y_g_test)["f1_macro"]

    modelo_g = construir(**parametros_mejor)
    modelo_g.fit(X_g_temp, y_g_temp)
    f1_agrupada = medir(modelo_g, X_g_test, y_g_test)["f1_macro"]

    print("\n {:<44} {:>12} {:>12}".format("particion", "base", "regularizado"))
    print(" {:<44} {:>12.4f} {:>12.4f}".format(
        "aleatoria (la del analisis)", antes["test"]["f1_macro"], despues["test"]["f1_macro"]))
    print(" {:<44} {:>12.4f} {:>12.4f}".format(
        "agrupada por VIN (sin vehiculos repetidos)", f1_g_base, f1_agrupada))
    print("\n La caida de {:.4f} entre las dos filas es la parte del desempeño que".format(
        despues["test"]["f1_macro"] - f1_agrupada))
    print(" venia de reconocer vehiculos ya vistos y no de generalizar a creditos nuevos.")
    print("\n Con la particion honesta la regularizacion aporta {:+.4f} de F1 macro,".format(
        f1_agrupada - f1_g_base))
    print(" contra {:+.4f} en la particion aleatoria: al quitar la fuga se ve mejor".format(
        despues["test"]["f1_macro"] - antes["test"]["f1_macro"]))
    print(" el beneficio real del ajuste de hiperparametros.")

    graficar_fuga({"aleatoria": despues["test"]["f1_macro"], "agrupada": f1_agrupada},
                  "fuga_por_vin.png")

    resumen_extra = {"f1_g_base": f1_g_base, "f1_g_reg": f1_agrupada,
                     "conservador": conservador, "d_cons": d_cons}

    # ==========================================================================
    predecir_casos_nuevos(modelo_reg)

    # ==========================================================================
    titulo("RESUMEN DEL ANALISIS")
    # ==========================================================================
    print(" 1. Particion            : 60% entrenamiento / 20% validacion / 20% prueba, estratificada.")
    print(" 2. Modelo base          : sesgo {}, varianza {}, ajuste {}.".format(
        d_antes["sesgo"], d_antes["varianza"], d_antes["ajuste"]))
    print(" 3. Regularizacion       : {}".format(
        ", ".join("{}={}".format(k, v) for k, v in parametros_mejor.items()
                  if k in parametros_base and parametros_base[k] != v) or "sin cambios"))
    print(" 4. Modelo regularizado  : sesgo {}, varianza {}, ajuste {}.".format(
        d_despues["sesgo"], d_despues["varianza"], d_despues["ajuste"]))
    print(" 5. F1 macro en prueba   : {:.4f}  ->  {:.4f}  ({:+.4f})".format(
        antes["test"]["f1_macro"], despues["test"]["f1_macro"],
        despues["test"]["f1_macro"] - antes["test"]["f1_macro"]))
    print(" 6. Brecha train-val     : {:.4f}  ->  {:.4f}  ({:+.4f})".format(
        d_antes["brecha"], d_despues["brecha"], d_despues["brecha"] - d_antes["brecha"]))
    if resumen_extra["conservador"] is not None:
        print(" 7. Variante conservadora: brecha {:.4f} (varianza {}), pero F1 en prueba {:.4f}".format(
            resumen_extra["d_cons"]["brecha"], resumen_extra["d_cons"]["varianza"],
            resumen_extra["conservador"]["test"]["f1_macro"]))
    print(" 8. Particion honesta por VIN: {:.4f} -> {:.4f} ({:+.4f} por regularizar)".format(
        resumen_extra["f1_g_base"], resumen_extra["f1_g_reg"],
        resumen_extra["f1_g_reg"] - resumen_extra["f1_g_base"]))
    print("\n Listo.")


if __name__ == "__main__":
    main()

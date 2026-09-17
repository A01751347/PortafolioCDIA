"""
Analisis del desempeño del Random Forest de la entrega con framework.

Separa los datos en entrenamiento, validacion y prueba, diagnostica el grado de
sesgo y de varianza, explica el nivel de ajuste y aplica regularizacion para
mejorar el modelo.

Ejecutar: python3 main.py     (tarda alrededor de un minuto)
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report


# -----------------------------
# CONFIGURACION
# -----------------------------

CARPETA = os.path.dirname(os.path.abspath(__file__))
RUTA = os.path.join(CARPETA, "interest_ledger.csv")
FIGURAS = os.path.join(CARPETA, "figuras")

COLUMNAS = ["principal_amount", "interest_amount", "tasa", "antiguedad", "mes"]
SEMILLA = 42

# Umbrales del diagnostico. Se fijan aqui para que no sea "a ojo".
# El sesgo se mide con el error sobre el propio entrenamiento: un modelo con
# sesgo alto es tan rigido que ni siquiera ajusta lo que ya vio.
# La varianza se mide con la brecha entre entrenamiento y validacion.
SESGO_BAJO = 0.95
SESGO_MEDIO = 0.80
VARIANZA_BAJA = 0.03
VARIANZA_MEDIA = 0.10


# -----------------------------
# CARGAR Y PARTIR
# -----------------------------

def cargar_datos():
    datos = pd.read_csv(RUTA)

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
    # train_test_split solo parte en dos, asi que se hace en dos pasos:
    # primero se aparta el 20% de prueba y luego el 20% de validacion.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEMILLA, stratify=y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=SEMILLA, stratify=y_temp)

    return X_train, y_train, X_val, y_val, X_test, y_test


# -----------------------------
# MODELO Y DIAGNOSTICO
# -----------------------------

def construir(**cambios):
    parametros = dict(n_estimators=300, max_depth=None, min_samples_leaf=1,
                      max_features="sqrt", ccp_alpha=0.0, class_weight=None,
                      random_state=SEMILLA, n_jobs=-1)
    parametros.update(cambios)
    return RandomForestClassifier(**parametros)


def medir(modelo, X, y):
    y_pred = modelo.predict(X)
    return {"accuracy": accuracy_score(y, y_pred),
            "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0)}


def diagnosticar(f1_train, f1_val):
    brecha = f1_train - f1_val

    if f1_train >= SESGO_BAJO:
        sesgo = "BAJO"
    elif f1_train >= SESGO_MEDIO:
        sesgo = "MEDIO"
    else:
        sesgo = "ALTO"

    if brecha < VARIANZA_BAJA:
        varianza = "BAJA"
    elif brecha < VARIANZA_MEDIA:
        varianza = "MEDIA"
    else:
        varianza = "ALTA"

    # El nivel de ajuste sale de combinar los dos anteriores.
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
    print("\n  ", nombre)
    print("     F1 macro entrenamiento:", round(f1_train, 4))
    print("     F1 macro validacion   :", round(f1_val, 4))
    print("     Brecha                :", round(d["brecha"], 4))
    print("     -> SESGO   :", d["sesgo"])
    print("     -> VARIANZA:", d["varianza"])
    print("     -> AJUSTE  :", d["ajuste"])


# -----------------------------
# CURVA DE APRENDIZAJE
# -----------------------------
# Entrena el mismo modelo con cada vez mas datos. Si la curva de entrenamiento
# se queda baja hay sesgo alto; si queda muy arriba y la de validacion muy
# abajo sin acercarse, hay varianza alta.

def curva_de_aprendizaje(X_train, y_train, X_val, y_val, X_test, y_test, parametros, etiqueta):
    fracciones = [0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.0]
    tamanos, f1_train, f1_val, f1_test = [], [], [], []

    print("\n   {:>8} {:>12} {:>14} {:>12} {:>9}".format(
        "n train", "F1 train", "F1 validacion", "F1 prueba", "brecha"))

    for fraccion in fracciones:
        if fraccion < 1.0:
            X_sub, _, y_sub, _ = train_test_split(
                X_train, y_train, train_size=int(len(X_train) * fraccion),
                random_state=SEMILLA, stratify=y_train)
        else:
            X_sub, y_sub = X_train, y_train

        modelo = construir(**parametros)
        modelo.fit(X_sub, y_sub)

        tr = medir(modelo, X_sub, y_sub)["f1_macro"]
        va = medir(modelo, X_val, y_val)["f1_macro"]
        te = medir(modelo, X_test, y_test)["f1_macro"]

        tamanos.append(len(X_sub))
        f1_train.append(tr)
        f1_val.append(va)
        f1_test.append(te)

        print("   {:>8} {:>12.4f} {:>14.4f} {:>12.4f} {:>9.4f}".format(
            len(X_sub), tr, va, te, tr - va))

    return {"tamanos": tamanos, "train": f1_train, "val": f1_val,
            "test": f1_test, "etiqueta": etiqueta}


# -----------------------------
# CURVAS DE VALIDACION
# -----------------------------
# Varian un solo hiperparametro a la vez para ver donde esta el equilibrio
# entre un modelo demasiado rigido y uno demasiado flexible.

def curva_de_validacion(X_train, y_train, X_val, y_val, parametro, valores):
    f1_train, f1_val = [], []

    print("\n   Variando", parametro + ":")
    print("   {:>12} {:>12} {:>12} {:>9}".format("valor", "F1 train", "F1 val", "brecha"))

    for valor in valores:
        modelo = construir(**{parametro: valor})
        modelo.fit(X_train, y_train)

        tr = medir(modelo, X_train, y_train)["f1_macro"]
        va = medir(modelo, X_val, y_val)["f1_macro"]

        f1_train.append(tr)
        f1_val.append(va)

        print("   {:>12} {:>12.4f} {:>12.4f} {:>9.4f}".format(str(valor), tr, va, tr - va))

    return {"parametro": parametro, "valores": valores, "train": f1_train, "val": f1_val}


# -----------------------------
# REGULARIZACION
# -----------------------------
# Tecnicas disponibles en un Random Forest:
#   max_depth     corta el arbol para que no aisle casos individuales
#   ccp_alpha     poda las ramas cuya mejora no compensa tener mas hojas
#   max_features  cuantas columnas ve cada corte (controla la diversidad)

REJILLA = []
for max_features in [2, 3, 5]:
    for max_depth in [None, 25, 12]:
        for ccp_alpha in [0.0, 0.0002]:
            REJILLA.append({"max_features": max_features, "max_depth": max_depth,
                            "ccp_alpha": ccp_alpha})


def buscar_regularizacion(X_train, y_train, X_val, y_val):
    resultados = []

    print("\n   Probando", len(REJILLA), "combinaciones sobre validacion...")
    print("\n   {:>10} {:>10} {:>10} {:>10} {:>9} {:>9}".format(
        "max_feat", "max_depth", "ccp_alpha", "F1 train", "F1 val", "brecha"))

    for parametros in REJILLA:
        modelo = construir(**parametros)
        modelo.fit(X_train, y_train)

        tr = medir(modelo, X_train, y_train)["f1_macro"]
        va = medir(modelo, X_val, y_val)["f1_macro"]

        resultados.append({"parametros": parametros, "train": tr, "val": va, "brecha": tr - va})

    resultados.sort(key=lambda r: r["val"], reverse=True)

    for r in resultados:
        p = r["parametros"]
        print("   {:>10} {:>10} {:>10} {:>10.4f} {:>9.4f} {:>9.4f}".format(
            p["max_features"], str(p["max_depth"]), p["ccp_alpha"],
            r["train"], r["val"], r["brecha"]))

    return resultados


# -----------------------------
# GRAFICAS
# -----------------------------

def guardar(nombre):
    if not os.path.isdir(FIGURAS):
        os.makedirs(FIGURAS)
    plt.savefig(os.path.join(FIGURAS, nombre), dpi=140)
    plt.close()
    print("   figura guardada: figuras/" + nombre)


def graficar_curvas_aprendizaje(curvas):
    fig, ejes = plt.subplots(1, 2, figsize=(13, 5))

    for eje, curva in zip(ejes, curvas):
        eje.plot(curva["tamanos"], curva["train"], marker="o", label="Entrenamiento")
        eje.plot(curva["tamanos"], curva["val"], marker="s", label="Validacion")
        eje.plot(curva["tamanos"], curva["test"], marker="^", linestyle="--", label="Prueba")
        eje.fill_between(curva["tamanos"], curva["train"], curva["val"],
                         color="tab:red", alpha=0.12)

        brecha = curva["train"][-1] - curva["val"][-1]
        eje.annotate("brecha = {:.3f}".format(brecha),
                     xy=(curva["tamanos"][-1], (curva["train"][-1] + curva["val"][-1]) / 2),
                     xytext=(-130, 0), textcoords="offset points", fontsize=9, color="tab:red",
                     arrowprops=dict(arrowstyle="->", color="tab:red"))

        eje.set_xlabel("Registros de entrenamiento")
        eje.set_ylabel("F1 macro")
        eje.set_title(curva["etiqueta"])
        eje.set_ylim(0.3, 1.02)
        eje.grid(alpha=0.3)
        eje.legend(loc="lower right")

    plt.tight_layout()
    guardar("curvas_aprendizaje.png")


def graficar_curvas_validacion(curvas):
    fig, ejes = plt.subplots(2, 2, figsize=(12.5, 9))
    ejes = ejes.reshape(-1)

    for eje, curva in zip(ejes, curvas):
        # Los valores pueden no ser numericos (None, "sqrt"), por eso el indice.
        x = list(range(len(curva["valores"])))

        eje.plot(x, curva["train"], marker="o", label="Entrenamiento")
        eje.plot(x, curva["val"], marker="s", label="Validacion")
        eje.fill_between(x, curva["train"], curva["val"], color="tab:red", alpha=0.12)

        mejor = int(np.argmax(curva["val"]))
        eje.axvline(mejor, color="tab:green", linestyle="--", alpha=0.8)

        eje.set_xticks(x)
        eje.set_xticklabels([str(v) for v in curva["valores"]], rotation=25, fontsize=8)
        eje.set_xlabel(curva["parametro"] + "   (menos flexible -> mas flexible)")
        eje.set_ylabel("F1 macro")
        eje.set_title("Curva de validacion: " + curva["parametro"])
        eje.set_ylim(0.3, 1.02)
        eje.grid(alpha=0.3)
        eje.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    guardar("curvas_validacion.png")


def graficar_antes_despues(antes, despues):
    conjuntos = ["Entrenamiento", "Validacion", "Prueba"]
    v_antes = [antes[c]["f1_macro"] for c in ["train", "val", "test"]]
    v_despues = [despues[c]["f1_macro"] for c in ["train", "val", "test"]]
    x = np.arange(3)

    fig, ejes = plt.subplots(1, 2, figsize=(13, 5))

    ejes[0].bar(x - 0.18, v_antes, 0.36, label="ANTES", color="tab:red", alpha=0.85)
    ejes[0].bar(x + 0.18, v_despues, 0.36, label="DESPUES", color="tab:green", alpha=0.85)
    for i in range(3):
        ejes[0].text(x[i] - 0.18, v_antes[i] + 0.012, round(v_antes[i], 3),
                     ha="center", fontsize=9)
        ejes[0].text(x[i] + 0.18, v_despues[i] + 0.012, round(v_despues[i], 3),
                     ha="center", fontsize=9)
    ejes[0].set_xticks(x)
    ejes[0].set_xticklabels(conjuntos)
    ejes[0].set_ylabel("F1 macro")
    ejes[0].set_ylim(0, 1.12)
    ejes[0].set_title("Desempeño antes y despues de regularizar")
    ejes[0].legend()
    ejes[0].grid(axis="y", alpha=0.3)

    brechas = [antes["train"]["f1_macro"] - antes["val"]["f1_macro"],
               despues["train"]["f1_macro"] - despues["val"]["f1_macro"]]
    ejes[1].bar(["ANTES", "DESPUES"], brechas, color=["tab:red", "tab:green"],
                alpha=0.85, width=0.5)
    for i, v in enumerate(brechas):
        ejes[1].text(i, v + 0.004, round(v, 4), ha="center", fontsize=10)
    ejes[1].axhline(VARIANZA_MEDIA, color="darkred", linestyle="--",
                    label="umbral varianza ALTA")
    ejes[1].set_ylabel("Brecha entrenamiento - validacion")
    ejes[1].set_title("Varianza antes y despues")
    ejes[1].legend(fontsize=8)
    ejes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    guardar("antes_despues.png")


def graficar_matrices(clases, matriz_antes, matriz_despues):
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5.8))

    for eje, matriz, subtitulo in [(ejes[0], matriz_antes, "ANTES"),
                                   (ejes[1], matriz_despues, "DESPUES")]:
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

    plt.suptitle("Matriz de confusion en PRUEBA", fontsize=13)
    plt.tight_layout()
    guardar("matrices_antes_despues.png")


def graficar_f1_por_clase(clases, f1_antes, f1_despues):
    x = np.arange(len(clases))

    plt.figure(figsize=(9.5, 5))
    plt.bar(x - 0.18, f1_antes, 0.36, label="ANTES", color="tab:red", alpha=0.85)
    plt.bar(x + 0.18, f1_despues, 0.36, label="DESPUES", color="tab:green", alpha=0.85)

    for i in range(len(clases)):
        delta = f1_despues[i] - f1_antes[i]
        plt.text(x[i], max(f1_antes[i], f1_despues[i]) + 0.03, "{:+.3f}".format(delta),
                 ha="center", fontsize=9, color="darkgreen" if delta >= 0 else "darkred")

    plt.xticks(x, clases, rotation=30, ha="right")
    plt.ylabel("F1 en prueba")
    plt.ylim(0, 1.15)
    plt.title("Efecto de la regularizacion clase por clase")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    guardar("f1_por_clase.png")


def graficar_fuga(f1_aleatoria, f1_agrupada):
    plt.figure(figsize=(8, 5))
    barras = plt.bar(["Particion ALEATORIA", "Particion POR VEHICULO"],
                     [f1_aleatoria, f1_agrupada], color=["tab:orange", "tab:blue"], width=0.5)
    for barra in barras:
        plt.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.012,
                 round(barra.get_height(), 3), ha="center", fontsize=11)

    plt.title("F1 macro segun como se parten los datos")
    plt.ylabel("F1 macro")
    plt.ylim(0, max(f1_aleatoria, f1_agrupada) * 1.25)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    guardar("fuga_por_vin.png")


# -----------------------------
# PRINCIPAL
# -----------------------------

X, y, vin = cargar_datos()
X_train, y_train, X_val, y_val, X_test, y_test = partir_en_tres(X, y)
clases = sorted(y.unique())

print("=" * 80)
print(" 1. PARTICION EN TRES CONJUNTOS")
print("=" * 80)
print(" Modelo analizado: Random Forest (entrega con framework)")
print(" Metrica: F1 macro")
print("\n {:<16} {:>8} {:>8}".format("conjunto", "n", "%"))
for nombre, conjunto in [("Entrenamiento", y_train), ("Validacion", y_val), ("Prueba", y_test)]:
    print(" {:<16} {:>8} {:>7.0f}%".format(nombre, len(conjunto), 100 * len(conjunto) / len(y)))

print("\n Proporcion de cada clase (stratify la conserva igual en los tres):")
print(" {:<12} {:>14} {:>12} {:>10}".format("clase", "entrenamiento", "validacion", "prueba"))
for clase in clases:
    print(" {:<12} {:>13.1f}% {:>11.1f}% {:>9.1f}%".format(
        clase, 100 * (y_train == clase).mean(), 100 * (y_val == clase).mean(),
        100 * (y_test == clase).mean()))

# -------------------- modelo base --------------------
print("\n" + "=" * 80)
print(" 2. MODELO BASE SIN REGULARIZAR")
print("=" * 80)

parametros_base = {"max_features": "sqrt", "max_depth": None, "ccp_alpha": 0.0}

modelo_base = construir(**parametros_base)
modelo_base.fit(X_train, y_train)

hojas = np.mean([a.get_n_leaves() for a in modelo_base.estimators_])
print(" Arboles sin restriccion: profundidad media",
      round(np.mean([a.get_depth() for a in modelo_base.estimators_]), 1),
      "| hojas por arbol", round(hojas))
print(" Eso deja", round(len(X_train) / hojas, 1),
      "registros por hoja: el bosque esta memorizando.")

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

# -------------------- curva de aprendizaje --------------------
print("\n" + "=" * 80)
print(" 3. CURVA DE APRENDIZAJE DEL MODELO BASE")
print("=" * 80)

curva_base = curva_de_aprendizaje(X_train, y_train, X_val, y_val, X_test, y_test,
                                  parametros_base, "ANTES: sin regularizar")

# -------------------- curvas de validacion --------------------
print("\n" + "=" * 80)
print(" 4. EFECTO DE CADA HIPERPARAMETRO")
print("=" * 80)

curvas_validacion = [
    curva_de_validacion(X_train, y_train, X_val, y_val,
                        "max_depth", [2, 4, 6, 8, 12, 16, 20, 25, None]),
    curva_de_validacion(X_train, y_train, X_val, y_val,
                        "min_samples_leaf", [50, 25, 10, 5, 3, 2, 1]),
    curva_de_validacion(X_train, y_train, X_val, y_val,
                        "ccp_alpha", [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0]),
    curva_de_validacion(X_train, y_train, X_val, y_val,
                        "max_features", [1, 2, 3, 4, 5]),
]
graficar_curvas_validacion(curvas_validacion)

# -------------------- regularizacion --------------------
print("\n" + "=" * 80)
print(" 5. BUSQUEDA DE LA MEJOR REGULARIZACION")
print("=" * 80)

resultados = buscar_regularizacion(X_train, y_train, X_val, y_val)
parametros_mejor = resultados[0]["parametros"]

print("\n" + "=" * 80)
print(" 6. MODELO REGULARIZADO")
print("=" * 80)
print(" Configuracion elegida:", parametros_mejor)

modelo_reg = construir(**parametros_mejor)
modelo_reg.fit(X_train, y_train)

hojas = np.mean([a.get_n_leaves() for a in modelo_reg.estimators_])
print(" Hojas por arbol:", round(hojas), "->", round(len(X_train) / hojas, 1),
      "registros por hoja (antes eran menos)")

despues = {"train": medir(modelo_reg, X_train, y_train),
           "val": medir(modelo_reg, X_val, y_val),
           "test": medir(modelo_reg, X_test, y_test)}

d_despues = diagnosticar(despues["train"]["f1_macro"], despues["val"]["f1_macro"])
imprimir_diagnostico("DIAGNOSTICO DEL MODELO REGULARIZADO",
                     despues["train"]["f1_macro"], despues["val"]["f1_macro"], d_despues)

curva_reg = curva_de_aprendizaje(X_train, y_train, X_val, y_val, X_test, y_test,
                                 parametros_mejor, "DESPUES: regularizado")
graficar_curvas_aprendizaje([curva_base, curva_reg])

# -------------------- comparacion --------------------
print("\n" + "=" * 80)
print(" 7. COMPARACION ANTES / DESPUES")
print("=" * 80)
print(" {:<24} {:>12} {:>12} {:>10}".format("", "ANTES", "DESPUES", "cambio"))
for nombre, a, b in [
        ("F1 macro entrenamiento", antes["train"]["f1_macro"], despues["train"]["f1_macro"]),
        ("F1 macro validacion", antes["val"]["f1_macro"], despues["val"]["f1_macro"]),
        ("F1 macro PRUEBA", antes["test"]["f1_macro"], despues["test"]["f1_macro"]),
        ("Accuracy PRUEBA", antes["test"]["accuracy"], despues["test"]["accuracy"]),
        ("Brecha train - val", d_antes["brecha"], d_despues["brecha"])]:
    print(" {:<24} {:>12.4f} {:>12.4f} {:>+10.4f}".format(nombre, a, b, b - a))

print("\n {:<24} {:>12} {:>12}".format("", "ANTES", "DESPUES"))
print(" {:<24} {:>12} {:>12}".format("Sesgo", d_antes["sesgo"], d_despues["sesgo"]))
print(" {:<24} {:>12} {:>12}".format("Varianza", d_antes["varianza"], d_despues["varianza"]))
print(" {:<24} {:>12} {:>12}".format("Ajuste", d_antes["ajuste"], d_despues["ajuste"]))

y_pred_base = modelo_base.predict(X_test)
y_pred_reg = modelo_reg.predict(X_test)

print("\n Reporte por clase en PRUEBA (modelo regularizado):")
print(classification_report(y_test, y_pred_reg, labels=clases, digits=3, zero_division=0))

f1_antes = f1_score(y_test, y_pred_base, labels=clases, average=None, zero_division=0)
f1_despues = f1_score(y_test, y_pred_reg, labels=clases, average=None, zero_division=0)

print(" F1 por clase en prueba:")
print(" {:<12} {:>10} {:>10} {:>10}".format("clase", "ANTES", "DESPUES", "cambio"))
for i, clase in enumerate(clases):
    print(" {:<12} {:>10.3f} {:>10.3f} {:>+10.3f}".format(
        clase, f1_antes[i], f1_despues[i], f1_despues[i] - f1_antes[i]))

graficar_antes_despues(antes, despues)
graficar_matrices(clases, confusion_matrix(y_test, y_pred_base, labels=clases),
                  confusion_matrix(y_test, y_pred_reg, labels=clases))
graficar_f1_por_clase(clases, f1_antes, f1_despues)

# -------------------- fuga por vehiculo --------------------
print("\n" + "=" * 80)
print(" 8. REVISION EXTRA: VEHICULOS REPETIDOS")
print("=" * 80)

# Cada fila es el pago mensual de un vehiculo y el mismo VIN aparece en varios
# meses, asi que con una particion al azar pagos del mismo coche caen a los dos
# lados y el bosque puede reconocerlo en vez de aprender la regla.
compartidos = vin.loc[X_test.index].isin(set(vin.loc[X_train.index]))
print(" Filas de prueba cuyo vehiculo tambien esta en entrenamiento: {} de {} ({:.0%})".format(
    compartidos.sum(), len(compartidos), compartidos.mean()))

particion = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEMILLA)
idx_temp, idx_test = next(particion.split(X, y, groups=vin))

modelo_g = construir(**parametros_mejor)
modelo_g.fit(X.iloc[idx_temp], y.iloc[idx_temp])
f1_agrupada = medir(modelo_g, X.iloc[idx_test], y.iloc[idx_test])["f1_macro"]

print("\n {:<44} {:>10}".format("particion", "F1 macro"))
print(" {:<44} {:>10.4f}".format("aleatoria (la del analisis)", despues["test"]["f1_macro"]))
print(" {:<44} {:>10.4f}".format("agrupada por VIN", f1_agrupada))
print("\n La caida de", round(despues["test"]["f1_macro"] - f1_agrupada, 4),
      "es lo que venia de reconocer coches ya vistos")
print(" y no de generalizar a creditos nuevos.")

graficar_fuga(despues["test"]["f1_macro"], f1_agrupada)

# -------------------- predicciones --------------------
casos = pd.DataFrame([
    [350000.0,  1500.0, 1500.0 / 350000.0,  1, 2],
    [1200000.0, 9000.0, 9000.0 / 1200000.0, 0, 5],
    [90000.0,    400.0,  400.0 / 90000.0,   4, 8],
], columns=COLUMNAS)

print("\n" + "=" * 80)
print(" 9. PREDICCIONES CON EL MODELO FINAL")
print("=" * 80)

predicciones = modelo_reg.predict(casos)
probabilidades = modelo_reg.predict_proba(casos)

for i in range(len(casos)):
    print("\n principal ${:,.2f} | interes ${:,.2f} | antiguedad {} | mes {}".format(
        casos["principal_amount"][i], casos["interest_amount"][i],
        casos["antiguedad"][i], casos["mes"][i]))
    print("   PREDICCION:", predicciones[i])
    orden = np.argsort(probabilidades[i])[::-1]
    print("   confianza:", "  ".join(
        "{} {:.1f}%".format(modelo_reg.classes_[j], probabilidades[i][j] * 100)
        for j in orden if probabilidades[i][j] > 0.005))

print("\n" + "=" * 80)
print(" RESUMEN")
print("=" * 80)
print(" Modelo base        : sesgo {}, varianza {}, ajuste {}".format(
    d_antes["sesgo"], d_antes["varianza"], d_antes["ajuste"]))
print(" Modelo regularizado: sesgo {}, varianza {}, ajuste {}".format(
    d_despues["sesgo"], d_despues["varianza"], d_despues["ajuste"]))
print(" F1 macro en prueba : {:.4f} -> {:.4f}".format(
    antes["test"]["f1_macro"], despues["test"]["f1_macro"]))
print(" Brecha train-val   : {:.4f} -> {:.4f}".format(d_antes["brecha"], d_despues["brecha"]))


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


# -----------------------------
# CONFIGURACION
# -----------------------------

RUTA = "interest_ledger.csv"

# Columnas que se le dan al modelo (las mismas que use en la entrega sin framework,
COLUMNAS = ["principal_amount", "interest_amount", "tasa", "antiguedad", "mes"]

SEMILLA = 42          # para que el resultado sea el mismo cada vez que se corre
PORCENTAJE_TEST = 0.25


# -----------------------------
# CARGAR DATOS
# -----------------------------

def cargar_datos():
    datos = pd.read_csv(RUTA)

    # El CSV trae espacios raros (\xa0) pegados a la clase.
    datos["collateral"] = datos["collateral"].str.replace("\xa0", " ")
    datos["collateral"] = datos["collateral"].str.strip().str.upper()

    # Del mes_date solo me interesan el año y el mes.
    datos["year"] = datos["month_date"].str[:4].astype(int)
    datos["mes"] = datos["month_date"].str[5:7].astype(int)

    # Tasa mensual: cuanto interes genera cada peso prestado.
    datos["tasa"] = 0.0
    con_principal = datos["principal_amount"] > 0
    datos.loc[con_principal, "tasa"] = (
        datos.loc[con_principal, "interest_amount"] / datos.loc[con_principal, "principal_amount"]
    )

    # Años que tenia el vehiculo al momento del pago.
    datos["antiguedad"] = datos["year"] - datos["year_model"]

    X = datos[COLUMNAS]
    y = datos["collateral"]

    return X, y


# -----------------------------
# ENTRENAR
# -----------------------------
# Random Forest no necesita normalizar, cada arbol parte las variables 

def crear_modelo():
    modelo = RandomForestClassifier(
        n_estimators=300,       # numero de arboles del bosque
        max_depth=None,         # los arboles crecen hasta separar bien los datos
        min_samples_leaf=1,     # minimo de datos que debe quedar en cada hoja
        max_features="sqrt",    # cada corte solo considera algunas columnas (da variedad)
        class_weight=None,      # probe "balanced" y bajo el F1 macro, ver reporte
        random_state=SEMILLA,
        n_jobs=-1,              # usa todos los nucleos del procesador
    )
    return modelo


# -----------------------------
# METRICAS
# -----------------------------

def mostrar_metricas(y_real, y_pred, clases):
    print("\nAccuracy:", round(accuracy_score(y_real, y_pred) * 100, 2), "%")
    print("F1 macro:", round(f1_score(y_real, y_pred, average="macro"), 3))
    print("F1 ponderado:", round(f1_score(y_real, y_pred, average="weighted"), 3))

    # La matriz de confusion: las filas son la clase real y las columnas la predicha.
    matriz = confusion_matrix(y_real, y_pred, labels=clases)

    print("\nMatriz de confusion")
    print("Clases:", list(clases))
    for i in range(len(clases)):
        print(clases[i], matriz[i])

    # Precision, recall y F1 de cada clase por separado.
    print("\nReporte por clase")
    print(classification_report(y_real, y_pred, labels=clases, digits=3, zero_division=0))

    return matriz


# -----------------------------
# COMPARACION DE CONFIGURACIONES
# -----------------------------

def comparar_configuraciones(X_train, y_train):
    configuraciones = [
        {"n_estimators": 100, "max_depth": 5, "class_weight": None},
        {"n_estimators": 100, "max_depth": 10, "class_weight": None},
        {"n_estimators": 100, "max_depth": None, "class_weight": None},
        {"n_estimators": 300, "max_depth": None, "class_weight": None},
        {"n_estimators": 300, "max_depth": None, "class_weight": "balanced"},
    ]

    print("\nValidacion cruzada (5 partes) sobre entrenamiento")

    for config in configuraciones:
        modelo = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            max_features="sqrt",
            class_weight=config["class_weight"],
            random_state=SEMILLA,
            n_jobs=-1,
        )

        # usa una parte distinta como validacion en cada vuelta.
        resultados = cross_val_score(modelo, X_train, y_train, cv=5, scoring="f1_macro")

        print(
            "arboles:", config["n_estimators"],
            "| profundidad:", config["max_depth"],
            "| pesos:", config["class_weight"],
            "| F1 macro:", round(resultados.mean(), 3),
        )


# -----------------------------
# PREDICCIONES NUEVAS
# -----------------------------
# Aqui es donde el modelo ya entrenado se usa con datos inventados

def predecir_casos_nuevos(modelo):
    casos = pd.DataFrame([
        # principal, interes, tasa, antiguedad, mes
        [350000.0, 1500.0, 1500.0 / 350000.0, 1, 2],
        [1200000.0, 9000.0, 9000.0 / 1200000.0, 0, 5],
        [90000.0, 400.0, 400.0 / 90000.0, 4, 8],
    ], columns=COLUMNAS)

    predicciones = modelo.predict(casos)
    probabilidades = modelo.predict_proba(casos)

    print("\nPredicciones de casos nuevos")

    for i in range(len(casos)):
        print("\nCaso", i + 1)
        print("Principal:", casos["principal_amount"][i], "| Interes:", casos["interest_amount"][i])
        print("Antiguedad:", casos["antiguedad"][i], "| Mes:", casos["mes"][i])
        print("Prediccion:", predicciones[i])

        # predict_proba regresa que tan seguro esta el bosque de cada clase.
        for j in range(len(modelo.classes_)):
            print("   ", modelo.classes_[j], round(probabilidades[i][j] * 100, 1), "%")


# -----------------------------
# GRAFICAS
# -----------------------------

def graficar_matriz(clases, matriz):
    plt.figure(figsize=(7, 5))
    plt.imshow(matriz)
    plt.title("Matriz de Confusion (Random Forest)")
    plt.xlabel("Prediccion")
    plt.ylabel("Clase real")
    plt.xticks(range(len(clases)), clases, rotation=45)
    plt.yticks(range(len(clases)), clases)

    for i in range(len(clases)):
        for j in range(len(clases)):
            plt.text(j, i, matriz[i][j], ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig("matriz_confusion.png")
    plt.show()


def graficar_importancias(modelo):
    # El modelo guarda que tanto ayudo cada columna a separar las clases.
    importancias = modelo.feature_importances_

    plt.figure(figsize=(7, 5))
    plt.bar(COLUMNAS, importancias)
    plt.title("Importancia de cada variable")
    plt.ylabel("Importancia")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("importancias.png")
    plt.show()


# -----------------------------
# PRINCIPAL
# -----------------------------

X, y = cargar_datos()

# stratify reparte las clases en la misma proporcion en train y en test, si no las clases chicas (DEMO, REFACC) podrian quedar casi todas de un solo lado.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=PORCENTAJE_TEST,
    random_state=SEMILLA,
    stratify=y,
)

print("Total de datos:", len(X))
print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))
print("\nDatos por clase en entrenamiento:")
print(y_train.value_counts())

comparar_configuraciones(X_train, y_train)

modelo = crear_modelo()
modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)

clases = list(modelo.classes_)
matriz = mostrar_metricas(y_test, predicciones, clases)

print("\nImportancia de cada variable")
for i in range(len(COLUMNAS)):
    print(COLUMNAS[i], round(modelo.feature_importances_[i], 3))

predecir_casos_nuevos(modelo)

graficar_matriz(clases, matriz)
graficar_importancias(modelo)

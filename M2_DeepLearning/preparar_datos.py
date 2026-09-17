"""
Prepara el dataset GTSRB (señales de transito alemanas).

Descarga los ZIP oficiales, recorta cada foto a la region donde esta la señal,
la deja en 32x32 y guarda todo en datos/gtsrb_32.npz.

Las fotos de entrenamiento NO son independientes: cada señal fisica se
fotografio 30 veces seguidas mientras el auto se le acercaba, y esas 30 forman
una "pista" que se reconoce en el nombre del archivo (00012_00007.ppm -> pista
12). Por eso la validacion se separa por PISTA COMPLETA: si se partiera al azar
caerian fotos casi identicas de los dos lados y el resultado saldria inflado.

Ejecutar: python3 preparar_datos.py    (solo hace falta una vez)
"""

import csv
import os
import sys
import urllib.request
import zipfile

import numpy as np
from PIL import Image


# ==============================================================================
# CONFIGURACION
# ==============================================================================

CARPETA = os.path.dirname(os.path.abspath(__file__))
CARPETA_DATOS = os.path.join(CARPETA, "datos")
RUTA_NPZ = os.path.join(CARPETA_DATOS, "gtsrb_32.npz")

BASE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
ARCHIVOS = [
    ("GTSRB_train.zip", "GTSRB_Final_Training_Images.zip"),
    ("GTSRB_test.zip", "GTSRB_Final_Test_Images.zip"),
    ("GTSRB_test_gt.zip", "GTSRB_Final_Test_GT.zip"),
]

TAMANO = 32                 # las imagenes se llevan todas a 32x32 pixeles
FRACCION_VALIDACION = 0.20  # 20% de las PISTAS, no de las imagenes
SEMILLA = 42

# Nombre legible de cada una de las 43 clases, en el orden oficial del dataset.
NOMBRES_CLASES = [
    "Limite 20 km/h", "Limite 30 km/h", "Limite 50 km/h", "Limite 60 km/h",
    "Limite 70 km/h", "Limite 80 km/h", "Fin limite 80 km/h", "Limite 100 km/h",
    "Limite 120 km/h", "Prohibido rebasar", "Prohibido rebasar camiones",
    "Prioridad en cruce", "Via preferente", "Ceda el paso", "Alto",
    "Circulacion prohibida", "Prohibido camiones", "Prohibido el paso",
    "Precaucion general", "Curva peligrosa izquierda", "Curva peligrosa derecha",
    "Curvas sucesivas", "Camino irregular", "Pavimento resbaloso",
    "Estrechamiento derecha", "Obras en la via", "Semaforo", "Peatones",
    "Cruce de niños", "Cruce de ciclistas", "Hielo o nieve", "Cruce de animales",
    "Fin de restricciones", "Giro obligatorio derecha", "Giro obligatorio izquierda",
    "Siga de frente", "Siga de frente o derecha", "Siga de frente o izquierda",
    "Circule por la derecha", "Circule por la izquierda", "Glorieta obligatoria",
    "Fin prohibido rebasar", "Fin prohibido rebasar camiones",
]


# ==============================================================================
# DESCARGA Y DESCOMPRESION
# ==============================================================================

def barra_progreso(bloques, tamano_bloque, tamano_total):
    if tamano_total <= 0:
        return
    leido = bloques * tamano_bloque
    porcentaje = min(100.0, leido * 100.0 / tamano_total)
    sys.stdout.write("\r      {:5.1f}%  ({:.0f} MB de {:.0f} MB)".format(
        porcentaje, leido / 1e6, tamano_total / 1e6))
    sys.stdout.flush()


def descargar_si_falta():
    if not os.path.isdir(CARPETA_DATOS):
        os.makedirs(CARPETA_DATOS)

    for nombre_local, nombre_remoto in ARCHIVOS:
        destino = os.path.join(CARPETA_DATOS, nombre_local)

        if os.path.exists(destino):
            print("   ya existe: {} ({:.0f} MB)".format(
                nombre_local, os.path.getsize(destino) / 1e6))
            continue

        print("   descargando {} ...".format(nombre_remoto))
        urllib.request.urlretrieve(BASE_URL + nombre_remoto, destino, barra_progreso)
        print()


def descomprimir_si_falta():
    marca = os.path.join(CARPETA_DATOS, "GTSRB", "Final_Training", "Images")

    if os.path.isdir(marca):
        print("   los archivos ya estaban descomprimidos")
        return

    for nombre_local, _ in ARCHIVOS:
        origen = os.path.join(CARPETA_DATOS, nombre_local)
        print("   descomprimiendo {} ...".format(nombre_local))
        with zipfile.ZipFile(origen) as zip_abierto:
            zip_abierto.extractall(CARPETA_DATOS)


# ==============================================================================
# LECTURA DE IMAGENES
# ==============================================================================

def leer_imagen(ruta, x1, y1, x2, y2):
    # Recortar por la region de interes quita el asfalto y el cielo, que no
    # aportan nada, y deja la señal siempre del mismo tamaño relativo.
    with Image.open(ruta) as imagen:
        imagen = imagen.convert("RGB")
        imagen = imagen.crop((x1, y1, x2, y2))
        imagen = imagen.resize((TAMANO, TAMANO), Image.BILINEAR)
        return np.asarray(imagen, dtype=np.uint8)


def cargar_entrenamiento():
    raiz = os.path.join(CARPETA_DATOS, "GTSRB", "Final_Training", "Images")

    imagenes, clases, pistas = [], [], []

    for id_clase in range(43):
        carpeta = os.path.join(raiz, "{:05d}".format(id_clase))
        anotaciones = os.path.join(carpeta, "GT-{:05d}.csv".format(id_clase))

        with open(anotaciones) as archivo:
            for fila in csv.DictReader(archivo, delimiter=";"):
                ruta = os.path.join(carpeta, fila["Filename"])

                imagenes.append(leer_imagen(ruta,
                                            int(fila["Roi.X1"]), int(fila["Roi.Y1"]),
                                            int(fila["Roi.X2"]), int(fila["Roi.Y2"])))
                clases.append(int(fila["ClassId"]))

                # "00012_00007.ppm" -> pista global 12 de la clase 5 => "5_12"
                numero_pista = fila["Filename"].split("_")[0]
                pistas.append("{}_{}".format(id_clase, numero_pista))

        print("\r      clase {:>2}/42 leida".format(id_clase), end="")

    print()
    return np.array(imagenes), np.array(clases), np.array(pistas)


def cargar_prueba():
    raiz = os.path.join(CARPETA_DATOS, "GTSRB", "Final_Test", "Images")
    anotaciones = os.path.join(CARPETA_DATOS, "GT-final_test.csv")

    imagenes, clases = [], []

    with open(anotaciones) as archivo:
        for fila in csv.DictReader(archivo, delimiter=";"):
            ruta = os.path.join(raiz, fila["Filename"])

            imagenes.append(leer_imagen(ruta,
                                        int(fila["Roi.X1"]), int(fila["Roi.Y1"]),
                                        int(fila["Roi.X2"]), int(fila["Roi.Y2"])))
            clases.append(int(fila["ClassId"]))

    return np.array(imagenes), np.array(clases)


# ==============================================================================
# PARTICION POR PISTA
# ==============================================================================

def separar_por_pista(imagenes, clases, pistas):
    # Se sortean las pistas de cada clase por separado para que la validacion
    # conserve la proporcion de clases del dataset.
    generador = np.random.default_rng(SEMILLA)

    pistas_validacion = set()
    for id_clase in range(43):
        pistas_clase = np.unique(pistas[clases == id_clase])
        generador.shuffle(pistas_clase)

        cuantas = max(1, int(round(len(pistas_clase) * FRACCION_VALIDACION)))
        pistas_validacion.update(pistas_clase[:cuantas])

    es_validacion = np.array([p in pistas_validacion for p in pistas])

    return (imagenes[~es_validacion], clases[~es_validacion],
            imagenes[es_validacion], clases[es_validacion],
            len(np.unique(pistas)), len(pistas_validacion))


# ==============================================================================
# PROGRAMA PRINCIPAL
# ==============================================================================

def preparar(forzar=False):
    if os.path.exists(RUTA_NPZ) and not forzar:
        print(" El archivo datos/gtsrb_32.npz ya existe; no hay nada que preparar.")
        return RUTA_NPZ

    print("=" * 78)
    print(" PREPARACION DEL DATASET GTSRB")
    print("=" * 78)

    print("\n [1/5] Descarga de los archivos oficiales")
    descargar_si_falta()

    print("\n [2/5] Descompresion")
    descomprimir_si_falta()

    print("\n [3/5] Lectura de las imagenes de entrenamiento (39,209 archivos .ppm)")
    imagenes, clases, pistas = cargar_entrenamiento()
    print("      leidas {} imagenes de {} pistas".format(len(imagenes), len(np.unique(pistas))))

    print("\n [4/5] Separacion entrenamiento / validacion POR PISTA")
    X_train, y_train, X_val, y_val, n_pistas, n_pistas_val = separar_por_pista(
        imagenes, clases, pistas)
    print("      pistas totales      : {}".format(n_pistas))
    print("      pistas a validacion : {}  ({:.0%})".format(n_pistas_val, n_pistas_val / n_pistas))
    print("      imagenes train      : {}".format(len(X_train)))
    print("      imagenes validacion : {}".format(len(X_val)))
    print("      NINGUNA señal fisica aparece en los dos conjuntos a la vez.")

    print("\n [5/5] Lectura del conjunto de prueba oficial (12,630 archivos .ppm)")
    X_test, y_test = cargar_prueba()
    print("      leidas {} imagenes".format(len(X_test)))

    print("\n Guardando datos/gtsrb_32.npz ...")
    np.savez_compressed(RUTA_NPZ,
                        X_train=X_train, y_train=y_train,
                        X_val=X_val, y_val=y_val,
                        X_test=X_test, y_test=y_test)
    print(" Listo: {:.0f} MB".format(os.path.getsize(RUTA_NPZ) / 1e6))

    return RUTA_NPZ


def cargar():
    if not os.path.exists(RUTA_NPZ):
        preparar()

    datos = np.load(RUTA_NPZ)
    return (datos["X_train"], datos["y_train"],
            datos["X_val"], datos["y_val"],
            datos["X_test"], datos["y_test"])


if __name__ == "__main__":
    preparar(forzar="--forzar" in sys.argv)

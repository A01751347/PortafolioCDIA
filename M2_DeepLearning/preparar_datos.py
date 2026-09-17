"""
================================================================================
 PREPARACION DEL DATASET GTSRB
================================================================================

 Modulo 2 - Portafolio de Implementacion (Deep Learning)
 Santiago Serrano Montalvo - A01751347

 QUE ES GTSRB
 ------------
 German Traffic Sign Recognition Benchmark. Son fotografias REALES de señales
 de transito tomadas desde una camara montada en un auto circulando por
 carreteras de Alemania. No es un dataset sintetico ni un ejemplo de clase:
 fue publicado para la competencia IJCNN 2011 y sigue siendo el estandar para
 medir reconocimiento de señales.

   * 39,209 imagenes de entrenamiento
   * 12,630 imagenes de prueba (conjunto oficial, con sus etiquetas aparte)
   * 43 clases (limites de velocidad, alto, ceda el paso, prohibiciones, etc.)
   * Tamaño variable, desde ~15x15 hasta ~250x250 pixeles
   * Condiciones reales: desenfoque de movimiento, sombras, contraluz, lluvia,
     señales parcialmente tapadas y muy distintos tamaños segun la distancia

 DETALLE CRITICO: LAS "PISTAS" (TRACKS)
 --------------------------------------
 Las imagenes de entrenamiento NO son independientes. Cada señal fisica fue
 fotografiada 30 veces seguidas mientras el auto se le acercaba, y esas 30
 fotos forman una "pista" (track). Se reconocen por el nombre del archivo:

       00012_00007.ppm
       ^^^^^ pista     ^^^^^ cuadro dentro de la pista

 Las 30 fotos de una misma pista son casi identicas. Si la particion de
 validacion se hiciera al azar, cuadros de la misma señal caerian en
 entrenamiento y en validacion a la vez, y el modelo obtendria una puntuacion
 inflada por reconocer una foto casi repetida en lugar de generalizar.

 POR ESO LA VALIDACION SE SEPARA POR PISTA COMPLETA: una señal fisica esta
 entera en entrenamiento o entera en validacion, nunca repartida.

 QUE HACE ESTE ARCHIVO
 ---------------------
   1. Descarga los tres archivos ZIP oficiales (si no estan ya en datos/).
   2. Los descomprime.
   3. Recorta cada imagen a la region de interes que marca la anotacion,
      la redimensiona a 32x32 y la guarda como arreglo de enteros.
   4. Separa entrenamiento y validacion POR PISTA.
   5. Guarda todo en datos/gtsrb_32.npz para que el entrenamiento no tenga
      que volver a leer 51,839 archivos sueltos.

 EJECUCION
 ---------
       python3 preparar_datos.py

 Solo hay que correrlo una vez. main.py lo invoca automaticamente si hace falta.

================================================================================
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
    """Baja los ZIP oficiales solo si no estan ya en la carpeta datos/."""
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
    """Descomprime los ZIP solo si la carpeta GTSRB todavia no esta armada."""
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
    """Abre una imagen .ppm, la recorta a su region de interes y la escala.

    El dataset incluye las coordenadas del rectangulo que contiene la señal.
    Recortar por ahi quita el fondo (asfalto, cielo, arboles) que no aporta
    nada y hace que la señal ocupe siempre una porcion parecida de la imagen.
    """
    with Image.open(ruta) as imagen:
        imagen = imagen.convert("RGB")
        imagen = imagen.crop((x1, y1, x2, y2))
        imagen = imagen.resize((TAMANO, TAMANO), Image.BILINEAR)
        return np.asarray(imagen, dtype=np.uint8)


def cargar_entrenamiento():
    """Recorre las 43 carpetas de entrenamiento y devuelve imagenes, clases y pistas."""
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
    """Lee el conjunto de prueba OFICIAL con sus etiquetas."""
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
    """Reparte las PISTAS (no las imagenes) entre entrenamiento y validacion.

    Se sortean las pistas de cada clase por separado para que la validacion
    conserve la proporcion de clases del dataset original. Todas las fotos de
    una misma señal fisica viajan juntas.
    """
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
    """Deja listo datos/gtsrb_32.npz y devuelve su ruta."""
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
    """Devuelve los seis arreglos, preparandolos antes si es necesario."""
    if not os.path.exists(RUTA_NPZ):
        preparar()

    datos = np.load(RUTA_NPZ)
    return (datos["X_train"], datos["y_train"],
            datos["X_val"], datos["y_val"],
            datos["X_test"], datos["y_test"])


if __name__ == "__main__":
    preparar(forzar="--forzar" in sys.argv)

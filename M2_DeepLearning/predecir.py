"""
================================================================================
 PREDICCIONES DESDE LA CONSOLA CON EL MODELO ENTRENADO
================================================================================

 Modulo 2 - Portafolio de Implementacion (Deep Learning)
 Santiago Serrano Montalvo - A01751347

 Carga el modelo que dejo main.py (modelo_gtsrb.pt) y clasifica señales de
 transito. No vuelve a entrenar nada: solo usa los pesos guardados.

 FORMAS DE USARLO
 ----------------
   python3 predecir.py
       Toma 10 imagenes al azar del conjunto de prueba, las clasifica y
       muestra las 3 clases mas probables de cada una.

   python3 predecir.py --cuantas 25
       Lo mismo pero con 25 imagenes.

   python3 predecir.py --imagen foto.jpg
       Clasifica una fotografia propia (jpg, png o ppm). La imagen debe estar
       recortada mas o menos alrededor de la señal.

   python3 predecir.py --dudosas
       Muestra las predicciones donde el modelo estuvo MENOS seguro. Es util
       para ver en que casos conviene que un sistema real pida confirmacion
       en lugar de decidir solo.

 En todos los casos ademas deja una figura en figuras/predicciones.png.

================================================================================
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from PIL import Image

import preparar_datos
from preparar_datos import NOMBRES_CLASES
from main import ModeloMejorado, DISPOSITIVO, CARPETA_FIGURAS, RUTA_MODELO, guardar


# ==============================================================================
# CARGA DEL MODELO
# ==============================================================================

def cargar_modelo():
    """Reconstruye la arquitectura y le carga los pesos entrenados."""
    if not os.path.exists(RUTA_MODELO):
        raise SystemExit(
            "\n No existe modelo_gtsrb.pt.\n"
            " Primero hay que entrenar el modelo con:  python3 main.py\n")

    guardado = torch.load(RUTA_MODELO, map_location=DISPOSITIVO, weights_only=False)

    modelo = ModeloMejorado().to(DISPOSITIVO)
    modelo.load_state_dict(guardado["pesos"])
    modelo.eval()          # apaga Dropout y fija BatchNorm: obligatorio al predecir

    return modelo, guardado["media"], guardado["desviacion"], guardado["accuracy_prueba"]


def preparar_imagen(arreglo, media, desviacion):
    """Convierte una imagen de 32x32x3 en el tensor que espera la red."""
    tensor = torch.from_numpy(arreglo).permute(2, 0, 1).float() / 255.0

    media = torch.tensor(media, dtype=torch.float32).view(3, 1, 1)
    desviacion = torch.tensor(desviacion, dtype=torch.float32).view(3, 1, 1)

    return (tensor - media) / desviacion


def predecir(modelo, tensores):
    """Devuelve las probabilidades de las 43 clases para un lote de imagenes."""
    with torch.no_grad():
        salida = modelo(torch.stack(tensores).to(DISPOSITIVO))
        return F.softmax(salida, dim=1).cpu().numpy()


# ==============================================================================
# SALIDA EN CONSOLA
# ==============================================================================

def mostrar_prediccion(numero, probabilidades, clase_real=None):
    """Imprime las 3 clases mas probables de una imagen."""
    orden = np.argsort(probabilidades)[::-1][:3]
    predicha = orden[0]

    print("\n " + "-" * 74)
    print(" Imagen {}".format(numero))

    if clase_real is not None:
        print("   clase real : {:>2}  {}".format(clase_real, NOMBRES_CLASES[clase_real]))

    marca = ""
    if clase_real is not None:
        marca = "   CORRECTO" if predicha == clase_real else "   <-- ERROR"

    print("   PREDICCION : {:>2}  {}{}".format(predicha, NOMBRES_CLASES[predicha], marca))
    print("   las 3 clases mas probables:")

    for posicion, clase in enumerate(orden, start=1):
        barra = "#" * int(probabilidades[clase] * 40)
        print("     {}. {:<28} {:>6.2f}%  {}".format(
            posicion, NOMBRES_CLASES[clase][:28], probabilidades[clase] * 100, barra))


def graficar_predicciones(imagenes, probabilidades, reales, nombre):
    """Deja una figura con las imagenes y lo que el modelo predijo para cada una."""
    cuantas = len(imagenes)
    columnas = min(5, cuantas)
    filas = (cuantas + columnas - 1) // columnas

    fig, ejes = plt.subplots(filas, columnas, figsize=(3.1 * columnas, 3.5 * filas))
    ejes = np.array(ejes).reshape(-1)

    for eje, imagen, probas, real in zip(ejes, imagenes, probabilidades, reales):
        predicha = int(np.argmax(probas))
        eje.imshow(imagen)
        eje.axis("off")

        if real is None:
            color = "black"
            encabezado = ""
        else:
            color = "darkgreen" if predicha == real else "darkred"
            encabezado = "real: {}\n".format(NOMBRES_CLASES[real][:22])

        eje.set_title("{}predijo: {}\n{:.1f}%".format(
            encabezado, NOMBRES_CLASES[predicha][:22], probas[predicha] * 100),
            fontsize=8, color=color)

    for eje in ejes[cuantas:]:
        eje.axis("off")

    plt.suptitle("Predicciones del modelo entrenado\n"
                 "(verde = acerto, rojo = se equivoco)", fontsize=12)
    plt.tight_layout()
    guardar(nombre)


# ==============================================================================
# MODOS DE USO
# ==============================================================================

def modo_imagen_propia(modelo, media, desviacion, ruta):
    """Clasifica una fotografia que el usuario pasa por la linea de comandos."""
    if not os.path.exists(ruta):
        raise SystemExit(" No se encontro el archivo: {}".format(ruta))

    with Image.open(ruta) as imagen:
        imagen = imagen.convert("RGB").resize((32, 32), Image.BILINEAR)
        arreglo = np.asarray(imagen, dtype=np.uint8)

    print("\n Archivo: {}".format(ruta))

    probabilidades = predecir(modelo, [preparar_imagen(arreglo, media, desviacion)])[0]
    mostrar_prediccion(1, probabilidades)

    graficar_predicciones([arreglo], [probabilidades], [None], "predicciones.png")


def modo_conjunto_prueba(modelo, media, desviacion, cuantas, dudosas):
    """Clasifica imagenes del conjunto de prueba oficial."""
    _, _, _, _, X_test, y_test = preparar_datos.cargar()

    if dudosas:
        # Se clasifica todo el conjunto y se rescatan los casos de menor confianza.
        print("\n Buscando los casos donde el modelo esta menos seguro...")

        todas = []
        for inicio in range(0, len(X_test), 512):
            trozo = X_test[inicio:inicio + 512]
            tensores = [preparar_imagen(imagen, media, desviacion) for imagen in trozo]
            todas.append(predecir(modelo, tensores))
        todas = np.concatenate(todas)

        confianza = todas.max(axis=1)
        indices = np.argsort(confianza)[:cuantas]
        probabilidades = todas[indices]

        print(" Estas son las {} predicciones menos seguras de las {:,} imagenes".format(
            cuantas, len(X_test)))
        print(" de prueba. En un sistema real serian las que conviene mandar a")
        print(" revision humana en vez de decidir automaticamente.")
    else:
        generador = np.random.default_rng()
        indices = generador.choice(len(X_test), cuantas, replace=False)
        tensores = [preparar_imagen(X_test[i], media, desviacion) for i in indices]
        probabilidades = predecir(modelo, tensores)

    aciertos = 0
    for posicion, indice in enumerate(indices):
        mostrar_prediccion(posicion + 1, probabilidades[posicion], int(y_test[indice]))
        if int(np.argmax(probabilidades[posicion])) == int(y_test[indice]):
            aciertos += 1

    print("\n " + "=" * 74)
    print(" Acertadas {} de {} ({:.0f}%)".format(aciertos, cuantas, 100 * aciertos / cuantas))

    graficar_predicciones([X_test[i] for i in indices], probabilidades,
                          [int(y_test[i]) for i in indices], "predicciones.png")


# ==============================================================================
# PROGRAMA PRINCIPAL
# ==============================================================================

def main():
    analizador = argparse.ArgumentParser(
        description="Clasifica señales de transito con el modelo entrenado.")
    analizador.add_argument("--imagen", type=str, default=None,
                            help="ruta de una imagen propia a clasificar")
    analizador.add_argument("--cuantas", type=int, default=10,
                            help="cuantas imagenes del conjunto de prueba clasificar")
    analizador.add_argument("--dudosas", action="store_true",
                            help="mostrar las predicciones menos seguras del modelo")
    argumentos = analizador.parse_args()

    modelo, media, desviacion, accuracy = cargar_modelo()

    print("=" * 76)
    print(" CLASIFICADOR DE SEÑALES DE TRANSITO")
    print("=" * 76)
    print(" Modelo    : CNN entrenada sobre GTSRB (43 clases)")
    print(" Dispositivo: {}".format(DISPOSITIVO))
    print(" Accuracy del modelo en el conjunto de prueba: {:.2f}%".format(accuracy * 100))

    if argumentos.imagen:
        modo_imagen_propia(modelo, media, desviacion, argumentos.imagen)
    else:
        modo_conjunto_prueba(modelo, media, desviacion, argumentos.cuantas, argumentos.dudosas)

    print()


if __name__ == "__main__":
    main()

"""
Predicciones desde la consola con el modelo ya entrenado.

Carga modelo_gtsrb.pt (lo genera main.py), toma imagenes del conjunto de prueba
y muestra las tres clases mas probables de cada una.

Ejecutar: python3 predecir.py                 10 imagenes al azar
          python3 predecir.py --cuantas 25    25 imagenes
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

import preparar_datos
from preparar_datos import NOMBRES_CLASES
from main import ModeloMejorado, DISPOSITIVO, RUTA_MODELO, guardar


def cargar_modelo():
    if not os.path.exists(RUTA_MODELO):
        raise SystemExit("\n No existe modelo_gtsrb.pt. Primero corre: python3 main.py\n")

    guardado = torch.load(RUTA_MODELO, map_location=DISPOSITIVO, weights_only=False)

    modelo = ModeloMejorado().to(DISPOSITIVO)
    modelo.load_state_dict(guardado["pesos"])
    modelo.eval()  # apaga el dropout y fija BatchNorm: obligatorio al predecir

    return modelo, guardado["media"], guardado["desviacion"], guardado["accuracy_prueba"]


def preparar_imagen(arreglo, media, desviacion):
    # Se normaliza con la misma media y desviacion con que se entreno.
    tensor = torch.from_numpy(arreglo).permute(2, 0, 1).float() / 255.0
    media = torch.tensor(media, dtype=torch.float32).view(3, 1, 1)
    desviacion = torch.tensor(desviacion, dtype=torch.float32).view(3, 1, 1)
    return (tensor - media) / desviacion


def mostrar_prediccion(numero, probabilidades, clase_real):
    # softmax da la probabilidad de las 43 clases; se muestran las 3 mayores.
    orden = np.argsort(probabilidades)[::-1][:3]
    predicha = orden[0]

    print("\n " + "-" * 70)
    print(" Imagen", numero)
    print("   clase real : {:>2}  {}".format(clase_real, NOMBRES_CLASES[clase_real]))
    print("   PREDICCION : {:>2}  {}   {}".format(
        predicha, NOMBRES_CLASES[predicha],
        "CORRECTO" if predicha == clase_real else "<-- ERROR"))
    print("   las 3 mas probables:")

    for posicion, clase in enumerate(orden, start=1):
        barra = "#" * int(probabilidades[clase] * 40)
        print("     {}. {:<28} {:>6.2f}%  {}".format(
            posicion, NOMBRES_CLASES[clase][:28], probabilidades[clase] * 100, barra))


def graficar_predicciones(imagenes, probabilidades, reales):
    columnas = min(5, len(imagenes))
    filas = (len(imagenes) + columnas - 1) // columnas

    fig, ejes = plt.subplots(filas, columnas, figsize=(3.1 * columnas, 3.5 * filas))
    ejes = np.array(ejes).reshape(-1)

    for eje, imagen, probas, real in zip(ejes, imagenes, probabilidades, reales):
        predicha = int(np.argmax(probas))
        eje.imshow(imagen)
        eje.axis("off")
        eje.set_title("real: {}\npredijo: {}\n{:.1f}%".format(
            NOMBRES_CLASES[real][:22], NOMBRES_CLASES[predicha][:22],
            probas[predicha] * 100), fontsize=8,
            color="darkgreen" if predicha == real else "darkred")

    for eje in ejes[len(imagenes):]:
        eje.axis("off")

    plt.suptitle("Predicciones (verde = acerto, rojo = se equivoco)", fontsize=12)
    plt.tight_layout()
    guardar("predicciones.png")


analizador = argparse.ArgumentParser()
analizador.add_argument("--cuantas", type=int, default=10)
argumentos = analizador.parse_args()

modelo, media, desviacion, accuracy = cargar_modelo()

print("=" * 72)
print(" CLASIFICADOR DE SEÑALES DE TRANSITO")
print("=" * 72)
print(" CNN entrenada sobre GTSRB (43 clases) |", DISPOSITIVO)
print(" Accuracy en el conjunto de prueba:", round(accuracy * 100, 2), "%")

_, _, _, _, X_test, y_test = preparar_datos.cargar()

# Sin semilla: cada corrida toma imagenes distintas.
indices = np.random.default_rng().choice(len(X_test), argumentos.cuantas, replace=False)
tensores = [preparar_imagen(X_test[i], media, desviacion) for i in indices]

with torch.no_grad():
    probabilidades = F.softmax(modelo(torch.stack(tensores).to(DISPOSITIVO)), dim=1).cpu().numpy()

aciertos = 0
for posicion, indice in enumerate(indices):
    mostrar_prediccion(posicion + 1, probabilidades[posicion], int(y_test[indice]))
    if int(np.argmax(probabilidades[posicion])) == int(y_test[indice]):
        aciertos += 1

print("\n " + "=" * 70)
print(" Acertadas {} de {}".format(aciertos, argumentos.cuantas))

graficar_predicciones([X_test[i] for i in indices], probabilidades,
                      [int(y_test[i]) for i in indices])

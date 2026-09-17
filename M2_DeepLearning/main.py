"""
================================================================================
 RED NEURONAL CONVOLUCIONAL PARA RECONOCIMIENTO DE SEÑALES DE TRANSITO
================================================================================

 Modulo 2 - Portafolio de Implementacion (Deep Learning)
 Santiago Serrano Montalvo - A01751347

 PROBLEMA
 --------
 Clasificar en cual de 43 señales de transito corresponde una fotografia
 tomada desde un auto en movimiento. Es el problema de percepcion mas basico
 de un sistema de asistencia al conductor: si el vehiculo no lee la señal, no
 puede avisar que el limite bajo a 30 ni que viene un alto.

 DATASET
 -------
 GTSRB (German Traffic Sign Recognition Benchmark). Fotografias reales
 capturadas en carreteras de Alemania. Ver preparar_datos.py para el detalle.

     Entrenamiento  31,380 imagenes   (pistas de entrenamiento)
     Validacion      7,829 imagenes   (pistas separadas, ninguna compartida)
     Prueba         12,630 imagenes   (conjunto oficial del benchmark)

 POR QUE UNA RED CONVOLUCIONAL Y NO UN PERCEPTRON
 ------------------------------------------------
 Una imagen de 32x32 en color tiene 3,072 numeros. Una capa densa conectada a
 todos ellos tendria que aprender por separado que un borde rojo en la esquina
 superior izquierda significa lo mismo que un borde rojo en el centro.

 La convolucion resuelve esto con dos propiedades:

   PESOS COMPARTIDOS   un filtro de 3x3 se desliza por toda la imagen. Los
                       mismos 9 pesos detectan el borde este donde este, asi
                       que el modelo aprende la forma una sola vez.
   LOCALIDAD           cada neurona solo mira una vecindad pequeña. Al apilar
                       capas, el campo receptivo crece: la primera capa ve
                       bordes, la segunda esquinas y arcos, la tercera la
                       silueta completa del triangulo o del circulo.

 Esto es lo que hace que la arquitectura sea PROFUNDA en el sentido util: la
 jerarquia de representaciones se construye sola, capa sobre capa.

 LOS DOS MODELOS QUE SE COMPARAN
 -------------------------------
 MODELO A - aproximacion inicial
     CNN sencilla, sin ninguna tecnica de regularizacion y sin aumento de
     datos. Sirve para medir el punto de partida y ver que problema aparece.

 MODELO B - version mejorada
     Sobre la misma idea se agregan, una por una, las tecnicas que atacan el
     problema que MODELO A deja ver. Cada una esta justificada en el codigo
     y su efecto se mide en el reporte.

 EJECUCION
 ---------
       python3 main.py                  entrena los dos modelos y genera todo
       python3 main.py --epocas 5       corrida corta para probar que funciona

 Al terminar deja el modelo entrenado en modelo_gtsrb.pt, listo para que
 predecir.py lo use desde la consola.

================================================================================
"""

import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix, classification_report, f1_score

import preparar_datos
from preparar_datos import NOMBRES_CLASES


# ==============================================================================
# CONFIGURACION
# ==============================================================================

CARPETA = os.path.dirname(os.path.abspath(__file__))
CARPETA_FIGURAS = os.path.join(CARPETA, "figuras")
RUTA_MODELO = os.path.join(CARPETA, "modelo_gtsrb.pt")
RUTA_HISTORIAL = os.path.join(CARPETA, "historial.json")

N_CLASES = 43
LOTE = 128
SEMILLA = 42

# Paciencia del early stopping: epocas seguidas sin mejorar la validacion
# antes de cortar. Evita seguir entrenando cuando el modelo ya solo memoriza.
PACIENCIA = 8


def elegir_dispositivo():
    """Usa la GPU integrada de Apple (MPS) si esta disponible; si no, la CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DISPOSITIVO = elegir_dispositivo()


def guardar(nombre):
    if not os.path.isdir(CARPETA_FIGURAS):
        os.makedirs(CARPETA_FIGURAS)
    ruta = os.path.join(CARPETA_FIGURAS, nombre)
    plt.savefig(ruta, dpi=140, bbox_inches="tight")
    plt.close()
    print("   figura guardada: figuras/" + nombre)


def titulo(texto):
    print("\n" + "=" * 86)
    print(" " + texto)
    print("=" * 86)


# ==============================================================================
# 1. DATOS
# ==============================================================================

class DatasetSenales(Dataset):
    """Envuelve los arreglos de numpy para que PyTorch pueda iterarlos por lotes.

    Las imagenes se guardan como enteros de 0 a 255 para ocupar poca memoria y
    se convierten a flotantes normalizados al momento de entregar cada muestra.

    Si `aumentar` es True se aplica el aumento de datos descrito mas abajo. El
    aumento SOLO se activa en entrenamiento: validacion y prueba deben medirse
    siempre sobre la imagen original, sin alterar.
    """

    def __init__(self, imagenes, clases, media, desviacion, aumentar=False):
        self.imagenes = imagenes
        self.clases = clases.astype(np.int64)
        self.media = torch.tensor(media, dtype=torch.float32).view(3, 1, 1)
        self.desviacion = torch.tensor(desviacion, dtype=torch.float32).view(3, 1, 1)
        self.aumentar = aumentar

    def __len__(self):
        return len(self.imagenes)

    def __getitem__(self, indice):
        imagen = torch.from_numpy(self.imagenes[indice]).permute(2, 0, 1).float() / 255.0

        if self.aumentar:
            imagen = aplicar_aumento(imagen)

        imagen = (imagen - self.media) / self.desviacion
        return imagen, self.clases[indice]


def aplicar_aumento(imagen):
    """Aumento de datos pensado especificamente para señales de transito.

    QUE SE HACE Y POR QUE
    ---------------------
    ROTACION +-12 grados, TRASLACION +-10%, ESCALA 0.9 a 1.1
        Reproduce que la camara nunca ve la señal perfectamente encuadrada:
        el auto se acerca (escala), la señal aparece descentrada (traslacion)
        y el poste puede estar ligeramente inclinado (rotacion).

    BRILLO y CONTRASTE +-30%
        Reproduce sol de frente, sombra de un arbol, tunel y dia nublado, que
        es la fuente de variacion mas grande del dataset real.

    QUE NO SE HACE Y POR QUE
    ------------------------
    NO se espeja la imagen horizontalmente. Es el aumento por defecto en casi
    cualquier problema de vision, pero aqui seria un error grave: espejar
    "curva peligrosa a la izquierda" produce exactamente la señal de "curva
    peligrosa a la derecha", que es OTRA clase del dataset. Lo mismo pasa con
    los giros obligatorios y con "circule por la derecha / izquierda". El
    modelo terminaria entrenandose con etiquetas equivocadas.

    Este es el punto donde el conocimiento del problema manda sobre la receta
    generica de aumento de datos.
    """
    # --- transformacion geometrica, hecha con una matriz afin 2x3 ---
    angulo = np.random.uniform(-12, 12) * np.pi / 180.0
    escala = np.random.uniform(0.9, 1.1)
    corrimiento_x = np.random.uniform(-0.1, 0.1)
    corrimiento_y = np.random.uniform(-0.1, 0.1)

    coseno = np.cos(angulo) / escala
    seno = np.sin(angulo) / escala

    matriz = torch.tensor([[coseno, -seno, corrimiento_x],
                           [seno, coseno, corrimiento_y]], dtype=torch.float32)

    rejilla = F.affine_grid(matriz.unsqueeze(0), (1, 3, 32, 32), align_corners=False)
    imagen = F.grid_sample(imagen.unsqueeze(0), rejilla,
                           padding_mode="border", align_corners=False).squeeze(0)

    # --- brillo y contraste ---
    imagen = imagen * np.random.uniform(0.7, 1.3)                       # brillo
    promedio = imagen.mean()
    imagen = (imagen - promedio) * np.random.uniform(0.7, 1.3) + promedio  # contraste

    return imagen.clamp(0.0, 1.0)


def estadisticas(imagenes):
    """Media y desviacion de cada canal de color, calculadas SOLO con entrenamiento."""
    muestra = imagenes.astype(np.float32) / 255.0
    return muestra.mean(axis=(0, 1, 2)), muestra.std(axis=(0, 1, 2))


# ==============================================================================
# 2. ARQUITECTURAS
# ==============================================================================

class ModeloBase(nn.Module):
    """MODELO A - aproximacion inicial, deliberadamente sin regularizacion.

        entrada 3x32x32
          bloque 1:  Conv 3->32   ReLU   Conv 32->32   ReLU   MaxPool  -> 32x16x16
          bloque 2:  Conv 32->64  ReLU   Conv 64->64   ReLU   MaxPool  -> 64x8x8
          aplanado (4096)  ->  Densa 512  ReLU  ->  Densa 43

    No lleva BatchNorm, ni Dropout, ni weight decay, ni aumento de datos.
    Es el punto de partida contra el cual se mide todo lo demas.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.densa1 = nn.Linear(64 * 8 * 8, 512)
        self.densa2 = nn.Linear(512, N_CLASES)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.densa1(x))
        return self.densa2(x)


class ModeloMejorado(nn.Module):
    """MODELO B - version mejorada. Cada cambio ataca un problema concreto.

        entrada 3x32x32
          bloque 1:  [Conv 3->32  + BN + ReLU] x2   MaxPool   Dropout(0.2) -> 32x16x16
          bloque 2:  [Conv 32->64 + BN + ReLU] x2   MaxPool   Dropout(0.3) -> 64x8x8
          bloque 3:  [Conv 64->128+ BN + ReLU] x2   MaxPool   Dropout(0.4) -> 128x4x4
          GlobalAvgPool (128)  ->  Densa 128  BN  ReLU  Dropout(0.5)  ->  Densa 43

    CAMBIO 1 - BatchNorm despues de cada convolucion
        Normaliza las activaciones de cada lote. Estabiliza el entrenamiento,
        permite una tasa de aprendizaje mas alta y agrega un ruido leve que
        por si mismo ya regulariza.

    CAMBIO 2 - Dropout creciente (0.2 / 0.3 / 0.4 / 0.5)
        Apaga neuronas al azar en cada paso, asi ninguna se vuelve
        indispensable y la red se ve obligada a repartir la representacion.
        Se usa poco al principio (las primeras capas detectan bordes, que
        siempre hacen falta) y mucho al final, que es donde se memoriza.

    CAMBIO 3 - un tercer bloque convolucional
        Sube el campo receptivo y deja que la red arme la silueta completa de
        la señal, no solo sus bordes.

    CAMBIO 4 - Global Average Pooling en lugar de aplanar
        Aplanar 64x8x8 hacia una densa de 512 costaba 2.1 millones de pesos,
        casi todos los del MODELO A, y es justo donde ocurre la memorizacion.
        Promediar cada mapa de activacion deja 128 numeros: la parte densa
        pasa a ser minuscula y el modelo se apoya en las convoluciones.
    """

    def __init__(self):
        super().__init__()

        def bloque(entrada, salida):
            return nn.Sequential(
                nn.Conv2d(entrada, salida, 3, padding=1, bias=False),
                nn.BatchNorm2d(salida),
                nn.ReLU(inplace=True),
                nn.Conv2d(salida, salida, 3, padding=1, bias=False),
                nn.BatchNorm2d(salida),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.bloque1 = bloque(3, 32)
        self.drop1 = nn.Dropout(0.2)
        self.bloque2 = bloque(32, 64)
        self.drop2 = nn.Dropout(0.3)
        self.bloque3 = bloque(64, 128)
        self.drop3 = nn.Dropout(0.4)

        self.densa1 = nn.Linear(128, 128)
        self.norma = nn.BatchNorm1d(128)
        self.drop4 = nn.Dropout(0.5)
        self.densa2 = nn.Linear(128, N_CLASES)

    def forward(self, x):
        x = self.drop1(self.bloque1(x))
        x = self.drop2(self.bloque2(x))
        x = self.drop3(self.bloque3(x))

        x = F.adaptive_avg_pool2d(x, 1).flatten(1)   # global average pooling

        x = self.drop4(F.relu(self.norma(self.densa1(x))))
        return self.densa2(x)


def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


# ==============================================================================
# 3. ENTRENAMIENTO
# ==============================================================================

def una_pasada(modelo, cargador, criterio, optimizador=None):
    """Recorre el cargador una vez. Si hay optimizador, entrena; si no, evalua."""
    entrenando = optimizador is not None
    modelo.train() if entrenando else modelo.eval()

    perdida_total = 0.0
    aciertos = 0
    total = 0

    with torch.set_grad_enabled(entrenando):
        for imagenes, clases in cargador:
            imagenes = imagenes.to(DISPOSITIVO, non_blocking=True)
            clases = clases.to(DISPOSITIVO, non_blocking=True)

            salida = modelo(imagenes)
            perdida = criterio(salida, clases)

            if entrenando:
                optimizador.zero_grad(set_to_none=True)
                perdida.backward()
                optimizador.step()

            perdida_total += perdida.item() * len(clases)
            aciertos += (salida.argmax(1) == clases).sum().item()
            total += len(clases)

    return perdida_total / total, aciertos / total


def entrenar(modelo, nombre, cargador_train, cargador_val, epocas,
             tasa=1e-3, weight_decay=0.0, usar_scheduler=False, early_stopping=False):
    """Entrena un modelo y devuelve el historial epoca por epoca.

    tasa             tasa de aprendizaje inicial del optimizador Adam.
    weight_decay     penalizacion L2 sobre los pesos. Empuja los pesos hacia
                     cero y con eso limita cuanto puede especializarse la red.
    usar_scheduler   baja la tasa de aprendizaje siguiendo un coseno. Al
                     principio conviene avanzar rapido y al final afinar con
                     pasos chicos para asentarse en el minimo.
    early_stopping   guarda los pesos de la mejor epoca en validacion y corta
                     si pasan PACIENCIA epocas sin mejorar. Es la
                     regularizacion mas directa: detiene el entrenamiento
                     justo antes de que empiece a memorizar.
    """
    modelo = modelo.to(DISPOSITIVO)
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=tasa, weight_decay=weight_decay)

    planificador = None
    if usar_scheduler:
        planificador = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=epocas)

    historial = {"perdida_train": [], "perdida_val": [],
                 "acc_train": [], "acc_val": [], "tasa": []}

    mejor_acc = 0.0
    mejores_pesos = None
    sin_mejorar = 0

    print("\n {:<7} {:>12} {:>11} {:>12} {:>11} {:>10} {:>8}".format(
        "epoca", "perdida tr", "acc tr", "perdida val", "acc val", "tasa", "seg"))
    print(" " + "-" * 82)

    for epoca in range(1, epocas + 1):
        inicio = time.time()

        perdida_tr, acc_tr = una_pasada(modelo, cargador_train, criterio, optimizador)
        perdida_va, acc_va = una_pasada(modelo, cargador_val, criterio)

        tasa_actual = optimizador.param_groups[0]["lr"]
        if planificador is not None:
            planificador.step()

        historial["perdida_train"].append(perdida_tr)
        historial["perdida_val"].append(perdida_va)
        historial["acc_train"].append(acc_tr)
        historial["acc_val"].append(acc_va)
        historial["tasa"].append(tasa_actual)

        marca = ""
        if acc_va > mejor_acc:
            mejor_acc = acc_va
            mejores_pesos = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
            sin_mejorar = 0
            marca = "  *mejor"
        else:
            sin_mejorar += 1

        print(" {:<7} {:>12.4f} {:>10.2f}% {:>12.4f} {:>10.2f}% {:>10.5f} {:>8.1f}{}".format(
            epoca, perdida_tr, acc_tr * 100, perdida_va, acc_va * 100,
            tasa_actual, time.time() - inicio, marca))

        if early_stopping and sin_mejorar >= PACIENCIA:
            print("\n Early stopping: {} epocas sin mejorar la validacion.".format(PACIENCIA))
            print(" Se conservan los pesos de la mejor epoca (acc val {:.2f}%).".format(
                mejor_acc * 100))
            break

    # Con early stopping se devuelven los mejores pesos, no los ultimos.
    if early_stopping and mejores_pesos is not None:
        modelo.load_state_dict(mejores_pesos)

    historial["nombre"] = nombre
    historial["mejor_acc_val"] = mejor_acc
    return modelo, historial


# ==============================================================================
# 4. EVALUACION
# ==============================================================================

def predecir_conjunto(modelo, cargador):
    """Devuelve las clases reales, las predichas y las probabilidades."""
    modelo.eval()
    reales, predichas, probabilidades = [], [], []

    with torch.no_grad():
        for imagenes, clases in cargador:
            salida = modelo(imagenes.to(DISPOSITIVO))
            probas = F.softmax(salida, dim=1).cpu().numpy()

            reales.append(clases.numpy())
            predichas.append(probas.argmax(1))
            probabilidades.append(probas)

    return (np.concatenate(reales), np.concatenate(predichas),
            np.concatenate(probabilidades))


def evaluar(modelo, cargador, nombre):
    reales, predichas, _ = predecir_conjunto(modelo, cargador)

    acc = (reales == predichas).mean()
    f1m = f1_score(reales, predichas, average="macro", zero_division=0)

    print("   {:<34} accuracy {:>7.2f}%   F1 macro {:.4f}".format(nombre, acc * 100, f1m))
    return {"accuracy": acc, "f1_macro": f1m, "reales": reales, "predichas": predichas}


# ==============================================================================
# 5. GRAFICAS
# ==============================================================================

def graficar_muestra(X, y, nombre):
    """Muestra imagenes del dataset para dejar ver como son los datos reales."""
    generador = np.random.default_rng(SEMILLA)
    indices = generador.choice(len(X), 24, replace=False)

    fig, ejes = plt.subplots(3, 8, figsize=(15, 7.8))
    for eje, indice in zip(ejes.ravel(), indices):
        eje.imshow(X[indice])
        eje.set_title("clase {}\n{}".format(y[indice], NOMBRES_CLASES[y[indice]][:20]),
                      fontsize=7.5, pad=5)
        eje.axis("off")

    plt.suptitle("GTSRB: fotografias reales tomadas desde un auto en movimiento\n"
                 "(desenfoque, sombras, contraluz y encuadres distintos)", fontsize=12)
    plt.tight_layout(h_pad=2.2)
    guardar(nombre)


def graficar_aumento(X, media, desviacion, nombre):
    """Compara una imagen original con varias versiones aumentadas."""
    generador = np.random.default_rng(7)
    indices = generador.choice(len(X), 4, replace=False)

    fig, ejes = plt.subplots(4, 7, figsize=(13, 7.6))

    for fila, indice in enumerate(indices):
        original = torch.from_numpy(X[indice]).permute(2, 0, 1).float() / 255.0

        ejes[fila, 0].imshow(original.permute(1, 2, 0).numpy())
        ejes[fila, 0].set_title("original", fontsize=9)
        ejes[fila, 0].axis("off")

        for columna in range(1, 7):
            aumentada = aplicar_aumento(original.clone())
            ejes[fila, columna].imshow(aumentada.permute(1, 2, 0).numpy())
            ejes[fila, columna].set_title("aumentada", fontsize=8)
            ejes[fila, columna].axis("off")

    plt.suptitle("Aumento de datos: rotacion, traslacion, escala, brillo y contraste\n"
                 "(NO se espeja la imagen: convertiria 'curva izquierda' en 'curva derecha')",
                 fontsize=12)
    plt.tight_layout()
    guardar(nombre)


def graficar_entrenamiento(historiales, nombre):
    """Curvas de perdida y accuracy de los dos modelos, entrenamiento vs validacion."""
    fig, ejes = plt.subplots(2, 2, figsize=(13.5, 9))

    for columna, historial in enumerate(historiales):
        epocas = range(1, len(historial["perdida_train"]) + 1)

        eje = ejes[0, columna]
        eje.plot(epocas, historial["perdida_train"], marker="o", ms=3,
                 color="tab:blue", label="Entrenamiento")
        eje.plot(epocas, historial["perdida_val"], marker="s", ms=3,
                 color="tab:orange", label="Validacion")
        eje.fill_between(epocas, historial["perdida_train"], historial["perdida_val"],
                         color="tab:red", alpha=0.1)
        eje.set_xlabel("Epoca")
        eje.set_ylabel("Perdida (entropia cruzada)")
        eje.set_title("{}\nPerdida".format(historial["nombre"]))
        eje.legend()
        eje.grid(alpha=0.3)

        eje = ejes[1, columna]
        eje.plot(epocas, [a * 100 for a in historial["acc_train"]], marker="o", ms=3,
                 color="tab:blue", label="Entrenamiento")
        eje.plot(epocas, [a * 100 for a in historial["acc_val"]], marker="s", ms=3,
                 color="tab:orange", label="Validacion")
        eje.fill_between(epocas, [a * 100 for a in historial["acc_train"]],
                         [a * 100 for a in historial["acc_val"]], color="tab:red", alpha=0.1)

        # Se prefiere la brecha de la EVALUACION FINAL (modelo en modo eval, sobre
        # datos sin aumentar). La de la ultima epoca esta medida con dropout y
        # aumento de datos activos, asi que subestima el ajuste del modelo.
        if "brecha_evaluada" in historial:
            brecha = historial["brecha_evaluada"] * 100
            etiqueta = "brecha evaluada = {:.2f} pp".format(brecha)
        else:
            brecha = (historial["acc_train"][-1] - historial["acc_val"][-1]) * 100
            etiqueta = "brecha ultima epoca = {:.2f} pp".format(brecha)

        eje.annotate(etiqueta,
                     xy=(len(epocas), (historial["acc_train"][-1] + historial["acc_val"][-1]) * 50),
                     xytext=(-140, -30), textcoords="offset points", fontsize=9, color="tab:red",
                     arrowprops=dict(arrowstyle="->", color="tab:red"))

        eje.set_xlabel("Epoca")
        eje.set_ylabel("Accuracy (%)")
        eje.set_title("Accuracy")
        eje.set_ylim(80, 101)
        eje.legend(loc="lower right")
        eje.grid(alpha=0.3)

    plt.suptitle("Aprendizaje de los dos modelos: entrenamiento contra validacion", fontsize=13)
    plt.tight_layout()
    guardar(nombre)


def graficar_matriz(reales, predichas, nombre):
    """Matriz de confusion 43x43 normalizada por fila."""
    matriz = confusion_matrix(reales, predichas, labels=range(N_CLASES)).astype(float)
    normalizada = matriz / np.clip(matriz.sum(axis=1, keepdims=True), 1, None)

    plt.figure(figsize=(12.5, 11))
    plt.imshow(normalizada, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="proporcion de la clase real", fraction=0.046)

    etiquetas = ["{:>2} {}".format(i, NOMBRES_CLASES[i][:20]) for i in range(N_CLASES)]
    plt.xticks(range(N_CLASES), range(N_CLASES), fontsize=7)
    plt.yticks(range(N_CLASES), etiquetas, fontsize=7)

    # Solo se anotan los errores visibles, para que la figura siga siendo legible.
    for i in range(N_CLASES):
        for j in range(N_CLASES):
            if i != j and normalizada[i, j] >= 0.03:
                plt.text(j, i, "{:.0f}".format(matriz[i, j]), ha="center", va="center",
                         fontsize=6, color="darkred")

    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.title("Matriz de confusion en el conjunto de PRUEBA (12,630 imagenes, 43 clases)\n"
              "los numeros rojos marcan confusiones del 3% o mas")
    plt.tight_layout()
    guardar(nombre)


def graficar_accuracy_por_clase(reales, predichas, nombre):
    """Accuracy de cada una de las 43 clases, ordenada de peor a mejor."""
    accs, soportes = [], []
    for clase in range(N_CLASES):
        mascara = reales == clase
        soportes.append(int(mascara.sum()))
        accs.append((predichas[mascara] == clase).mean() if mascara.sum() else 0.0)

    orden = np.argsort(accs)
    colores = ["tab:red" if accs[i] < 0.90 else "tab:green" for i in orden]

    plt.figure(figsize=(9, 12))
    plt.barh(range(N_CLASES), [accs[i] * 100 for i in orden], color=colores)
    plt.yticks(range(N_CLASES),
               ["{:>2} {} (n={})".format(i, NOMBRES_CLASES[i][:24], soportes[i]) for i in orden],
               fontsize=8)
    plt.xlabel("Accuracy en el conjunto de prueba (%)")
    plt.xlim(0, 105)
    plt.axvline(90, color="gray", linestyle="--", alpha=0.7)
    plt.title("Accuracy por clase, de la peor a la mejor\n(rojo = por debajo del 90%)")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    guardar(nombre)


def graficar_errores(X_test, reales, predichas, probabilidades, nombre):
    """Muestra los errores en los que el modelo estuvo mas seguro."""
    fallos = np.where(reales != predichas)[0]

    if len(fallos) == 0:
        return

    confianza = probabilidades[fallos, predichas[fallos]]
    peores = fallos[np.argsort(confianza)[::-1][:18]]

    fig, ejes = plt.subplots(3, 6, figsize=(15, 8.5))
    for eje, indice in zip(ejes.ravel(), peores):
        eje.imshow(X_test[indice])
        eje.set_title("real: {}\npredijo: {}\nseguridad {:.0f}%".format(
            NOMBRES_CLASES[reales[indice]][:20],
            NOMBRES_CLASES[predichas[indice]][:20],
            probabilidades[indice, predichas[indice]] * 100), fontsize=7)
        eje.axis("off")

    for eje in ejes.ravel()[len(peores):]:
        eje.axis("off")

    plt.suptitle("Los 18 errores con mayor seguridad del modelo\n"
                 "(casi siempre imagenes muy oscuras, borrosas o de baja resolucion)",
                 fontsize=12)
    plt.tight_layout()
    guardar(nombre)


def graficar_comparacion(resultados, nombre):
    """Barras comparando MODELO A y MODELO B en validacion y prueba."""
    etiquetas = ["Validacion", "Prueba"]
    a = [resultados["A"]["val"]["accuracy"] * 100, resultados["A"]["test"]["accuracy"] * 100]
    b = [resultados["B"]["val"]["accuracy"] * 100, resultados["B"]["test"]["accuracy"] * 100]

    x = np.arange(len(etiquetas))
    ancho = 0.36

    fig, ejes = plt.subplots(1, 2, figsize=(13, 5))

    barras_a = ejes[0].bar(x - ancho / 2, a, ancho, label="MODELO A (inicial)", color="tab:red", alpha=0.85)
    barras_b = ejes[0].bar(x + ancho / 2, b, ancho, label="MODELO B (mejorado)", color="tab:green", alpha=0.85)
    for barras in (barras_a, barras_b):
        for barra in barras:
            ejes[0].text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.25,
                         "{:.2f}%".format(barra.get_height()), ha="center", fontsize=9)
    ejes[0].set_xticks(x)
    ejes[0].set_xticklabels(etiquetas)
    ejes[0].set_ylabel("Accuracy (%)")
    ejes[0].set_ylim(80, 103)
    ejes[0].set_title("Accuracy de los dos modelos")
    # La leyenda va abajo: arriba taparia las etiquetas de las barras.
    ejes[0].legend(loc="lower center", fontsize=9)
    ejes[0].grid(axis="y", alpha=0.3)

    brechas = [resultados["A"]["brecha"] * 100, resultados["B"]["brecha"] * 100]
    barras = ejes[1].bar(["MODELO A", "MODELO B"], brechas,
                         color=["tab:red", "tab:green"], alpha=0.85, width=0.5)
    for barra in barras:
        ejes[1].text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.05,
                     "{:.2f} pp".format(barra.get_height()), ha="center", fontsize=10)
    ejes[1].set_ylabel("Brecha accuracy entrenamiento - validacion (puntos porcentuales)")
    ejes[1].set_title("Sobreajuste: cuanto mejor le va al modelo\nen lo que ya vio que en datos nuevos")
    ejes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    guardar(nombre)


# ==============================================================================
# PROGRAMA PRINCIPAL
# ==============================================================================

def main():
    analizador = argparse.ArgumentParser()
    analizador.add_argument("--epocas", type=int, default=30)
    argumentos = analizador.parse_args()

    torch.manual_seed(SEMILLA)
    np.random.seed(SEMILLA)

    # ==========================================================================
    titulo("1. DATOS")
    # ==========================================================================

    X_train, y_train, X_val, y_val, X_test, y_test = preparar_datos.cargar()

    media, desviacion = estadisticas(X_train)

    print(" Dataset      : GTSRB - señales de transito alemanas (fotografias reales)")
    print(" Dispositivo  : {}".format(DISPOSITIVO))
    print(" Clases       : {}".format(N_CLASES))
    print(" Resolucion   : 32 x 32 pixeles en color")
    print("\n {:<34} {:>10}".format("conjunto", "imagenes"))
    print(" {:<34} {:>10}".format("Entrenamiento", len(X_train)))
    print(" {:<34} {:>10}".format("Validacion (pistas separadas)", len(X_val)))
    print(" {:<34} {:>10}".format("Prueba (conjunto oficial GTSRB)", len(X_test)))

    print("\n Normalizacion por canal, calculada solo con entrenamiento:")
    print("   media      R {:.4f}  G {:.4f}  B {:.4f}".format(*media))
    print("   desviacion R {:.4f}  G {:.4f}  B {:.4f}".format(*desviacion))

    cuentas = np.bincount(y_train, minlength=N_CLASES)
    print("\n Clases con menos ejemplos: ", end="")
    for clase in np.argsort(cuentas)[:4]:
        print("{} ({}) ".format(NOMBRES_CLASES[clase][:18], cuentas[clase]), end="")
    print("\n Clases con mas ejemplos  : ", end="")
    for clase in np.argsort(cuentas)[::-1][:3]:
        print("{} ({}) ".format(NOMBRES_CLASES[clase][:18], cuentas[clase]), end="")
    print()

    cargador_train_simple = DataLoader(
        DatasetSenales(X_train, y_train, media, desviacion, aumentar=False),
        batch_size=LOTE, shuffle=True)
    cargador_train_aumentado = DataLoader(
        DatasetSenales(X_train, y_train, media, desviacion, aumentar=True),
        batch_size=LOTE, shuffle=True)
    cargador_train_eval = DataLoader(
        DatasetSenales(X_train, y_train, media, desviacion, aumentar=False),
        batch_size=256, shuffle=False)
    cargador_val = DataLoader(
        DatasetSenales(X_val, y_val, media, desviacion, aumentar=False),
        batch_size=256, shuffle=False)
    cargador_test = DataLoader(
        DatasetSenales(X_test, y_test, media, desviacion, aumentar=False),
        batch_size=256, shuffle=False)

    graficar_muestra(X_train, y_train, "muestra_dataset.png")
    graficar_aumento(X_train, media, desviacion, "aumento_datos.png")

    resultados = {}

    # ==========================================================================
    titulo("2. MODELO A - APROXIMACION INICIAL")
    # ==========================================================================

    modelo_a = ModeloBase()
    print(" Arquitectura : CNN de 4 convoluciones + 2 capas densas")
    print(" Parametros   : {:,}".format(contar_parametros(modelo_a)))
    print(" Optimizador  : Adam, tasa fija 1e-3")
    print(" Regularizacion: NINGUNA (sin BatchNorm, sin Dropout, sin weight decay)")
    print(" Aumento de datos: NO")

    modelo_a, historial_a = entrenar(
        modelo_a, "MODELO A - inicial, sin regularizar",
        cargador_train_simple, cargador_val, argumentos.epocas)

    print("\n Evaluacion del MODELO A:")
    a_train = evaluar(modelo_a, cargador_train_eval, "entrenamiento")
    a_val = evaluar(modelo_a, cargador_val, "validacion")
    a_test = evaluar(modelo_a, cargador_test, "prueba")

    brecha_a = a_train["accuracy"] - a_val["accuracy"]
    print("\n   Brecha entrenamiento - validacion: {:.2f} puntos porcentuales".format(brecha_a * 100))
    print("   El modelo llega a {:.2f}% en lo que ya vio y baja a {:.2f}% en datos".format(
        a_train["accuracy"] * 100, a_val["accuracy"] * 100))
    print("   nuevos: esta MEMORIZANDO. Ese es el problema que ataca el MODELO B.")

    resultados["A"] = {"train": a_train, "val": a_val, "test": a_test, "brecha": brecha_a}
    historial_a["brecha_evaluada"] = brecha_a

    # ==========================================================================
    titulo("3. MODELO B - VERSION MEJORADA")
    # ==========================================================================

    modelo_b = ModeloMejorado()
    print(" Cambios respecto al MODELO A:")
    print("   1. BatchNorm despues de cada convolucion")
    print("   2. Dropout creciente (0.2 / 0.3 / 0.4 / 0.5)")
    print("   3. Un tercer bloque convolucional (32 -> 64 -> 128 canales)")
    print("   4. Global Average Pooling en vez de aplanar hacia una densa de 512")
    print("   5. Aumento de datos: rotacion, traslacion, escala, brillo y contraste")
    print("   6. Weight decay 1e-4 (penalizacion L2 sobre los pesos)")
    print("   7. Tasa de aprendizaje con decaimiento coseno")
    print("   8. Early stopping con paciencia de {} epocas".format(PACIENCIA))
    print("\n Parametros   : {:,}  ({:.1f}x menos que el MODELO A)".format(
        contar_parametros(modelo_b),
        contar_parametros(modelo_a) / contar_parametros(modelo_b)))

    modelo_b, historial_b = entrenar(
        modelo_b, "MODELO B - mejorado y regularizado",
        cargador_train_aumentado, cargador_val, argumentos.epocas,
        tasa=2e-3, weight_decay=1e-4, usar_scheduler=True, early_stopping=True)

    print("\n Evaluacion del MODELO B:")
    b_train = evaluar(modelo_b, cargador_train_eval, "entrenamiento")
    b_val = evaluar(modelo_b, cargador_val, "validacion")
    b_test = evaluar(modelo_b, cargador_test, "prueba")

    brecha_b = b_train["accuracy"] - b_val["accuracy"]
    print("\n   Brecha entrenamiento - validacion: {:.2f} puntos porcentuales".format(brecha_b * 100))

    resultados["B"] = {"train": b_train, "val": b_val, "test": b_test, "brecha": brecha_b}
    historial_b["brecha_evaluada"] = brecha_b

    # ==========================================================================
    titulo("4. COMPARACION DE LOS DOS MODELOS")
    # ==========================================================================

    print(" {:<38} {:>12} {:>12} {:>10}".format("", "MODELO A", "MODELO B", "cambio"))
    filas = [
        ("Parametros entrenables", contar_parametros(modelo_a), contar_parametros(modelo_b), "n"),
        ("Accuracy entrenamiento (%)", a_train["accuracy"] * 100, b_train["accuracy"] * 100, "f"),
        ("Accuracy validacion (%)", a_val["accuracy"] * 100, b_val["accuracy"] * 100, "f"),
        ("Accuracy PRUEBA (%)", a_test["accuracy"] * 100, b_test["accuracy"] * 100, "f"),
        ("F1 macro PRUEBA", a_test["f1_macro"], b_test["f1_macro"], "f4"),
        ("Brecha train - val (pp)", brecha_a * 100, brecha_b * 100, "f"),
    ]
    for etiqueta, va, vb, tipo in filas:
        if tipo == "n":
            print(" {:<38} {:>12,} {:>12,} {:>9.1f}x".format(etiqueta, va, vb, va / vb))
        elif tipo == "f4":
            print(" {:<38} {:>12.4f} {:>12.4f} {:>+10.4f}".format(etiqueta, va, vb, vb - va))
        else:
            print(" {:<38} {:>12.2f} {:>12.2f} {:>+10.2f}".format(etiqueta, va, vb, vb - va))

    errores_a = int((1 - a_test["accuracy"]) * len(y_test))
    errores_b = int((1 - b_test["accuracy"]) * len(y_test))
    print("\n Errores sobre las {:,} imagenes de prueba: {} -> {}  ({:.0f}% menos)".format(
        len(y_test), errores_a, errores_b, 100 * (errores_a - errores_b) / max(errores_a, 1)))

    # ==========================================================================
    titulo("5. ANALISIS DETALLADO DEL MODELO FINAL EN PRUEBA")
    # ==========================================================================

    reales, predichas, probabilidades = predecir_conjunto(modelo_b, cargador_test)

    print(classification_report(reales, predichas, labels=range(N_CLASES),
                                target_names=[n[:26] for n in NOMBRES_CLASES],
                                digits=3, zero_division=0))

    # ==========================================================================
    titulo("6. GRAFICAS")
    # ==========================================================================

    graficar_entrenamiento([historial_a, historial_b], "curvas_entrenamiento.png")
    graficar_comparacion(resultados, "comparacion_modelos.png")
    graficar_matriz(reales, predichas, "matriz_confusion.png")
    graficar_accuracy_por_clase(reales, predichas, "accuracy_por_clase.png")
    graficar_errores(X_test, reales, predichas, probabilidades, "errores.png")

    # ==========================================================================
    titulo("7. GUARDADO DEL MODELO")
    # ==========================================================================

    torch.save({"pesos": modelo_b.state_dict(),
                "media": media, "desviacion": desviacion,
                "accuracy_prueba": float(b_test["accuracy"])}, RUTA_MODELO)
    print(" Modelo guardado en modelo_gtsrb.pt ({:.1f} MB)".format(
        os.path.getsize(RUTA_MODELO) / 1e6))

    with open(RUTA_HISTORIAL, "w") as archivo:
        json.dump({"A": historial_a, "B": historial_b,
                   "resumen": {clave: {"accuracy_test": float(r["test"]["accuracy"]),
                                       "f1_test": float(r["test"]["f1_macro"]),
                                       "accuracy_val": float(r["val"]["accuracy"]),
                                       "brecha": float(r["brecha"])}
                               for clave, r in resultados.items()}}, archivo, indent=2)
    print(" Historial de entrenamiento guardado en historial.json")

    print("\n Para hacer predicciones desde la consola:  python3 predecir.py")
    print("\n Listo.")


if __name__ == "__main__":
    main()

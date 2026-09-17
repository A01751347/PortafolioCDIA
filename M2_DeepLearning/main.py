"""
Red neuronal convolucional para reconocer señales de transito.

Clasifica fotografias reales de 43 señales del dataset GTSRB, tomadas desde un
auto en movimiento. Entrena dos modelos: uno inicial sin regularizacion y una
version mejorada, para comparar y documentar que cambios funcionaron.

Ejecutar: python3 main.py            (o --epocas 5 para una corrida corta)
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
from torchvision import transforms

from sklearn.metrics import confusion_matrix, classification_report, f1_score

import preparar_datos
from preparar_datos import NOMBRES_CLASES


# -----------------------------
# CONFIGURACION
# -----------------------------

CARPETA = os.path.dirname(os.path.abspath(__file__))
FIGURAS = os.path.join(CARPETA, "figuras")
RUTA_MODELO = os.path.join(CARPETA, "modelo_gtsrb.pt")
RUTA_HISTORIAL = os.path.join(CARPETA, "historial.json")

N_CLASES = 43
LOTE = 128
SEMILLA = 42
PACIENCIA = 8  # epocas sin mejorar antes de cortar el entrenamiento

if torch.backends.mps.is_available():
    DISPOSITIVO = torch.device("mps")
elif torch.cuda.is_available():
    DISPOSITIVO = torch.device("cuda")
else:
    DISPOSITIVO = torch.device("cpu")


# -----------------------------
# DATOS Y AUMENTO
# -----------------------------
# Rotacion, traslacion y escala reproducen que la camara nunca ve la señal bien
# encuadrada; brillo y contraste reproducen sol de frente, sombra y tunel.
#
# NO se usa RandomHorizontalFlip, que es el aumento por defecto en vision: aqui
# espejar "curva peligrosa izquierda" produce la señal de "curva peligrosa
# derecha", que es OTRA clase del dataset. Pasaria igual con los giros
# obligatorios y con "circule por la derecha / izquierda".

AUMENTO = transforms.Compose([
    transforms.RandomAffine(degrees=12, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
])


class DatasetSenales(Dataset):
    # Las imagenes se guardan como enteros de 0 a 255 para ocupar poca memoria
    # y se convierten a flotantes normalizados al entregar cada muestra.

    def __init__(self, imagenes, clases, media, desviacion, aumentar=False):
        self.imagenes = imagenes
        self.clases = clases.astype(np.int64)
        self.media = torch.tensor(media, dtype=torch.float32).view(3, 1, 1)
        self.desviacion = torch.tensor(desviacion, dtype=torch.float32).view(3, 1, 1)
        self.aumentar = aumentar  # solo en entrenamiento

    def __len__(self):
        return len(self.imagenes)

    def __getitem__(self, indice):
        imagen = torch.from_numpy(self.imagenes[indice]).permute(2, 0, 1).float() / 255.0

        if self.aumentar:
            imagen = AUMENTO(imagen)

        return (imagen - self.media) / self.desviacion, self.clases[indice]


def estadisticas(imagenes):
    # Media y desviacion de cada canal, calculadas solo con entrenamiento.
    muestra = imagenes.astype(np.float32) / 255.0
    return muestra.mean(axis=(0, 1, 2)), muestra.std(axis=(0, 1, 2))


# -----------------------------
# LOS DOS MODELOS
# -----------------------------
# Una capa densa sobre los 3072 pixeles tendria que aprender por separado que
# un borde rojo arriba significa lo mismo que uno al centro. La convolucion no:
# el mismo filtro de 3x3 se desliza por toda la imagen, asi que la forma se
# aprende una sola vez, y al apilar capas el campo receptivo crece hasta cubrir
# la silueta completa de la señal.

class ModeloBase(nn.Module):
    # Version inicial: 4 convoluciones y 2 densas, sin ninguna regularizacion.

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
    # BatchNorm estabiliza el entrenamiento, el dropout crece hacia el final
    # (las primeras capas detectan bordes, que siempre hacen falta) y el global
    # average pooling sustituye a la densa de 512, que era donde estaban casi
    # todos los parametros y donde ocurria la memorizacion.

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
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.drop4(F.relu(self.norma(self.densa1(x))))
        return self.densa2(x)


def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


# -----------------------------
# ENTRENAMIENTO
# -----------------------------

def una_pasada(modelo, cargador, criterio, optimizador=None):
    # Con optimizador entrena, sin el solo evalua.
    entrenando = optimizador is not None
    modelo.train() if entrenando else modelo.eval()

    perdida_total = 0.0
    aciertos = 0
    total = 0

    with torch.set_grad_enabled(entrenando):
        for imagenes, clases in cargador:
            imagenes = imagenes.to(DISPOSITIVO)
            clases = clases.to(DISPOSITIVO)

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
             tasa=1e-3, weight_decay=0.0, scheduler=False, early_stopping=False):
    modelo = modelo.to(DISPOSITIVO)
    criterio = nn.CrossEntropyLoss()

    # weight_decay es la penalizacion L2: empuja los pesos hacia cero.
    optimizador = torch.optim.Adam(modelo.parameters(), lr=tasa, weight_decay=weight_decay)

    # El coseno baja la tasa poco a poco: avanzar rapido al inicio y afinar al final.
    planificador = None
    if scheduler:
        planificador = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=epocas)

    historial = {"perdida_train": [], "perdida_val": [], "acc_train": [],
                 "acc_val": [], "tasa": [], "nombre": nombre}

    mejor_acc = 0.0
    mejores_pesos = None
    sin_mejorar = 0

    print("\n {:<7} {:>12} {:>11} {:>12} {:>11} {:>10} {:>7}".format(
        "epoca", "perdida tr", "acc tr", "perdida val", "acc val", "tasa", "seg"))
    print(" " + "-" * 80)

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

        print(" {:<7} {:>12.4f} {:>10.2f}% {:>12.4f} {:>10.2f}% {:>10.5f} {:>7.1f}{}".format(
            epoca, perdida_tr, acc_tr * 100, perdida_va, acc_va * 100,
            tasa_actual, time.time() - inicio, marca))

        if early_stopping and sin_mejorar >= PACIENCIA:
            print("\n Early stopping:", PACIENCIA, "epocas sin mejorar.")
            break

    # Se conservan los pesos de la mejor epoca, no los de la ultima.
    if early_stopping and mejores_pesos is not None:
        modelo.load_state_dict(mejores_pesos)

    historial["mejor_acc_val"] = mejor_acc
    return modelo, historial


# -----------------------------
# EVALUACION
# -----------------------------

def predecir_conjunto(modelo, cargador):
    modelo.eval()
    reales, predichas, probabilidades = [], [], []

    with torch.no_grad():
        for imagenes, clases in cargador:
            probas = F.softmax(modelo(imagenes.to(DISPOSITIVO)), dim=1).cpu().numpy()
            reales.append(clases.numpy())
            predichas.append(probas.argmax(1))
            probabilidades.append(probas)

    return np.concatenate(reales), np.concatenate(predichas), np.concatenate(probabilidades)


def evaluar(modelo, cargador, nombre):
    reales, predichas, _ = predecir_conjunto(modelo, cargador)

    acc = (reales == predichas).mean()
    f1m = f1_score(reales, predichas, average="macro", zero_division=0)

    print("   {:<20} accuracy {:>7.2f}%   F1 macro {:.4f}".format(nombre, acc * 100, f1m))
    return {"accuracy": acc, "f1_macro": f1m}


# -----------------------------
# GRAFICAS
# -----------------------------

def guardar(nombre):
    if not os.path.isdir(FIGURAS):
        os.makedirs(FIGURAS)
    plt.savefig(os.path.join(FIGURAS, nombre), dpi=140, bbox_inches="tight")
    plt.close()
    print("   figura guardada: figuras/" + nombre)


def graficar_muestra(X, y):
    indices = np.random.default_rng(SEMILLA).choice(len(X), 24, replace=False)

    fig, ejes = plt.subplots(3, 8, figsize=(15, 7.8))
    for eje, indice in zip(ejes.ravel(), indices):
        eje.imshow(X[indice])
        eje.set_title("clase {}\n{}".format(y[indice], NOMBRES_CLASES[y[indice]][:20]),
                      fontsize=7.5, pad=5)
        eje.axis("off")

    plt.suptitle("GTSRB: fotografias reales tomadas desde un auto en movimiento", fontsize=12)
    plt.tight_layout(h_pad=2.2)
    guardar("muestra_dataset.png")


def graficar_aumento(X):
    indices = np.random.default_rng(7).choice(len(X), 4, replace=False)

    fig, ejes = plt.subplots(4, 7, figsize=(13, 7.6))

    for fila, indice in enumerate(indices):
        original = torch.from_numpy(X[indice]).permute(2, 0, 1).float() / 255.0

        ejes[fila, 0].imshow(original.permute(1, 2, 0).numpy())
        ejes[fila, 0].set_title("original", fontsize=9)
        ejes[fila, 0].axis("off")

        for columna in range(1, 7):
            aumentada = AUMENTO(original.clone()).clamp(0, 1)
            ejes[fila, columna].imshow(aumentada.permute(1, 2, 0).numpy())
            ejes[fila, columna].set_title("aumentada", fontsize=8)
            ejes[fila, columna].axis("off")

    plt.suptitle("Aumento de datos: rotacion, traslacion, escala, brillo y contraste\n"
                 "(sin espejado horizontal: convertiria 'curva izquierda' en 'curva derecha')",
                 fontsize=12)
    plt.tight_layout()
    guardar("aumento_datos.png")


def graficar_entrenamiento(historiales):
    fig, ejes = plt.subplots(2, 2, figsize=(13.5, 9))

    for columna, historial in enumerate(historiales):
        epocas = range(1, len(historial["perdida_train"]) + 1)

        eje = ejes[0, columna]
        eje.plot(epocas, historial["perdida_train"], marker="o", ms=3, label="Entrenamiento")
        eje.plot(epocas, historial["perdida_val"], marker="s", ms=3, label="Validacion")
        eje.fill_between(epocas, historial["perdida_train"], historial["perdida_val"],
                         color="tab:red", alpha=0.1)
        eje.set_xlabel("Epoca")
        eje.set_ylabel("Perdida")
        eje.set_title(historial["nombre"] + "\nPerdida")
        eje.legend()
        eje.grid(alpha=0.3)

        eje = ejes[1, columna]
        eje.plot(epocas, [a * 100 for a in historial["acc_train"]], marker="o", ms=3,
                 label="Entrenamiento")
        eje.plot(epocas, [a * 100 for a in historial["acc_val"]], marker="s", ms=3,
                 label="Validacion")
        eje.fill_between(epocas, [a * 100 for a in historial["acc_train"]],
                         [a * 100 for a in historial["acc_val"]], color="tab:red", alpha=0.1)

        # Se anota la brecha de la evaluacion final, no la de la ultima epoca:
        # esa se mide con dropout y aumento activos, asi que subestima el ajuste.
        brecha = historial["brecha_evaluada"] * 100
        eje.annotate("brecha = {:.2f} pp".format(brecha),
                     xy=(len(epocas), (historial["acc_train"][-1] + historial["acc_val"][-1]) * 50),
                     xytext=(-130, -30), textcoords="offset points", fontsize=9, color="tab:red",
                     arrowprops=dict(arrowstyle="->", color="tab:red"))

        eje.set_xlabel("Epoca")
        eje.set_ylabel("Accuracy (%)")
        eje.set_title("Accuracy")
        eje.set_ylim(80, 101)
        eje.legend(loc="lower right")
        eje.grid(alpha=0.3)

    plt.suptitle("Entrenamiento contra validacion", fontsize=13)
    plt.tight_layout()
    guardar("curvas_entrenamiento.png")


def graficar_comparacion(resultados):
    x = np.arange(2)
    a = [resultados["A"]["val"]["accuracy"] * 100, resultados["A"]["test"]["accuracy"] * 100]
    b = [resultados["B"]["val"]["accuracy"] * 100, resultados["B"]["test"]["accuracy"] * 100]

    fig, ejes = plt.subplots(1, 2, figsize=(13, 5))

    ejes[0].bar(x - 0.18, a, 0.36, label="MODELO A (inicial)", color="tab:red", alpha=0.85)
    ejes[0].bar(x + 0.18, b, 0.36, label="MODELO B (mejorado)", color="tab:green", alpha=0.85)
    for i in range(2):
        ejes[0].text(x[i] - 0.18, a[i] + 0.25, "{:.2f}%".format(a[i]), ha="center", fontsize=9)
        ejes[0].text(x[i] + 0.18, b[i] + 0.25, "{:.2f}%".format(b[i]), ha="center", fontsize=9)
    ejes[0].set_xticks(x)
    ejes[0].set_xticklabels(["Validacion", "Prueba"])
    ejes[0].set_ylabel("Accuracy (%)")
    ejes[0].set_ylim(80, 103)
    ejes[0].set_title("Accuracy de los dos modelos")
    ejes[0].legend(loc="lower center", fontsize=9)
    ejes[0].grid(axis="y", alpha=0.3)

    brechas = [resultados["A"]["brecha"] * 100, resultados["B"]["brecha"] * 100]
    ejes[1].bar(["MODELO A", "MODELO B"], brechas, color=["tab:red", "tab:green"],
                alpha=0.85, width=0.5)
    for i, v in enumerate(brechas):
        ejes[1].text(i, v + 0.05, "{:.2f} pp".format(v), ha="center", fontsize=10)
    ejes[1].set_ylabel("Brecha entrenamiento - validacion (pp)")
    ejes[1].set_title("Sobreajuste")
    ejes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    guardar("comparacion_modelos.png")


def graficar_matriz(reales, predichas):
    matriz = confusion_matrix(reales, predichas, labels=range(N_CLASES)).astype(float)
    normalizada = matriz / np.clip(matriz.sum(axis=1, keepdims=True), 1, None)

    plt.figure(figsize=(12.5, 11))
    plt.imshow(normalizada, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(fraction=0.046)

    plt.xticks(range(N_CLASES), range(N_CLASES), fontsize=7)
    plt.yticks(range(N_CLASES),
               ["{:>2} {}".format(i, NOMBRES_CLASES[i][:20]) for i in range(N_CLASES)], fontsize=7)

    # Solo se anotan las confusiones visibles, para que se siga leyendo.
    for i in range(N_CLASES):
        for j in range(N_CLASES):
            if i != j and normalizada[i, j] >= 0.03:
                plt.text(j, i, "{:.0f}".format(matriz[i, j]), ha="center", va="center",
                         fontsize=6, color="darkred")

    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.title("Matriz de confusion en PRUEBA (12,630 imagenes, 43 clases)")
    plt.tight_layout()
    guardar("matriz_confusion.png")


def graficar_accuracy_por_clase(reales, predichas):
    accs, soportes = [], []
    for clase in range(N_CLASES):
        mascara = reales == clase
        soportes.append(int(mascara.sum()))
        accs.append((predichas[mascara] == clase).mean() if mascara.sum() else 0.0)

    orden = np.argsort(accs)

    plt.figure(figsize=(9, 12))
    plt.barh(range(N_CLASES), [accs[i] * 100 for i in orden],
             color=["tab:red" if accs[i] < 0.90 else "tab:green" for i in orden])
    plt.yticks(range(N_CLASES),
               ["{:>2} {} (n={})".format(i, NOMBRES_CLASES[i][:24], soportes[i]) for i in orden],
               fontsize=8)
    plt.xlabel("Accuracy en prueba (%)")
    plt.xlim(0, 105)
    plt.axvline(90, color="gray", linestyle="--", alpha=0.7)
    plt.title("Accuracy por clase, de la peor a la mejor")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    guardar("accuracy_por_clase.png")


def graficar_errores(X_test, reales, predichas, probabilidades):
    fallos = np.where(reales != predichas)[0]
    if len(fallos) == 0:
        return

    # Los errores donde el modelo estuvo mas seguro de su respuesta equivocada.
    peores = fallos[np.argsort(probabilidades[fallos, predichas[fallos]])[::-1][:18]]

    fig, ejes = plt.subplots(3, 6, figsize=(15, 8.5))
    for eje, indice in zip(ejes.ravel(), peores):
        eje.imshow(X_test[indice])
        eje.set_title("real: {}\npredijo: {}\n{:.0f}%".format(
            NOMBRES_CLASES[reales[indice]][:20], NOMBRES_CLASES[predichas[indice]][:20],
            probabilidades[indice, predichas[indice]] * 100), fontsize=7)
        eje.axis("off")

    for eje in ejes.ravel()[len(peores):]:
        eje.axis("off")

    plt.suptitle("Los 18 errores con mayor seguridad del modelo", fontsize=12)
    plt.tight_layout()
    guardar("errores.png")


# -----------------------------
# PRINCIPAL
# -----------------------------


if __name__ == "__main__":
    analizador = argparse.ArgumentParser()
    analizador.add_argument("--epocas", type=int, default=30)
    argumentos = analizador.parse_args()

    torch.manual_seed(SEMILLA)
    np.random.seed(SEMILLA)

    X_train, y_train, X_val, y_val, X_test, y_test = preparar_datos.cargar()
    media, desviacion = estadisticas(X_train)

    print("=" * 80)
    print(" DATOS")
    print("=" * 80)
    print(" Dataset     : GTSRB, señales de transito alemanas (fotografias reales)")
    print(" Dispositivo :", DISPOSITIVO)
    print(" Clases      :", N_CLASES, "| Resolucion: 32x32 en color")
    print("\n Entrenamiento               :", len(X_train))
    print(" Validacion (pistas aparte)  :", len(X_val))
    print(" Prueba (conjunto oficial)   :", len(X_test))
    print("\n Media por canal      : R {:.4f}  G {:.4f}  B {:.4f}".format(*media))
    print(" Desviacion por canal : R {:.4f}  G {:.4f}  B {:.4f}".format(*desviacion))

    cuentas = np.bincount(y_train, minlength=N_CLASES)
    print("\n Clase con menos ejemplos:", NOMBRES_CLASES[np.argmin(cuentas)], "(" + str(cuentas.min()) + ")")
    print(" Clase con mas ejemplos  :", NOMBRES_CLASES[np.argmax(cuentas)], "(" + str(cuentas.max()) + ")")

    cargador_train = DataLoader(DatasetSenales(X_train, y_train, media, desviacion),
                                batch_size=LOTE, shuffle=True)
    cargador_train_aumentado = DataLoader(
        DatasetSenales(X_train, y_train, media, desviacion, aumentar=True),
        batch_size=LOTE, shuffle=True)
    cargador_train_eval = DataLoader(DatasetSenales(X_train, y_train, media, desviacion),
                                     batch_size=256)
    cargador_val = DataLoader(DatasetSenales(X_val, y_val, media, desviacion), batch_size=256)
    cargador_test = DataLoader(DatasetSenales(X_test, y_test, media, desviacion), batch_size=256)

    graficar_muestra(X_train, y_train)
    graficar_aumento(X_train)

    resultados = {}

    # -------------------- modelo A --------------------
    print("\n" + "=" * 80)
    print(" MODELO A - APROXIMACION INICIAL")
    print("=" * 80)

    modelo_a = ModeloBase()
    print(" 4 convoluciones + 2 densas |", format(contar_parametros(modelo_a), ","), "parametros")
    print(" Adam con tasa fija 1e-3, sin regularizacion y sin aumento de datos.")

    modelo_a, historial_a = entrenar(modelo_a, "MODELO A - sin regularizar",
                                     cargador_train, cargador_val, argumentos.epocas)

    print("\n Evaluacion del MODELO A:")
    a_train = evaluar(modelo_a, cargador_train_eval, "entrenamiento")
    a_val = evaluar(modelo_a, cargador_val, "validacion")
    a_test = evaluar(modelo_a, cargador_test, "prueba")

    brecha_a = a_train["accuracy"] - a_val["accuracy"]
    print("\n   Brecha entrenamiento - validacion:", round(brecha_a * 100, 2), "puntos porcentuales")
    print("   Llega al 100% en lo que ya vio: esta memorizando.")

    resultados["A"] = {"train": a_train, "val": a_val, "test": a_test, "brecha": brecha_a}
    historial_a["brecha_evaluada"] = brecha_a

    # -------------------- modelo B --------------------
    print("\n" + "=" * 80)
    print(" MODELO B - VERSION MEJORADA")
    print("=" * 80)

    modelo_b = ModeloMejorado()
    print(" Cambios: BatchNorm, dropout creciente, un tercer bloque convolucional,")
    print(" global average pooling, aumento de datos, weight decay, decaimiento")
    print(" coseno de la tasa y early stopping.")
    print("\n", format(contar_parametros(modelo_b), ","), "parametros ({:.1f} veces menos)".format(
        contar_parametros(modelo_a) / contar_parametros(modelo_b)))

    modelo_b, historial_b = entrenar(modelo_b, "MODELO B - regularizado",
                                     cargador_train_aumentado, cargador_val, argumentos.epocas,
                                     tasa=2e-3, weight_decay=1e-4, scheduler=True,
                                     early_stopping=True)

    print("\n Evaluacion del MODELO B:")
    b_train = evaluar(modelo_b, cargador_train_eval, "entrenamiento")
    b_val = evaluar(modelo_b, cargador_val, "validacion")
    b_test = evaluar(modelo_b, cargador_test, "prueba")

    brecha_b = b_train["accuracy"] - b_val["accuracy"]
    print("\n   Brecha entrenamiento - validacion:", round(brecha_b * 100, 2), "puntos porcentuales")

    resultados["B"] = {"train": b_train, "val": b_val, "test": b_test, "brecha": brecha_b}
    historial_b["brecha_evaluada"] = brecha_b

    # -------------------- comparacion --------------------
    print("\n" + "=" * 80)
    print(" COMPARACION")
    print("=" * 80)
    print(" {:<34} {:>12} {:>12}".format("", "MODELO A", "MODELO B"))
    print(" {:<34} {:>12,} {:>12,}".format("Parametros", contar_parametros(modelo_a),
                                           contar_parametros(modelo_b)))
    for etiqueta, va, vb in [
            ("Accuracy entrenamiento (%)", a_train["accuracy"] * 100, b_train["accuracy"] * 100),
            ("Accuracy validacion (%)", a_val["accuracy"] * 100, b_val["accuracy"] * 100),
            ("Accuracy PRUEBA (%)", a_test["accuracy"] * 100, b_test["accuracy"] * 100),
            ("Brecha train - val (pp)", brecha_a * 100, brecha_b * 100)]:
        print(" {:<34} {:>12.2f} {:>12.2f}".format(etiqueta, va, vb))
    print(" {:<34} {:>12.4f} {:>12.4f}".format("F1 macro PRUEBA", a_test["f1_macro"], b_test["f1_macro"]))

    errores_a = int(round((1 - a_test["accuracy"]) * len(y_test)))
    errores_b = int(round((1 - b_test["accuracy"]) * len(y_test)))
    print("\n Errores sobre las {:,} imagenes de prueba: {} -> {} ({:.0f}% menos)".format(
        len(y_test), errores_a, errores_b, 100 * (errores_a - errores_b) / max(errores_a, 1)))

    # -------------------- analisis final --------------------
    print("\n" + "=" * 80)
    print(" REPORTE POR CLASE DEL MODELO FINAL EN PRUEBA")
    print("=" * 80)

    reales, predichas, probabilidades = predecir_conjunto(modelo_b, cargador_test)
    print(classification_report(reales, predichas, labels=range(N_CLASES),
                                target_names=[n[:26] for n in NOMBRES_CLASES],
                                digits=3, zero_division=0))

    print("=" * 80)
    print(" GRAFICAS")
    print("=" * 80)
    graficar_entrenamiento([historial_a, historial_b])
    graficar_comparacion(resultados)
    graficar_matriz(reales, predichas)
    graficar_accuracy_por_clase(reales, predichas)
    graficar_errores(X_test, reales, predichas, probabilidades)

    torch.save({"pesos": modelo_b.state_dict(), "media": media, "desviacion": desviacion,
                "accuracy_prueba": float(b_test["accuracy"])}, RUTA_MODELO)
    print("\n Modelo guardado en modelo_gtsrb.pt")

    with open(RUTA_HISTORIAL, "w") as archivo:
        json.dump({"A": historial_a, "B": historial_b}, archivo, indent=2)
    print(" Historial guardado en historial.json")
    print("\n Para hacer predicciones: python3 predecir.py")

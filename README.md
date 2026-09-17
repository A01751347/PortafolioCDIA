# Portafolio Módulo 2 — Aprendizaje Máquina

**Santiago Serrano Montalvo · A01751347**
Inteligencia artificial avanzada para la ciencia de datos

Este repositorio tiene las cuatro entregas del Módulo 2. Cada carpeta funciona por separado: tiene su código, sus datos, sus figuras y su reporte en PDF, y se corre desde la consola con un solo comando.

## Entregas

| Entrega | Carpeta | Qué es | Reporte |
|---|---|---|---|
| Sin framework | [`M2_SinFramework/`](M2_SinFramework/) | KNN programado desde cero | [PDF](M2_SinFramework/Reporte_SinFramework.pdf) |
| Con framework | [`M2_ConFramework/`](M2_ConFramework/) | Random Forest con scikit-learn | [PDF](M2_ConFramework/Reporte_ConFramework.pdf) |
| Análisis del modelo | [`M2_Analisis/`](M2_Analisis/) | Sesgo, varianza, ajuste y regularización | [PDF](M2_Analisis/Reporte_Analisis.pdf) |
| Deep learning | [`M2_DeepLearning/`](M2_DeepLearning/) | CNN para señales de tránsito con PyTorch | [PDF](M2_DeepLearning/Reporte_DeepLearning.pdf) |

---

## 1. Sin framework — KNN desde cero

```
cd M2_SinFramework
python3 main.py
```

Clasifica el tipo de garantía (`collateral`) de un crédito automotriz con K-Nearest Neighbors programado a mano. No usa ninguna biblioteca de aprendizaje máquina: la media, la desviación estándar, la normalización z-score, la distancia euclidiana, la votación, la matriz de confusión y las métricas están escritas en el archivo. Matplotlib se usa solo para las gráficas.

Dataset: `interest_ledger.csv`, 10,576 pagos mensuales de créditos automotrices, 5 clases. Partición 75/25 con semilla fija.

Resultado con k=5: **81.51 % de accuracy** y **0.739 de F1 macro**, contra 42.55 % de la referencia que siempre predice la clase mayoritaria.

## 2. Con framework — Random Forest

```
cd M2_ConFramework
python3 main.py
```

El mismo problema y las mismas cinco variables, ahora con scikit-learn, para poder comparar. Se eligen los hiperparámetros con validación cruzada estratificada de 5 particiones sin tocar el conjunto de prueba.

Resultado: **86.20 % de accuracy** y **0.8180 de F1 macro**. Casi toda la mejora sobre el KNN está en la clase minoritaria DEMO, cuyo F1 sube de 0.187 a 0.479 gracias a `class_weight="balanced"`.

## 3. Análisis del desempeño

```
cd M2_Analisis
python3 main.py
```

Diagnóstico del Random Forest con tres conjuntos separados (60/20/20) y umbrales de diagnóstico fijos en el código.

| | Modelo base | Regularizado |
|---|---|---|
| Sesgo | BAJO (F1 train 0.9975) | BAJO |
| Varianza | ALTA (brecha 0.2109) | ALTA, pero 38 % menor (0.1300) |
| Ajuste | Overfitting | Overfitting reducido |
| F1 macro validación | 0.7866 | **0.8510** |
| F1 macro prueba | 0.8163 | **0.8381** |

El hallazgo principal: el valor por defecto `max_features="sqrt"` deja solo 2 de las 5 variables disponibles en cada corte, lo que priva a muchos cortes de las dos más informativas. Subirlo a 5 mejora la validación 6 puntos y además baja la varianza.

También se detectó que el 75 % de las filas de prueba comparten vehículo (VIN) con el entrenamiento. Con una partición agrupada por VIN el F1 macro baja a 0.8116, que es la estimación honesta.

## 4. Deep learning — CNN para señales de tránsito

```
cd M2_DeepLearning
python3 preparar_datos.py    # descarga y prepara GTSRB (solo la primera vez)
python3 main.py              # entrena y evalua los dos modelos
python3 predecir.py          # predicciones desde la consola
```

Red convolucional en PyTorch que reconoce 43 tipos de señales de tránsito a partir de fotografías reales tomadas desde un auto en movimiento.

Dataset: GTSRB, con 31,380 imágenes de entrenamiento, 7,829 de validación y 12,630 de prueba (el conjunto oficial del benchmark). La validación se separa por pista completa, porque cada señal física se fotografió 30 veces seguidas y una partición al azar dejaría fotos casi idénticas de los dos lados.

Se comparan dos modelos: una CNN inicial sin regularización y una versión mejorada con BatchNorm, dropout, un tercer bloque convolucional, global average pooling, aumento de datos, weight decay, decaimiento coseno y early stopping.

El aumento de datos no incluye espejado horizontal, aunque sea el aumento por defecto en visión: espejar "curva peligrosa a la izquierda" produce la señal de "curva peligrosa a la derecha", que es otra clase del dataset.

| | Modelo A (inicial) | Modelo B (mejorado) |
|---|---:|---:|
| Parámetros | 2,185,291 | **309,771** |
| Accuracy validación | 97.74 % | **99.89 %** |
| Accuracy prueba | 97.18 % | **99.11 %** |
| F1 macro prueba | 0.9550 | **0.9861** |
| Brecha train − val | 2.26 pp | **0.11 pp** |
| Errores sobre 12,630 imágenes | 356 | **112** |

El cambio de mayor impacto fue sustituir la capa densa de 512 por global average pooling, que quitó 1.9 millones de parámetros que eran justo los que se usaban para memorizar.

El dataset no está versionado, ocupa alrededor de 1 GB una vez descargado. `preparar_datos.py` lo baja y lo prepara solo.

---

## Requisitos

```
pip install numpy pandas scikit-learn matplotlib torch pillow
```

| Entrega | Necesita |
|---|---|
| Sin framework | Python 3 y matplotlib |
| Con framework | pandas, numpy, scikit-learn, matplotlib |
| Análisis | lo mismo que con framework |
| Deep learning | además torch y pillow |

Todo son archivos `.py` que corren desde la consola, no hay notebooks.

## Estructura

```
PortafolioCDIA/
├── M2_SinFramework/
│   ├── main.py
│   ├── interest_ledger.csv
│   ├── reporte.md
│   ├── Reporte_SinFramework.pdf
│   ├── salida.txt
│   ├── ayuda_de_IA.txt
│   └── figuras/
├── M2_ConFramework/       (misma estructura)
├── M2_Analisis/           (misma estructura)
└── M2_DeepLearning/
    ├── preparar_datos.py
    ├── main.py
    ├── predecir.py
    ├── modelo_gtsrb.pt
    ├── historial.json
    ├── reporte.md
    ├── Reporte_DeepLearning.pdf
    ├── datos/             (no versionado, lo genera preparar_datos.py)
    └── figuras/
```

Las primeras tres entregas usan el mismo dataset y el mismo problema, así que cada mejora se puede atribuir a una causa concreta:

```
KNN sin framework        F1 macro 0.739
Random Forest            F1 macro 0.818   (+0.079 por class_weight="balanced")
Random Forest ajustado   F1 macro 0.838   (+0.020 por regularizacion)
```

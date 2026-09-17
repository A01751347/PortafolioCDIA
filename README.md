# Portafolio Módulo 2 — Aprendizaje Máquina

**Santiago Serrano Montalvo — A01751347**
TC3007C · Inteligencia artificial avanzada para la ciencia de datos

Este repositorio contiene las cuatro entregas del Módulo 2. Cada carpeta es **autocontenida**: tiene su propio código, sus datos, sus figuras y su reporte en PDF, y se ejecuta de forma independiente desde la consola con un solo comando.

---

## Índice de entregas

| # | Entrega | Carpeta | Qué contiene | Reporte |
|---|---|---|---|---|
| 1 | Implementación **sin** framework | [`M2_SinFramework/`](M2_SinFramework/) | KNN programado desde cero en Python puro | [PDF](M2_SinFramework/Reporte_SinFramework.pdf) |
| 2 | Implementación **con** framework | [`M2_ConFramework/`](M2_ConFramework/) | Random Forest con scikit-learn | [PDF](M2_ConFramework/Reporte_ConFramework.pdf) |
| 3 | **Análisis** y reporte del desempeño | [`M2_Analisis/`](M2_Analisis/) | Sesgo, varianza, ajuste y regularización | [PDF](M2_Analisis/Reporte_Analisis.pdf) |
| 4 | Modelo de **Deep Learning** | [`M2_DeepLearning/`](M2_DeepLearning/) | CNN para señales de tránsito con PyTorch | [PDF](M2_DeepLearning/Reporte_DeepLearning.pdf) |

---

## 1. Implementación sin framework — KNN desde cero

**Carpeta:** [`M2_SinFramework/`](M2_SinFramework/) · **Ejecutar:** `cd M2_SinFramework && python3 main.py`

Clasificación del tipo de garantía (`collateral`) de un crédito automotriz usando **K-Nearest Neighbors programado íntegramente a mano**. No se usa ninguna biblioteca de aprendizaje máquina ni de estadística: la media, la desviación estándar, la estandarización z-score, la distancia euclidiana, la votación, la matriz de confusión, el accuracy, la precisión, el recall y el F1 están escritos en el archivo. `matplotlib` se usa sólo para dibujar las gráficas.

| | |
|---|---|
| **Dataset** | `interest_ledger.csv` — 10,576 pagos mensuales de créditos automotrices |
| **Partición** | 75 % entrenamiento (7,932) / 25 % prueba (2,644), semilla fija 42 |
| **Modelo** | KNN con k = 5 y distancia euclidiana sobre variables estandarizadas |
| **Accuracy** | **81.51 %** (línea base: 42.55 %) |
| **F1 macro** | **0.739** |

Incluye **7 pruebas automáticas de correctitud** que verifican cada componente del algoritmo contra casos resueltos a mano (distancia 3-4-5, media y desviación, votación por mayoría, desempate, memorización con k=1, matriz de confusión y F1).

**Archivos:** `main.py` · `interest_ledger.csv` · `reporte.md` · `Reporte_SinFramework.pdf` · `figuras/` · `salida.txt`

---

## 2. Implementación con framework — Random Forest

**Carpeta:** [`M2_ConFramework/`](M2_ConFramework/) · **Ejecutar:** `cd M2_ConFramework && python3 main.py`

El **mismo problema y las mismas cinco variables**, ahora con **scikit-learn**, para que la comparación contra el KPN implementado a mano sea directa y justa.

| | |
|---|---|
| **Modelo** | `RandomForestClassifier` — 300 árboles, `class_weight="balanced"` |
| **Selección** | `StratifiedKFold` de 5 particiones + `cross_val_score`, sin tocar el conjunto de prueba |
| **Métrica** | **F1 macro**, justificada por el desbalance 19:1 del dataset |
| **Accuracy** | **86.20 %** |
| **F1 macro** | **0.8180** |

Mejora de **+4.69 puntos de accuracy** y **+0.079 de F1 macro** sobre el KNN. Casi toda la ganancia está en la clase minoritaria DEMO, cuyo F1 sube de 0.187 a 0.479 gracias a `class_weight="balanced"`.

**Archivos:** `main.py` · `interest_ledger.csv` · `reporte.md` · `Reporte_ConFramework.pdf` · `figuras/` · `salida.txt`

---

## 3. Análisis y reporte sobre el desempeño

**Carpeta:** [`M2_Analisis/`](M2_Analisis/) · **Ejecutar:** `cd M2_Analisis && python3 main.py` *(≈2 min)*

Diagnóstico completo del Random Forest con **tres conjuntos separados** y umbrales de diagnóstico definidos explícitamente en el código.

| Aspecto | Modelo base | Tras regularizar |
|---|---|---|
| **Sesgo** | BAJO (F1 train 0.9975) | BAJO |
| **Varianza** | ALTA (brecha 0.2109) | ALTA pero **−38 %** (brecha 0.1300) |
| **Ajuste** | OVERFITTING | OVERFITTING reducido |
| **F1 macro validación** | 0.7866 | **0.8510** (+0.0645) |
| **F1 macro prueba** | 0.8163 | **0.8381** (+0.0218) |

**Hallazgo principal:** el valor por defecto `max_features="sqrt"` deja sólo 2 de las 5 variables disponibles en cada corte, privando a muchos cortes de las dos más informativas. Subirlo a 5 mejora la validación 6 puntos **y** reduce la varianza.

**Hallazgo adicional:** el **75 % de las filas de prueba comparten vehículo (VIN) con el entrenamiento**. Se cuantifica esa fuga con una partición agrupada por VIN, que da la estimación honesta de **0.8116** de F1 macro.

Seis figuras: curvas de aprendizaje antes/después, curvas de validación de cuatro hiperparámetros, frontera varianza-desempeño de las 192 configuraciones probadas, comparación antes/después, matrices de confusión lado a lado, F1 por clase y efecto de la fuga por VIN.

**Archivos:** `main.py` · `interest_ledger.csv` · `reporte.md` · `Reporte_Analisis.pdf` · `figuras/` · `salida.txt`

---

## 4. Modelo de Deep Learning — CNN para señales de tránsito

**Carpeta:** [`M2_DeepLearning/`](M2_DeepLearning/)

```
cd M2_DeepLearning
python3 preparar_datos.py    # descarga y prepara GTSRB (sólo la primera vez)
python3 main.py              # entrena y evalúa los dos modelos
python3 predecir.py          # predicciones desde la consola
```

Red neuronal convolucional en **PyTorch** que reconoce **43 tipos de señales de tránsito** a partir de fotografías reales tomadas desde un auto en movimiento.

| | |
|---|---|
| **Dataset** | GTSRB — German Traffic Sign Recognition Benchmark |
| **Entrenamiento** | 31,380 imágenes |
| **Validación** | 7,829 imágenes (**pistas separadas**, ninguna señal física compartida) |
| **Prueba** | 12,630 imágenes (conjunto oficial del benchmark) |

Se comparan dos modelos: una **CNN inicial sin regularización** y una **versión mejorada** con BatchNorm, Dropout creciente, un tercer bloque convolucional, Global Average Pooling, aumento de datos, weight decay, decaimiento coseno de la tasa de aprendizaje y early stopping. El reporte documenta qué cambió cada técnica y por qué.

| | MODELO A (inicial) | MODELO B (mejorado) |
|---|---:|---:|
| Parámetros | 2,185,291 | **309,771** (7.1× menos) |
| Accuracy validación | 97.82 % | **99.91 %** |
| **Accuracy prueba** | 97.31 % | **99.52 %** |
| F1 macro prueba | 0.9602 | **0.9915** |
| Brecha train−val | 2.18 pp | **0.09 pp** |
| Errores sobre 12,630 imágenes | 340 | **60** (82 % menos) |

El modelo final supera el desempeño humano reportado en el benchmark original (98.84 %) usando 7 veces menos parámetros que la aproximación inicial. El cambio de mayor impacto fue sustituir la capa densa de 512 neuronas por Global Average Pooling: eliminó 1.9 millones de parámetros que eran justamente los que se usaban para memorizar.

> **Detalle de diseño:** el aumento de datos **no incluye espejado horizontal**, aunque sea el aumento por defecto en visión. Espejar «curva peligrosa a la izquierda» produce exactamente la señal de «curva peligrosa a la derecha», que es otra clase del dataset.

**Archivos:** `preparar_datos.py` · `main.py` · `predecir.py` · `reporte.md` · `Reporte_DeepLearning.pdf` · `figuras/` · `modelo_gtsrb.pt` · `historial.json` · `salida.txt`

El dataset **no está versionado** (ocupa ≈1 GB una vez descargado y descomprimido); `preparar_datos.py` lo descarga y lo prepara automáticamente en `datos/`.

---

## Requisitos

| Entrega | Requisitos |
|---|---|
| M2_SinFramework | Python 3 + `matplotlib` |
| M2_ConFramework | `+ pandas`, `numpy`, `scikit-learn` |
| M2_Analisis | los mismos que M2_ConFramework |
| M2_DeepLearning | `+ torch`, `pillow` |

```
pip install numpy pandas scikit-learn matplotlib torch pillow
```

Todo el código son archivos `.py` que corren desde la consola. **No hay notebooks** ni dependencias de un IDE.

---

## Estructura del repositorio

```
PortafolioCDIA/
├── README.md
├── M2_SinFramework/        KNN desde cero
│   ├── main.py
│   ├── interest_ledger.csv
│   ├── reporte.md
│   ├── Reporte_SinFramework.pdf
│   ├── salida.txt
│   ├── ayuda_de_IA.txt
│   └── figuras/
├── M2_ConFramework/        Random Forest con scikit-learn
│   └── (misma estructura)
├── M2_Analisis/            Sesgo, varianza, ajuste y regularización
│   └── (misma estructura)
├── M2_DeepLearning/        CNN con PyTorch sobre GTSRB
│   ├── preparar_datos.py
│   ├── main.py
│   ├── predecir.py
│   ├── modelo_gtsrb.pt
│   ├── historial.json
│   ├── reporte.md
│   ├── Reporte_DeepLearning.pdf
│   ├── salida.txt
│   ├── ayuda_de_IA.txt
│   ├── datos/              (no versionado; lo genera preparar_datos.py)
│   └── figuras/
└── herramientas/
    ├── md_a_pdf.sh         genera los PDF a partir de los reportes .md
    └── estilo.css
```

Los reportes se escriben en Markdown y se convierten a PDF con:

```
herramientas/md_a_pdf.sh M2_Analisis/reporte.md M2_Analisis/Reporte_Analisis.pdf
```

---

## Hilo conductor de las cuatro entregas

Las tres primeras trabajan sobre **el mismo dataset y el mismo problema**, lo que permite atribuir cada mejora a una causa concreta y no a un cambio de escenario:

```
KNN sin framework        F1 macro 0.739   →  línea de partida
Random Forest            F1 macro 0.818   →  +0.079 por class_weight="balanced"
Random Forest ajustado   F1 macro 0.838   →  +0.020 por regularización
```

En la cuarta entrega la misma disciplina produce el salto más grande:

```
CNN inicial              F1 macro 0.960   →  sobreajustada, 340 errores
CNN regularizada         F1 macro 0.992   →  +0.031 y 60 errores, con 7x menos parámetros
```

La cuarta cambia de dominio a visión por computadora porque la rúbrica exige una arquitectura profunda, pero conserva el mismo método de trabajo: partición honesta que evita la fuga de datos, diagnóstico explícito del sobreajuste, y mejora documentada técnica por técnica.

De hecho, el problema metodológico central resultó ser **el mismo en los dos dominios**:

| Entrega | Unidad que se repite | Riesgo | Solución |
|---|---|---|---|
| M2_Analisis | un vehículo (VIN) aparece en varios meses | 75 % de la prueba comparte VIN con entrenamiento | partición agrupada por VIN |
| M2_DeepLearning | una señal física se fotografía 30 veces | validación inflada por fotos casi idénticas | partición por pista completa |

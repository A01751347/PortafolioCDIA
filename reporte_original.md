
# Clasificación del tipo de garantía usando KNN

## 1. Introducción

En este proyecto se implementó el algoritmo K-Nearest Neighbors (KNN) sin utilizar frameworks de Machine Learning. El objetivo es clasificar el tipo de garantía de diferentes créditos automotrices utilizando información de un dataset.

Para hacer el programa se utilizó Python. Los cálculos principales del algoritmo se hicieron manualmente y se utilizó Matplotlib únicamente para mostrar algunas gráficas de los resultados.

## 2. Algoritmo KNN

KNN es un algoritmo de clasificación que utiliza los datos más cercanos a un nuevo punto para decidir a qué clase pertenece.

El funcionamiento que se utilizó fue el siguiente:

1. Se toma un dato que se quiere predecir.
2. Se calcula su distancia con todos los datos de entrenamiento.
3. Se ordenan las distancias.
4. Se seleccionan los k vecinos más cercanos.
5. Cada vecino vota por su clase.
6. La clase que tenga más votos es la predicción.

Para calcular las distancias se utilizó la distancia euclidiana:

$$
d = \sqrt{\sum (x_i-y_i)^2}
$$

Se decidió utilizar un valor de:

$$
k=5
$$

## 3. Dataset

El dataset utilizado es `interest_ledger.csv`.

Tiene 10,576 registros relacionados con créditos automotrices.

La variable que se quiere predecir es `collateral`, que representa el tipo de garantía.

Las clases que aparecen en el dataset son:

* DEMO
* FLOTILLA
* NUEVO
* REFACC
* SEMINUEVOS

Para realizar las predicciones se utilizaron cinco variables:

* Monto principal del crédito.
* Interés.
* Tasa calculada como interés / principal.
* Antigüedad del vehículo.
* Mes.

Los datos se dividieron en dos partes:

* 75 % para entrenamiento.
* 25 % para prueba.

Se utilizó `random.seed(42)` para que la división fuera la misma cada vez que se ejecutara el programa.

## 4. Normalización

Antes de utilizar KNN se normalizaron las variables.

Esto fue necesario porque algunas variables tienen valores mucho más grandes que otras. Por ejemplo, el monto del crédito puede ser de miles de pesos mientras que la tasa es un valor pequeño.

Si se utilizan directamente estos valores, las variables grandes afectan más el cálculo de la distancia.

Se utilizó la siguiente fórmula:

$$
z = \frac{x-\text{media}}{\text{desviación}}
$$

La media y desviación fueron calculadas utilizando los datos de entrenamiento.

## 5. Resultados

Después de ejecutar el modelo con \(k=5\), se obtuvo un accuracy de:

$$
81.51\%
$$

Esto significa que aproximadamente 81 de cada 100 datos del conjunto de prueba fueron clasificados correctamente.

También se creó una matriz de confusión para observar en qué clases hubo más errores.

### Matriz de confusión

| Real / Predicción | DEMO | FLOTILLA | NUEVO | REFACC | SEMINUEVOS |
| ------------------ | ---: | -------: | ----: | -----: | ---------: |
| DEMO               |    7 |       18 |    25 |      0 |          0 |
| FLOTILLA           |    6 |      692 |   151 |      0 |          5 |
| NUEVO              |   11 |      227 |   933 |      0 |          8 |
| REFACC             |    0 |        0 |     2 |     57 |          0 |
| SEMINUEVOS         |    1 |       19 |    16 |      0 |        466 |

Se puede observar que una gran parte de los errores ocurre entre las clases NUEVO y FLOTILLA.

También se calcularon precision, recall y F1 para cada una de las clases.

| Clase      | Precision | Recall |    F1 |
| ---------- | --------: | -----: | ----: |
| DEMO       |     0.280 |  0.140 | 0.187 |
| FLOTILLA   |     0.724 |  0.810 | 0.765 |
| NUEVO      |     0.828 |  0.791 | 0.809 |
| REFACC     |     1.000 |  0.966 | 0.983 |
| SEMINUEVOS |     0.973 |  0.928 | 0.950 |

Las clases REFACC y SEMINUEVOS tuvieron los mejores resultados.

DEMO fue la clase con peores resultados. Esto puede estar relacionado con que existen muchos menos datos de esta clase comparados con las otras.

## 6. Pruebas con diferentes valores de K

También se hicieron algunas pruebas cambiando el número de vecinos.

|  K | Accuracy |
| -: | -------: |
|  1 |  84.23 % |
|  3 |  82.75 % |
|  5 |  81.51 % |
|  7 |  81.01 % |
|  9 |  80.22 % |
| 11 |  80.48 % |
| 15 |  79.84 % |
| 21 |  79.01 % |

Aunque \(k=1\) obtuvo un accuracy mayor, se decidió utilizar \(k=5\) para que la clasificación no dependiera únicamente del vecino más cercano.

Con más vecinos también se toma en cuenta información de varios puntos antes de decidir la clase.

## 7. Análisis

En general, el algoritmo tuvo buenos resultados.

Las clases REFACC y SEMINUEVOS fueron clasificadas correctamente en la mayoría de los casos.

Los principales errores se encontraron entre NUEVO y FLOTILLA. Esto probablemente ocurre porque sus características numéricas son similares, por lo que para KNN los puntos pueden quedar cerca entre sí.

La clase DEMO también tuvo resultados bajos. Esta clase tiene menos datos que otras clases, por lo que muchas veces los vecinos más cercanos pertenecen a otra categoría.

Por esta razón, el accuracy general no es suficiente para evaluar completamente el modelo y también se utilizaron precision, recall y F1.

## 8. Conclusión

Se logró implementar el algoritmo KNN desde cero sin utilizar una biblioteca de Machine Learning.

El programa puede leer el dataset, dividir los datos, normalizarlos, calcular las distancias entre los puntos, seleccionar los vecinos más cercanos y realizar predicciones.

Con \(k=5\) se obtuvo un accuracy de 81.51 %, por lo que el algoritmo logra clasificar correctamente la mayoría de los datos de prueba.

Sin embargo, existen diferencias importantes entre las clases. Algunas, como REFACC y SEMINUEVOS, tienen resultados altos, mientras que DEMO presenta mayor dificultad.

Como mejora futura podrían utilizarse más variables que permitan diferenciar mejor las clases o buscar una forma de reducir el efecto del desbalance entre ellas.

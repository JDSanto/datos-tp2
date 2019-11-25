# Trabajo Práctico 2: Machine Learning

## Índice
- [Introducción](#introducci%c3%b3n)
  - [Filtrado](#filtrado)
  - [Valores Faltantes](#valores-faltantes)
  - [Clustering](#clustering)
- [Features](#features)
  - [Feature engineering](#feature-engineering)
  - [Independientes del precio](#independientes-del-precio)
    - [Metros Totales y Cubiertos](#metros-totales-y-cubiertos)
    - [Por tipo de propiedad](#por-tipo-de-propiedad)
    - [Por ubicación](#por-ubicaci%c3%b3n
    - [Por fecha](#por-fecha)
    - [Propiedades booleanas](#por-propiedades-booleanas)
    - [Por cantidad de habitaciones, garages y baños](#por-cantidad-de-habitaciones-garages-y-ba%c3%b1os)
  -[Dependientes del precio](#dependientes-del-precio)
    - [Por ubicación](#por-ubicaci%c3%b3n)
    - [Por tipo de propiedad](#por-tipo-de-propiedad)
    - [Por fecha](#por-fecha)
    - [Por cantidad de habitaciones, garages y baños](#por-cantidad-de-habitaciones-garages-y-ba%c3%b1os)
    - [Por propiedades booleanas](#por-propiedades-booleanas)
    - [Por puntajes(Insight)](#por-puntajesinsight)
  - [En relación a los textos](#en-relaci%c3%b3n-a-los-textos)
  - [Distancias](#distancias)
  - [One Hot Encoding](#one-hot-encoding)
  - [Feature Selection](#feature-selection)
    - [Random Forest](#random-forest)
    - [SelectKBest](#selectkbest)
    - [SelectFromModel](#selectfrommodel)
    - [Recursive Feature Elimination (RFE)](#recursive-feature-elimination-rfe)
    - [Recursive Feature Elimination w/ Cross Validation (RFECV)](#recursive-feature-elimination-w-cross-validation-rfecv)
- [Modelos](#modelos)
    - [XGBoost](#xgboost)
    - [KNN](#knn)
    - [Redes neuronales](#redes-neuronales)
    - [LightGBM](#lightgbm)
- [Parameter tuning](#parameter-tuning)
- [Ensambles](#ensambles)
  - [Stacking](#stacking)
  - [Blending](#blending)
- [Desarrollo](#desarrollo)
- [Resultados obtenidos](#resultados-obtenidos)_
- [Conclusiones](#conclusiones)

# Introducción
El objetivo principal del trabajo es  determinar, para cada propiedad presentada, cuál es su valor de mercado.
La realización del trabajo se hace con algoritmos de Machine Learning, una disciplina que busca poder generar clasificaciones en base a un entrenamiento sobre información pasada, seguida de una validación de las predicciones generadas. En el trabajo se prueban distintos algoritmos, los cuales todos en distinta manera hacen uso de los datos. Es por esto que es muy importante saber qué datos usar, y buscar cómo codificarlos de tal forma que mejor se aprovechen.

El primer paso del trabajo consistió en realizar una breve investigación sobre lo ya hecho en el trabajo anterior. En el primer trabajo práctico de la materia se realizó un análisis exploratorio de datos de la Zona Prop. Si bien no son exactamente los mismos datos que los trabajados acá, son de la misma índole.

### Filtrado

La detección de anomalı́as (outliers) implica el econocimiento y corrección o eliminación de datos erróneos, un dato anómalo es aquel que tiene valores imposibles para uno o mas de sus atributos. Por lo que en una primera instancia se decide, filtrar aquellos registros que semanticamente son posible pero no tiene sentido en el
contexto de los demás datos, es decir que probablemente se trate de un dato mal ingresado.
En las Figuras I-IV se pueden observar los recortes realizados.

![Busqueda de outliers con features base sin precio](./images/out_features.png)

![Filtrado de outliers con features base sin precio](./images/out_features_recortado.png)

![Busqueda de outliers con features base incluyendo precio](./images/out_features_precio.png)

![Filtrado de outliers con features base incluyendo precio](./images/out_features_precio_recortado.png)

### Nulos

## Clustering
Con la intenci´on de ver si se encontraban clusters que agrupen a los usuarios
que compraron, se utiliz´o el algoritmo T-SNE, que es el estado del arte para la
representaci´on de datos en dos dimensiones.

### LOFANO ESTA HACIENDO CLUSTERIG


# Features

## Feature engineering
Con lo investigado del previ´o trabajo y todos los dataframes generados,
se busca todo tipo de atributos de los usuarios, para que luego puedan ser
seleccionados y aprovechados por los algoritmos a aplicar.

### Independientes del precio
#### Metros Totales y Cubiertos
#### Por tipo de propiedad
#### Por ubicación
#### Por fecha
#### Propiedades booleanas
#### Por cantidad de habitaciones, garages y baños

### Dependientes del precio

#### Por ubicación
#### Por tipo de propiedad
#### Por fecha
#### Por cantidad de habitaciones, garages y baños
#### Por propiedades booleanas
#### Por puntajes(Insight)
### En relación a los textos
### Distancias
### One Hot Encoding


## Feature Selection
Una vez que se tienen todos los atributos en un mismo dataframe y luego
de un par de pruebas se ve que no siempre hay que entrenar los modelos con
la totalidad del dataframe. Viendo que para algunos algoritmos el orden y la
selecci´on de los features logrababa distintos resultados, se busca la forma de
encontrar la combinaci´on ´optima de features y eliminar todo el ruido posible.

### Random Forest

### SelectKBest
### SelectFromModel
### Recursive Feature Elimination (RFE)
### Recursive Feature Elimination w/ Cross Validation (RFECV)


# Modelos
## XGBoost
XGBoost es un algoritmo muy eficiente de gradient boosting en ´arboles. Uno
de sus h´ıper-par´ametros m´as importantes es la funcion objetivo, de la cual se
usa la log´ıstica binaria, ya que se tiene un problema de clasificaci´on binaria y
no uno de regresión

## KNN
Un algoritmo sencillo utilizado es el de los K vecinos m´as cercanos. Este
algoritmo consiste en encontrar los vecinos m´as cercanos del punto a clasificar,
y luego simplemente predecir que el punto en cuesti´on es de la clase del cual la
mayor´ıa de sus vecinos sean parte. Sus h´ıper-par´ametros a definir es la cantidad
de vecinos a tomar en cuenta y la distancia a utilizar entre ellos.
Es tal vez en su sencillez que est´a una de sus mayores desventajas: es un
algoritmo de orden cuadr´atico, y esto impacta mucho sobre la ejecuci´on de
cada modelo que se hace. Otra cosa a remarcar es que los datos deben est´ar
normalizados a la hora de procesarlos.

## Redes neuronales

## LightGBM
LightGBM es simplemente el algoritmo que constantemente mejores resultados nos di´o. Este algoritmo de gradient boosting sobre ´arboles se diferencia
de XGBoost en que construye los ´arboles seg´un las hojas, y no los niveles. Es
importante que sus h´ıper-par´ametros est´en bien configurados (por ejemplo, la
profundidad m´axima de los ´arboles), r´apidamente
se encuentran muy buenos saltos de calidad en el modelo al utilizar hyperopt para encontrar los mejores hiperparametros segun los diferentes features utlizados.

Este algoritmo tambi´en se destaca por ser r´apido y consumir poca memoria.
Tambi´en, tiene un gran manejo de la dimensionalidad de los datos, sin cambiar
mucho frente a ellos.

## Parameter tuning
En los primeros modelos corridos fue cuando se empezo a notar lo que ya
se sab´ıa: los h´ıper-par´ametros son sencillamente lo m´as importante de cada
algoritmo, y la diferencia entre un buen modelo y uno promedio o incluso malo.
Inicialmente, debido a la gran cantidad de opciones para algunos algoritmos,
optamos por un m´etodo greedy, que consist´ıa en un peque˜no framework donde se
van probando distintos h´ıper-par´ametros con distintos valores progresivamente.
Es decir, se parte un grid search grande en varios m´as peque˜nos, ya que por la
naturaleza del algoritmo este proceso es muy costoso en tiempo. Si bien esto no
encuentra la combinaci´on ´optima, al menos da resultados bastante favorables y
se ahorra mucho tiempo de ejecuci´on.
Esto dio resultados cuestionables, por lo que implementamos una b´usqueda
aleatoria de h´ıper-par´ametros, y luego, utilizamos grid search (un m´etodo de
fuerza bruta que lo que hace es correr las posibilidades planteadas hasta encontrar la mejor combinaci´on de los h´ıper-par´ametros a probar) sobre un rango
acotado basado en los resultados anteriores

HYPEROPT

## Ensambles
### Stacking
### Blending


# Desarrollo
5.1. Submission framework
Se define un framework y una serie de funciones para armar las postulaciones
de predicciones del trabajo pr´actico. Las mismas siguen los siguientes pasos:
1. Creaci´on de la matriz X y el vector y para entrenar.
2. Generaci´on del split para obtener los sets de entrenamiento y de prueba.
3. Ejecuci´on del algoritmo de Machine Learning que devuelve un dataframe
con person como ´ındice y los labels como ´unica columna.
4. Se obtienen las 3 medidas utilizadas como m´etrica para evaluar el rendimiento del algoritmo: precisi´on, auc y aucpr.
5. Se predicen las probabilidades.
6. Se observa informaci´on relevante de la ejecuci´on como la importancia de
los features elegidos.
7. Se guardan los resultados como csv para ser submiteados.


# Resultados obtenidos

# Conclusiones
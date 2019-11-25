# Trabajo Práctico 2: Machine Learning

## Índice
- [Introducción](#introducci%c3%b3n)
  - [Filtrado](#filtrado)
  - [Valores Faltantes](#valores-faltantes)
  - [Clustering](#clustering)
  - [PCA](#pca)
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
Algunos algoritmos admiten datos incompletos y otros no. En los casos en los que los datos incompletos no son admisibles, por lo tanto debemos solucionarlos de alguna forma.

El proceso de imputación de valores faltantes puede hacerse de muchas formas, una forma muy simple es completar con el valor promedio para atributos numéricos y con el valor mas popular para atributos categóricos. Esto se realiza tanto con las habitaciones, garages, baños y antiguedad.

Por un otro, se puede ver que para toda propiedad con valor nulo en los metros totales, tiene valor en los metros cubiertos y viseversa. Por lo que se decide completar con este mismo.

Se completan los id zonas faltantes con un promedio de los ids para aquellas propiedades que comparten ciudades y en su defecto que comparten provincia. Y algo similar se realiza al momento de completar la longitud y latitud, verificando que el promedio se encuentre dentro del mapa.


## Clustering
Con la intención de ver si se encontraban clusters que agrupen a las propiedades, se utilizó el algoritmo T-SNE, que es el estado del arte para la representación de datos en dos dimensiones.

### LOFANO ESTA HACIENDO CLUSTERIG?

## PCA

Se utilizo PCA en un intento de manipular la dimensionalidad de los datos. La idea de PCA es encontrar las ”direcciones” principales de los datos, es decir, aquellas direcciones sobre las cuales podemos proyectar los datos reteniendo su variabilidad.

# Features

## Feature engineering

Con lo investigado del previo trabajo y todos los dataframes generados, se busca todo tipo de atributos de los usuarios, para que luego puedan ser seleccionados y aprovechados por los algoritmos a aplicar.

Primero se separan las features según su dependencia con el precio de la propiedad:

### Independientes del precio

#### Metros Totales y Cubiertos

- metrostotales_log, metroscubiertos_log: Escala logaritmica de los metros totales y cubiertos
- porcentaje_metros: Metros cubiertos sobre metros totales
- diferencia_metros: La diferencia entre los metros totales y cubiertos
- intervalo_metros_totales, intervalo_metros_cubiertos: Se agrupan en 5 intervalos dependiendo del tamaño de la propiedad
- metroscubiertos_bins_unif, metroscubiertos_bins_perc: Se generan n intervalos dependiendo del tamaño de la propiedad (intervalos uniformes y segun los percentiles)
- metros_totales_normalizados, metros_cubiertos_normalizados: Los metros totales y cubiertos normalizados
  
#### Por tipo de propiedad

- escomercial: Booleano que indica si la propiedad es del tipo comercial(Bodega comercial, centro comercial, etc.) o no(Casa, apartamento, etc.)
- tipo_propiedad_compartida:  Booleano que indica si la propiedad es compartida(Casa en condominio, Duplex, etc.) o no(Casa, apartamento, etc.)
- promedio_metros_tipo_propiedad, promedio_metros_cub_tipo_propiedad: el promedio de metros que tiene cada tipo de propiedad
- prop_frecuente: Aquellas propiedades que aparecen en el mercado mayor cantidad de veces.

#### Por ubicación

- zona: Divición del mapa por Norte, Centro o Sur
- top_provincia: Ranking de provincias ordenadas de las mas caras a las menos(No depende del precio especificamente, solo se basa en los datos encontrados en el primer trabajo práctico).
- es_ciudad_centrica: Booleano que indica si la ciudad es la más poblada de la provincia(Ej: en Distrito Federal,Benito Juárez)
- promedio_metros_totales_provincia, promedio_metros_cubiertos_provincia : el promedio de metros que tiene cada provincia

#### Por fecha

- anio,mes,dia: Separo del feature fecha en día, mes y año 
- trimestre: Indico en que trimestre del año pertenece la propiedad
- dias_desde_datos, meses_desde_datos: Distancias entre las diferentes publicaciones

#### Propiedades booleanas

- escuelas_centros_cercanos: Verifica si tiene escuelas cercanas, centros comerciales cercanos, ambos o ninguno
- delincuencia: Booleando indicando si es una ciudad con alta tasa de delincuencia
- turismo: Booleano indicando si es una ciudad considerada turística
- es_antigua: Booleando que indica si la propiedad tiene más de 30 de antiguedad, indicando si es o no antigua.
- antiguedad_bins_unif, antiguedad_bins_perc: Divición en n intervalos uniformes o divididos según los percentiles para la antiguedad de la propiedad
 
#### Por cantidad de habitaciones, garages y baños

- cantidad_inquilinos: Promedio aproximado de personas que viven en la propiedad según la cantidad de  habitaciones, garages y baños
- tam_ambientes: Los metros cubiertos dividido la cantidad de habitaciones y baños

### Dependientes del precio

#### Por ubicación

- promedio_precio_provincia, promedio_provincia_log: Agrupo por provincia y calculo el promedio del precio de cada una. Se genera un feature con el promedio normal y otro con una escala logaritmica
- promedio_precio_ciudad, promedio_ciudad_log: Idem para cada una de las ciudades de México.
- promedio_precio_ciudad_gen: Se generaliza el feature
- varianza_precio_ciudad: Agrupo por ciudades y calculo la varianza del precio.
- count_ciudad: Agrupo por ciudades y calculo la cantidad de propiedades que hay en esta.
- promedio_id_zona, promedio_id_zona_log, promedio_id_zona_gen, varianza_id_zona count_id_zona:: Idem provincia y ciudades.

#### Por tipo de propiedad

- promedio_precio_tipo_propiedad: Agrupo por tipo de propiedad y le asigno el promedio del precio
- promedio_precio_tipo_propiedad_ciudad,  promedio_precio_tipo_propiedad_ciudad_gen: Agrupo tanto por el tipo de propiedad como para la ciudad a la que pertence y le asigno el precio promedio
- count_tipo_propiedad: Agrupo por tipo de propiedad y calculo la cantidad de propiedades de ese mismo tipo hay.
- count_tipo_propiedad_ciudad:  Agrupo por tipo de propiedad  y ciudad y calculo la cantidad de propiedades de ese mismo tipo hay en cada ciudad.

#### Por fecha

- promedio_por_mes: Se calcula el promedio de precios por cada mes del año.
- varianza_por_mes:  Se calcula la varianza de precios por cada mes del año.


#### Por cantidad de habitaciones, garages y baños

- promedio_precio_habitaciones: Agrupo por cantidad de habitaciones y calculo el precio promedio.
- promedio_precio_banos_garages: Idem pero agrupo por cantidad de baños y garages.
- promedio_precio_habitaciones_banos_garages: Idem pero agrupo por cantidad de habitaciones, garages y baños.
- promedio_precio_hbg_tipo_propiedad: Idem pero agrupo también por tipo de propiedad
- promedio_precio_hbg_tipo_propiedad_provincia: Idem pero también agrupo por provincia.
- promedio_precio_hbg_tipo_propiedad_provincia_gen: Generalizo el último feature
  
#### Por propiedades booleanas

- promedio_precio_booleanos: Agrupo por las propiedades booleanas(si tiene o no piscina, gimnasio y usos multiples) y calculo el precio promedio.

#### Por puntajes(Insight)

- puntaje: A partir de lo aprendido en el Trabajo Práctico I, se realizo un importante aporte a Navent donde calificabamos cada una de las propiedades con el objetivo de poder encontrar el 'precio ideal' para todas. Por ende teniendo en cuenta: la piscina, gimnasio, usos multiples, cantidad de baños, habitaciones, garages, metros totales, cubiertos y provincia se van a rankear todas las propiedades.

### En relación a los textos

idf_titulo, idf_descripcion: La idea de TF-IDF es darle a cada término un peso que sea inversamente proporcional a su frecuencia. Los términos que aparecen en muchas propiedades serán entonces menos importantes que los términos que solo
aparecen en unos pocos.El IDF de un término se calcula de la forma: $ IDF(t_i) = log(\frac{N + 1}{f_(t_i))} Donde N es la cantidad de documentos y $f_(t_i)$ es la cantidad de registros en los que aparece el término. Y TF es el term frequency. Se calculan tanto para el título, como para la descripción de cada propiedad.

peso_titulo, peso_descripcion: Un contador de palabras importantes en el título y la descripción

### Distancias

- distancia_ciudad_centrica: Con la distancia Euclediana se encuentra la distancia entre la propiedad con la ciudad más importante de la Erovincia.
es un conjunto de árboles de decisión en donde cada
árbol usa un bootstrap del set de entrenamiento y un cierto conjunto de atri utos
tomados al azar.

- distancia_centro_mexico: Con la distancia Euclediana se encuentra la distancia entre la propiedad y en centro de México: Distrito Federal


### One Hot Encoding

Consiste en dividir el atributo en tantas columnas como valores posibles puede tener y usar cada columna como un dato binario indicando si el atributo toma o no dicho valor. Por lo que esta idea se repite para todos los features categóricos: provincia, tipodepropiedad, zona, etc.

## Feature Selection
Una vez que se tienen todos los atributos en un mismo dataframe y luego de un par de pruebas se ve que no siempre hay que entrenar los modelos con la totalidad del dataframe. Viendo que para algunos algoritmos el orden y la
selección de los features logrababa distintos resultados, se busca la forma de encontrar la combinación óptima de features y eliminar todo el ruido posible.

### Random Forest

Es un conjunto de árboles de decisión en donde cada
árbol usa un bootstrap del set de entrenamiento y un cierto conjunto de atributos tomados al azar. Es una
aplicación directa de bagging a árboles de decisión pero con una diferencia, cada árbol no usa el total de atributos sino un subset de los mismos.
suele producir buenos resultados para la mayorı́a de los
sets de datos, esto se debe a su habilidad para evitar overfitting y la fuerza de funcionar en base a un ensamble.

### SelectKBest

Elige los mejores features tomando los k que cuenten con el mayor puntaje.

### SelectFromModel

Métodoque que transforma los features en pesos y se basa en la importancia de cada uno de ellos

### Recursive Feature Elimination (RFE)

RFE fue la herramienta que nos dejo los mejores resultados. Se trata de un metodo que va eliminando aquellos features que sean debiles hasta quedarse con la cantidad establecida. Para ello, rankea todos los features y a través de una eliminación recursiva obtengo el resultado final.

### Recursive Feature Elimination w/ Cross Validation (RFECV)

Selecciona los mejores subsets de features usando RFE, y se queda con el mejor de ellos basandose en el puntaje obtenido con cross-validation.



# Modelos

## XGBoost

XGBoost es un algoritmo muy eficiente de gradient boosting en árboles. Uno de sus híper-parámetros más importantes es la funcion objetivo, de la cual se usa la logística binaria, ya que se tiene un problema de clasificación binaria y
no uno de regresión

## KNN

Un algoritmo sencillo utilizado es el de los K vecinos más cercanos. Este algoritmo consiste en encontrar los vecinos más cercanos del punto a clasificar, y luego simplemente predecir que el punto en cuestión es de la clase del cual la
mayoría de sus vecinos sean parte. Sus híper-parámetros a definir es la cantidad de vecinos a tomar en cuenta y la distancia a utilizar entre ellos. Es tal vez en su sencillez que está una de sus mayores desventajas: es un
algoritmo de orden cuadrático, y esto impacta mucho sobre la ejecución decada modelo que se hace. Otra cosa a remarcar es que los datos deben estar normalizados a la hora de procesarlos.

## Redes neuronales

### Keras


## LightGBM
LightGBM es simplemente el algoritmo que constantemente mejores resultados nos dio. Este algoritmo de gradient boosting sobre árboles se diferencia de XGBoost en que construye los árboles según las hojas, y no los niveles. Es
importante que sus híper-parámetros estén bien configurados (por ejemplo, la profundidad máxima de los árboles), para que rápidamente se encuentran muy buenos saltos de calidad en el modelo.
Este algoritmo también se destaca por ser rápido y consumir poca memoria. También, tiene un gran manejo de la  dimensionalidad de los datos, sin cambiar mucho frente a ellos.

## Parameter tuning
En los primeros modelos corridos fue cuando se empezo a notar lo que ya se sabía: los híper-parámetros son muy importantes y puede marcar la diferencia entre un buen modelo y uno promedio o incluso malo.
Inicialmente, debido a la gran cantidad de opciones para algunos algoritmos, optamos por utilizar hyperopt, 



## Ensambles
### Stacking
### Blending


# Resultados obtenidos

# Conclusiones
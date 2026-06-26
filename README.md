# Análisis Epidemiológico mediante Reglas de Asociación en Indicadores de Salud (BRFSS 2015)

Este repositorio contiene la implementación, experimentación y documentación del proyecto semestral para las entregas unificadas P2 + P3 del curso de Ingeniería de Datos / Machine Learning.

## Integrantes
*   Julio Contreras
*   Sebastian Aillapan
*   Orlando Caullan

---

## Tabla de Contenidos
1.  [Descripción del Proyecto](#descripción-del-proyecto)
2.  [El Dataset de Origen (BRFSS 2015)](#el-dataset-de-origen-brfss-2015)
3.  [Preprocesamiento y Limpieza de Ruido](#preprocesamiento-y-limpieza-de-ruido)
4.  [Modelamiento y Técnica Aplicada](#modelamiento-y-técnica-aplicada)
5.  [Diseño Experimental y Comparación de Rendimiento](#diseño-experimental-y-comparación-de-rendimiento)
6.  [Resultados y Reglas Clínicas Destacadas](#resultados-y-reglas-clínicas-destacadas)
7.  [Explorador Interactivo de Reglas](#explorador-interactivo-de-reglas)
8.  [Estructura del Proyecto](#estructura-del-proyecto)
9.  [Instrucciones de Ejecución (Reproducibilidad)](#instrucciones-de-ejecución-reproducibilidad)

---

## Descripción del Proyecto

El objetivo de este proyecto es realizar un **Análisis Epidemiológico de Comorbilidades** y factores de riesgo en el estilo de vida de la población estadounidense en el año 2015. 

### Enfoque de Aprendizaje No Supervisado:
A diferencia de los modelos supervisados tradicionales (como clasificación o regresión), este proyecto emplea **Minería de Reglas de Asociación**. 
*   **Sin división Train/Test**: Al ser una tarea no supervisada de descubrimiento de patrones descriptivos en toda la población, no se realiza una división de entrenamiento y prueba. Se utiliza la totalidad del conjunto de datos para garantizar la precisión estadística de las métricas (Soporte y Confianza).
*   **Tratamiento de Variables**: No se define una variable objetivo fija (como predecir la diabetes). La columna de diabetes (`Diabetes_012`) se transforma en un elemento o ítem descriptivo adicional dentro de la "canasta" de características de salud de cada encuestado, lo que permite mapear relaciones de comorbilidad multidireccionales.

---

## El Dataset de Origen (BRFSS 2015)

Utilizamos el dataset **Diabetes Health Indicators (BRFSS 2015)** de Kaggle, derivado del Sistema de Vigilancia de Factores de Riesgo en el Comportamiento del CDC.
*   **Registros**: 253.680 encuestas individuales (transacciones).
*   **Variables**: 22 columnas iniciales que recogen diagnósticos clínicos (diabetes, hipertensión, infartos, accidentes cerebrovasculares), comportamientos (tabaquismo, alcoholismo, dieta, ejercicio) y datos socioeconómicos/demográficos.

---

## Preprocesamiento y Limpieza de Ruido

Para evitar la generación de miles de reglas obvias o redundantes que carecen de interés clínico, aplicamos una rigurosa limpieza y selección de columnas antes de codificar la matriz *one-hot* (`df_onehot`):

1.  **Exclusión de variables de "Alto Soporte" (Ruido Estadístico)**:
    *   `AnyHealthcare` (Tiene cobertura médica: **95.1% de soporte**).
    *   `CholCheck` (Chequeo de colesterol en los últimos 5 años: **96.3% de soporte**).
    *   *Justificación*: Al estar presentes en casi la totalidad de la muestra, se acoplan de forma espuria a cualquier combinación (ej. `{diabetes} -> {cobertura_médica}`), inundando los resultados con reglas sin utilidad médica.
2.  **Exclusión de variables socioeconómicas y demográficas**:
    *   Eliminamos `Education` (Educación), `Income` (Ingresos) y `NoDocbcCost` (no consultó al médico por dinero) para evitar que los algoritmos infieran reglas puramente sociológicas (como `ingreso_alto -> educacion_universitaria`).
    *   Eliminamos `Sex` (Sexo) y `Age` (Edad) para evitar reglas obvias de envejecimiento natural (como `edad_avanzada -> dificultad_caminar`), concentrando el análisis estrictamente en patologías clínicas y hábitos de conducta.
3.  **Discretización de Variables Continuas**:
    *   `BMI` (IMC): Discretizado en los 6 rangos oficiales de la OMS (Bajo peso, Normal, Sobrepeso, y Obesidad clase I, II y III).
    *   `PhysHlth` (Salud física mala en días) y `MentHlth` (Salud mental mala en días): Discretizados en rangos (1-7 días, 8-14 días, 15-30 días).

---

## Modelamiento y Técnica Aplicada

Comparamos dos algoritmos fundamentales de minería de patrones frecuentes:
1.  **Apriori**: Estrategia clásica basada en búsqueda por niveles (*level-wise*). Genera de forma iterativa candidatos de tamaño $k+1$ a partir de los frecuentes de tamaño $k$ y requiere múltiples escaneos del dataset, lo que puede provocar una explosión combinatoria.
2.  **FP-Growth (Frequent Pattern Growth)**: Compone una estructura de árbol compacta en memoria llamada **FP-Tree**. Extrae los itemsets frecuentes directamente del árbol sin generar candidatos explícitos y realizando exactamente 2 escaneos del dataset.

---

## Diseño Experimental y Comparación de Rendimiento

El experimento consiste en un **barrido de parámetros** sobre el soporte mínimo (`0.10`, `0.05`, `0.03`, `0.02`), evaluando de forma consecutiva la velocidad de procesamiento de ambos modelos con una confianza del `0.60`.

Los resultados reales de rendimiento en la máquina de ejecución (Google Colab) son:

| Soporte Mínimo (`min_support`) | Itemsets Frecuentes | Reglas Generadas | Tiempo Apriori (seg) | Tiempo FP-Growth (seg) |
| :---: | :---: | :---: | :---: | :---: |
| **0.10 (10%)** | 157 | 165 | 0.2637 s | 84.0506 s |
| **0.05 (5%)** | 435 | 392 | 0.7896 s | 248.2944 s |
| **0.03 (3%)** | 811 | 730 | 1.9164 s | 477.0526 s |
| **0.02 (2%)** | 1275 | 1034 | 2.7589 s | 753.7672 s |

### Análisis de la Paradoja de Rendimiento (Apriori vs FP-Growth)
Aunque la teoría indica que FP-Growth es computacionalmente más eficiente que Apriori, en nuestro escenario **Apriori fue hasta 270 veces más rápido**:
1.  **Dimensión de Columnas**: Tras la limpieza de ruido, el dataset one-hot se redujo a solo 26 ítems (columnas). Con tan pocas variables, el costo combinatorio de candidatos para Apriori es mínimo, terminando en milisegundos usando las funciones vectorizadas optimizadas en C que posee Pandas.
2.  **Volumen de Filas y Overhead de Python**: El dataset cuenta con **253.680 filas**. La función `fpgrowth` de la librería `mlxtend` está programada en Python puro, lo que obliga al procesador a iterar a través de las 253.000 transacciones fila por fila en bucles para construir y recorrer las ramas del árbol `FP-Tree`, generando un grave cuello de botella computacional.

---

## Resultados y Reglas Clínicas Destacadas

Aplicando los filtros óptimos de nuestro explorador (`min_support=0.04`, `min_confidence=0.70`, `min_lift=1.5`, `max_rule_size=4`), el modelo filtra un conjunto robusto de **39 reglas**. Para el póster y la presentación clínica, se seleccionaron los siguientes **3 Casos de Uso**:

### Caso de Uso 1: Paciente diabético con movilidad reducida
*   **Regla**: `{diabetes, dificultad_caminar} -> {hipertension}`
*   **Métricas**: Soporte: **4.25%** | Confianza: **82.2%** | Lift: **1.92**
*   **Aplicación**: Pacientes diabéticos con problemas motores tienen un 82% de probabilidad de tener hipertensión. Útil para activar alertas tempranas de control de presión en visitas domiciliarias del CESFAM.

### Caso de Uso 2: Paciente con antecedentes cardiovasculares graves
*   **Regla**: `{enfermedad_cardiaca_o_infarto, hipertension} -> {colesterol_alto}`
*   **Métricas**: Soporte: **5.39%** | Confianza: **76.2%** | Lift: **1.80**
*   **Aplicación**: Guía la co-prescripción agresiva de estatinas y antihipertensivos en el tratamiento post-infarto en centros de rehabilitación cardiovascular.

### Caso de Uso 3: Alerta en Control de Síndrome Metabólico
*   **Regla**: `{colesterol_alto, diabetes} -> {hipertension}`
*   **Métricas**: Soporte: **7.57%** | Confianza: **81.1%** | Lift: **1.89**
*   **Aplicación**: Afecta a más de 19.200 personas de la muestra. Automatiza derivaciones prioritarias a control de presión arterial cuando se registra un perfil lipídico alterado en pacientes diabéticos.

---

## Explorador Interactivo de Reglas

El notebook implementa un explorador visual al final de la celda 37 mediante `ipywidgets`. Permite al usuario (médico o epidemiólogo) filtrar las reglas resultantes en tiempo real mediante deslizadores dinámicos de:
*   `min_support` (Soporte Mínimo)
*   `min_confidence` (Confianza Mínima)
*   `min_lift` (Lift Mínimo)
*   `max_rule_size` (Tamaño máximo de ítems en la regla)

---

## Estructura del Proyecto

```text
├── diabetes_BRFSS2015.csv         # Dataset original de indicadores de salud
├── notebook.ipynb                 # Jupyter Notebook con el código y visualizaciones
├── reglas_apriori.csv             # Reglas completas generadas por Apriori
├── reglas_fpgrowth.csv            # Reglas completas generadas por FP-Growth
├── README.md                      # Documentación del proyecto (este archivo)
└── poster_fisico.jpg              # Fotografía de respaldo del póster hecho a mano
```

---

## Instrucciones de Ejecución (Reproducibilidad)

### Entorno Local (Jupyter Notebook)
1.  Clona el repositorio:
    ```bash
    git clone https://github.com/[Tu-Usuario]/Analisis-Epidemiologico.git
    cd Analisis-Epidemiologico
    ```
2.  Crea un entorno virtual e instala las dependencias:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install pandas numpy mlxtend matplotlib seaborn ipywidgets
    ```
3.  Ejecuta Jupyter:
    ```bash
    jupyter notebook
    ```
4.  Abre `notebook.ipynb` y ejecuta todas las celdas en orden.

### Entorno Cloud (Google Colab)
1.  Abre Google Colab y sube el archivo `notebook.ipynb`.
2.  Sube el archivo `diabetes_BRFSS2015.csv` al directorio raíz (`/content/`).
3.  En el menú superior, selecciona **Entorno de ejecución** -> **Ejecutar todo** (*Runtime -> Run all*). Las dependencias se instalarán automáticamente en la celda 3.

# Predicción del precio por noche de alojamientos Airbnb en Ciudad de México

Proyecto final de **Machine Learning (Aprendizaje Supervisado)** · Maestría en Inteligencia Artificial
**Autor:** Jaime Astudillo · noviembre de 2026

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jastk45/airbnb-cdmx-ml/blob/main/proyecto_final_airbnb_cdmx.ipynb)

## Problema

Estimar el **precio por noche publicado** de alojamientos completos (departamentos y casas enteras) de Airbnb en Ciudad de México a partir de las características del anuncio: capacidad, ubicación, reseñas, servicios y la descripción en texto libre. El resultado es una estimación del precio de oferta, no del precio pagado.

El notebook es el informe del proyecto: incluye la motivación, la metodología, el EDA, los modelos, los resultados, el análisis, las conclusiones y las referencias.

## Resultados principales

| Modelo | MAE CV anidada (MXN) | MAE prueba (MXN) | RMSE prueba (MXN) | R² prueba |
|---|---|---|---|---|
| Baseline (mediana) | 982 | 1 066 | 2 320 | −0.06 |
| Ridge (optimizado) | 706 | — | — | — |
| **Random Forest (optimizado)** | **640** | **739** | **1 784** | **0.37** |

- Random Forest tuvo menor MAE que Ridge en los 20 folds de la validación cruzada anidada (Optuna se repite dentro de cada fold externo), pero con la prueba t corregida de Nadeau-Bengio la diferencia no es significativa al 5 % (p = 0.058).
- Todas las particiones se agrupan por anfitrión, así que sus anuncios nunca quedan a ambos lados.
- El modelo final se eligió solo con validación cruzada y se evaluó una sola vez en prueba. Esa partición resultó más difícil que el promedio de la CV: el baseline también empeora (1 066 frente a 982 MXN).

## Metodología

1. **Datos.** *Snapshot* de Ciudad de México del 23 de septiembre de 2026 de [Inside Airbnb](https://insideairbnb.com/get-the-data/) (licencia CC BY 4.0). Filtros documentados con conteos: precio publicado, alojamiento completo, anuncio activo, rango de precio, estancia mínima y duplicados exactos. De 30 468 anuncios quedan 15 412 elegibles, de los que uso una muestra aleatoria de 5 000 por costo computacional.
2. **Partición.** 80 % entrenamiento y 20 % prueba con `StratifiedGroupKFold`: agrupada por anfitrión y estratificada por quintiles de precio, con semilla 42. Se hace antes del EDA y de ajustar cualquier transformación, y las validaciones cruzadas también se agrupan por anfitrión.
3. **EDA** solo con entrenamiento.
4. **Feature extraction con transfer learning.** Las descripciones se convierten en vectores de 384 dimensiones con el modelo preentrenado `paraphrase-multilingual-MiniLM-L12-v2` (Sentence-Transformers, *mean pooling*, máximo 128 tokens).
5. **Pipeline de scikit-learn.** Imputación, escalado, *one-hot*, PCA del texto (con *whitening*, para que sus componentes pesen como las demás variables), selección tipo filter (`SelectPercentile` con `f_regression`) y el modelo. El objetivo se entrena en escala log con `TransformedTargetRegressor`, pero las métricas se reportan en MXN.
6. **Optimización.** Optuna (TPE, 30 pruebas por modelo) con validación cruzada agrupada de 2 repeticiones con 5 folds sobre entrenamiento. Incluye las curvas de validación de λ (Ridge) y de `min_samples_leaf` (Random Forest), con entrenamiento y validación, y las curvas de aprendizaje de ambos modelos.
7. **Comparación estadística.** Ridge frente a Random Forest con validación cruzada anidada: CV externa agrupada de 2 repeticiones con 10 folds y, dentro de cada fold externo, la misma búsqueda de Optuna con solo su entrenamiento. La guía del curso permite la validación anidada aunque no la recomienda por su costo; la uso para que la comparación no reutilice los datos con que se eligieron los hiperparámetros. Se reportan la prueba t corregida (Nadeau-Bengio) y Wilcoxon como contraste.
8. **t-SNE.** Proyección de lo que ve el modelo, es decir, la salida del pipeline justo antes del regresor, coloreada por quintil de precio.
9. **Interpretación.** Residuos, errores por rango de precio e importancia por permutación agrupada en prueba.

## Cómo reproducir

El notebook ya incluye todas las salidas, así que no hace falta ejecutarlo para revisarlo.

**En Google Colab:** abre el notebook con el botón de arriba y ejecuta *Entorno de ejecución → Ejecutar todo*. La primera celda instala `sentence-transformers` y `optuna`, y el notebook descarga los datos de Inside Airbnb y el modelo de lenguaje. Lo más lento es la validación anidada (40 búsquedas de Optuna): en la máquina de 24 hilos donde lo ejecuté, el notebook completo tardó unos 22 minutos; en el Colab gratuito (2 núcleos) puede tardar varias horas.

**En local** (probado con Python 3.13 y las versiones exactas de `requirements.txt`):

```bash
pip install -r requirements.txt notebook
jupyter notebook proyecto_final_airbnb_cdmx.ipynb
```

Si existe `data/listings.csv.gz`, el notebook usa esa copia; si no, descarga el archivo desde Inside Airbnb. Inside Airbnb puede retirar los *snapshots* antiguos: si la URL deja de responder, hay que conseguir el archivo del 23 de septiembre de 2026 y guardarlo en `data/listings.csv.gz`.

## Estructura

```
.
├── proyecto_final_airbnb_cdmx.ipynb   # notebook ejecutado: informe y código
├── requirements.txt                   # versiones probadas
├── README.md
└── .gitignore                         # excluye data/ (los datos se descargan al ejecutar)
```

## Datos y privacidad

Los datos son públicos y se descargan de Inside Airbnb al ejecutar el notebook; no se incluyen en el repositorio. No uso como variables datos personales de los anfitriones (nombres, fotos o identificadores), ni el identificador o la URL del anuncio, ni variables calculadas a partir del precio. El id del anfitrión solo se usa para agrupar las particiones. Las variables usadas se describen en la Sección 2 del notebook.

## Uso de inteligencia artificial

El código se generó con asistencia de Claude (Anthropic). Como pide la guía del curso, cada celda de código del notebook empieza con un comentario `# Prompt (IA, Claude): «…»` con la instrucción usada para esa celda.

## Dependencias

numpy, pandas, scipy, scikit-learn, joblib, matplotlib, seaborn, optuna, sentence-transformers y torch, con las versiones de `requirements.txt`.

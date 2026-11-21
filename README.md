# Predicción del precio por noche de alojamientos Airbnb en Ciudad de México

Proyecto final de **Machine Learning (Aprendizaje Supervisado)** · Maestría en Inteligencia Artificial
**Autor:** Jaime Astudillo · noviembre de 2026

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jastk45/airbnb-cdmx-ml/blob/main/proyecto_final_airbnb_cdmx.ipynb)

## Problema

Estimar el **precio por noche publicado** de alojamientos completos (departamentos y casas enteras) de Airbnb en Ciudad de México a partir de las características del anuncio: capacidad, ubicación, reseñas, servicios y la descripción en texto libre. El resultado es una estimación del precio de oferta, no del precio pagado.

El notebook es el informe del proyecto: incluye la motivación, la metodología, el EDA, los modelos, los resultados, el análisis, las conclusiones y las referencias.

## Resultados principales

| Modelo | MAE CV 10×10 (MXN) | MAE prueba (MXN) | RMSE prueba (MXN) | R² prueba |
|---|---|---|---|---|
| Baseline (mediana) | 1 003 | 984 | 1 988 | −0.06 |
| Ridge (optimizado) | 721 | — | — | — |
| **Random Forest (optimizado)** | **647** | **631** | **1 314** | **0.54** |

- Random Forest es significativamente mejor que Ridge: prueba t corregida de Nadeau-Bengio sobre 10×10 folds emparejados, p = 0.006.
- El modelo final se eligió solo con validación cruzada y se evaluó una sola vez en el conjunto de prueba.

## Metodología

1. **Datos.** *Snapshot* de Ciudad de México del 23 de septiembre de 2026 de [Inside Airbnb](https://insideairbnb.com/get-the-data/) (licencia CC BY 4.0). Filtros documentados con conteos: precio publicado, alojamiento completo, anuncio activo, rango de precio, estancia mínima y duplicados. De 30 468 anuncios quedan 15 412 elegibles, de los que uso una muestra aleatoria de 5 000 por costo computacional.
2. **Feature extraction con transfer learning.** Las descripciones se convierten en vectores de 384 dimensiones con el modelo preentrenado `paraphrase-multilingual-MiniLM-L12-v2` (Sentence-Transformers, *mean pooling*, máximo 128 tokens).
3. **Pipeline de scikit-learn.** Imputación, escalado, *one-hot*, PCA del texto, selección tipo filter (`SelectPercentile` con `f_regression`) y el modelo. El objetivo se entrena en escala log con `TransformedTargetRegressor`, pero las métricas se reportan en MXN.
4. **Partición.** 80 % entrenamiento y 20 % prueba, estratificada por quintiles de precio, con semilla 42. La partición se hace antes de ajustar cualquier transformación.
5. **Optimización.** Optuna (TPE, 30 pruebas por modelo) con validación cruzada repetida 2×5 sobre entrenamiento. Incluye la curva de validación de λ (entrenamiento y validación) y las curvas de aprendizaje de ambos modelos.
6. **Comparación estadística.** Ridge frente a Random Forest, ambos en su versión optimizada, con validación cruzada repetida 10×10. Se reportan la prueba t corregida (Nadeau-Bengio) y Wilcoxon como contraste.
7. **t-SNE.** Proyección de lo que ve el modelo, es decir, la salida del pipeline justo antes del regresor, coloreada por quintil de precio.
8. **Interpretación.** Residuos, errores por rango de precio e importancia por permutación agrupada en prueba.

## Cómo reproducir

**En Google Colab:** abre el notebook con el botón de arriba y ejecuta *Entorno de ejecución → Ejecutar todo*. La primera celda instala `sentence-transformers` y `optuna`, y el notebook descarga los datos de Inside Airbnb y el modelo de lenguaje. Con un entorno con GPU el texto se procesa en segundos. El Random Forest es lo más lento: en la máquina de 24 núcleos donde lo ejecuté, el notebook completo tardó unos 4 minutos; en el Colab gratuito (2 núcleos) puede tardar bastante más.

**En local:**

```bash
pip install -r requirements.txt
jupyter notebook proyecto_final_airbnb_cdmx.ipynb
```

Si existe `data/listings.csv.gz`, el notebook usa esa copia; si no, descarga el archivo desde Inside Airbnb.

## Estructura

```
.
├── proyecto_final_airbnb_cdmx.ipynb   # notebook ejecutado: informe y código
├── requirements.txt                   # dependencias
├── README.md
└── .gitignore                         # excluye data/ (los datos se descargan al ejecutar)
```

## Datos y privacidad

Los datos son públicos y se descargan de Inside Airbnb al ejecutar el notebook; no se incluyen en el repositorio. No uso como variables datos personales de los anfitriones (nombres, fotos o identificadores), ni el identificador o la URL del anuncio, ni variables calculadas a partir del precio. Las variables usadas se describen en la Sección 2 del notebook.

## Uso de inteligencia artificial

El código se generó con asistencia de Claude (Anthropic). Como pide la guía del curso, cada celda de código del notebook empieza con un comentario `# Prompt (IA, Claude): «…»` con la instrucción usada para esa celda.

## Dependencias

Python 3.10 o superior: numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, optuna, sentence-transformers y torch (ver `requirements.txt`).

# Proyecto Equipo 6 — Familia 3: Probabilísticos y Estructurados

Actividad práctica: **Despliegue y Análisis Comparativo de Familias de Aprendizaje Supervisado**.

Modelos implementados:

1. **Naive Bayes** — `MultinomialNB` y `ComplementNB` para clasificación de texto.
2. **Perceptrón Multicapa para regresión** — `MLPRegressor`.
3. **Análisis Discriminante** — `LDA` y `QDA`.

Todo el código, los comentarios y la interfaz están en español. La semilla global es `random_state=42`.

---

## 1. Instalación

Requisitos: **Python 3.10+** (probado con Python 3.11 en Windows).

```powershell
cd "C:\Users\Josh\Desktop\Opencode\Comparativo de Familias\proyecto_equipo6"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si la política de ejecución de PowerShell bloquea la activación:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

## 2. Cómo ejecutar

### 2.1 Todos los experimentos

```powershell
python -m src.run_all
```

Tiempo aproximado en el equipo de referencia: **~14 minutos** (la búsqueda del MLP es la parte más costosa). Genera automáticamente:

- `results/figures/`: figuras PNG a 300 DPI y una visualización 3D en HTML.
- `results/tables/`: métricas, `cv_results_`, reportes y tablas de sensibilidad.
- `results/models/`: modelos entrenados con `joblib`.

### 2.2 Simulación interactiva

```powershell
python -m streamlit run app/streamlit_app.py
```

Alternativa si el ejecutable `streamlit` está en el PATH:

```powershell
streamlit run app/streamlit_app.py
```

La aplicación abre una pestaña por modelo más una pestaña de comparación global.

### 2.3 Verificación rápida

```powershell
python -m compileall src app
```

---

## 3. Datasets y justificación

| Modelo | Dataset | Motivo |
|---|---|---|
| Naive Bayes | **20 Newsgroups** (4 categorías: `comp.graphics`, `rec.sport.baseball`, `sci.med`, `talk.politics.mideast`) | Clasificación de texto realista y multiclase; permite comparar conteos vs TF-IDF, n-gramas y el efecto de `alpha`. También se construye una versión **desbalanceada** submuestreando una clase para comparar `MultinomialNB` vs `ComplementNB`. |
| MLP Regressor | **California Housing** (`fetch_california_housing`) | Regresión continua con 8 variables numéricas y 20 640 muestras; tamaño suficiente para mostrar curvas de aprendizaje y regularización. Fallback automático a **Diabetes** y, si también falla, a `make_regression`. |
| LDA / QDA | **Breast Cancer** (principal) y **Wine** (proyección discriminante) | Breast Cancer es un benchmark binario de 30 variables numéricas, adecuado para comparar LDA y QDA. Wine tiene 3 clases y permite proyectar en 2 componentes discriminantes. Se añade un **dataset sintético 2D** con covarianzas muy distintas por clase para evidenciar cuándo QDA supera a LDA. |

**Caché:** 20 Newsgroups se descarga en `data/` (`20news-bydate_py3.pkz`). Si no hay internet, `src/data_loader.py` genera un **corpus sintético en español** con 4 categorías para que el pipeline no se interrumpa; en ese caso los resultados de texto no son comparables con los de 20 Newsgroups.

**Variables categóricas:** ninguno de los datasets elegidos contiene variables categóricas. Aun así, `src/preprocessing.py` implementa `OneHotEncoder` dentro de un `ColumnTransformer` para que el código sea reutilizable con datos mixtos. En los experimentos actuales ese bloque no se activa.

---

## 4. Preprocesamiento (sin fuga de datos)

- División **80/20**: estratificada en clasificación y aleatoria en regresión, con `random_state=42`.
- `StandardScaler` dentro de `Pipeline`:
  - **Obligatorio** en MLP.
  - **Sí** en LDA/QDA (decisión del equipo; LDA/QDA son invariantes a escala, pero el escalado estabiliza la estimación numérica y la regularización).
  - **No** en Naive Bayes Multinomial/Complement: requieren conteos o frecuencias no negativas. El comentario está explícito en `src/models_nb.py`.
- Vectorización de texto (`CountVectorizer` / `TfidfVectorizer`) dentro del `Pipeline`, por lo que el vocabulario se aprende solo con entrenamiento.
- `ColumnTransformer` + `OneHotEncoder` disponibles para datasets mixtos.
- No se usa el conjunto de prueba para ajustar escaladores, vectorizadores ni hiperparámetros.

---

## 5. Estructura del repositorio

```text
proyecto_equipo6/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── 20news-bydate_py3.pkz        # caché de 20 Newsgroups
├── docs/
│   └── fundamentos_matematicos.md   # Fase 2, con LaTeX
├── src/
│   ├── __init__.py
│   ├── config.py                    # semilla, rutas, parámetros
│   ├── data_loader.py               # datasets y fallbacks
│   ├── preprocessing.py             # split, ColumnTransformer, pipelines
│   ├── models_nb.py                 # MultinomialNB / ComplementNB
│   ├── models_mlp.py                # MLPRegressor
│   ├── models_discriminant.py       # LDA / QDA
│   ├── evaluation.py                # métricas y utilidades de gráficas
│   └── run_all.py                   # ejecuta todo y guarda resultados
├── results/
│   ├── figures/                     # PNG 300 DPI + HTML 3D
│   ├── tables/                      # CSV de métricas y cv_results_
│   └── models/                      # .joblib
└── app/
    └── streamlit_app.py             # simulación interactiva
```

---

## 6. Experimentos realizados

### 6.1 Naive Bayes

- **Búsqueda:** `GridSearchCV` con 5 folds estratificados.
- **Espacio:** `vectorizer__tipo` (`count`/`tfidf`), `ngram_range` ((1,1)/(1,2)), `min_df` (1/3), `clf__alpha` (0.1/0.5/1.0), `clf__fit_prior` (True/False).
- **Mejor configuración (ambas variantes):** TF-IDF, unigramas, `min_df=1`, `alpha=0.1`, `fit_prior=True`.
- **Conjunto desbalanceado:** se submuestrea la clase 0 al 10 % de la clase mayoritaria. Se comparan ambas variantes con su mejor `alpha`.
- **Sensibilidad:** `alpha ∈ {0.01, 0.05, 0.1, 0.5, 1, 2, 5}` con validación cruzada.
- **Top palabras:** `feature_log_prob_` por clase, top-15, guardado en CSV y PNG.

### 6.2 MLPRegressor

- **Búsqueda:** `RandomizedSearchCV`, 10 candidatos, 5 folds, sobre una submuestra del 50 % del entrenamiento para acotar el costo; el mejor modelo se reentrena con **todo** el conjunto de entrenamiento.
- **Espacio:** `hidden_layer_sizes`, `activation`, `alpha`, `learning_rate_init`, `solver`, `early_stopping`.
- **Mejor configuración:** `adam`, `tanh`, `(100, 50)`, `alpha=0.01`, `lr=0.01`, `early_stopping=True`, 135 iteraciones.
- **Artefactos:** curva de pérdida, curvas de aprendizaje, `permutation_importance`, real vs predicho, residuos y sensibilidad al tamaño de la red.

### 6.3 LDA / QDA

- **Búsqueda:** `GridSearchCV`, 5 folds.
  - LDA: `solver ∈ {svd, lsqr, eigen}` y `shrinkage ∈ {None, auto, 0.1, 0.5}`.
  - QDA: `reg_param ∈ {0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5}`.
- **Mejor LDA:** `solver=lsqr`, `shrinkage=0.1`.
- **Mejor QDA:** `reg_param=0.5`.
- **Visualizaciones:** fronteras 2D lado a lado en Breast Cancer (PCA-2) y en el dataset sintético; proyección discriminante y varianza por componente en Wine; dispersión PCA-3D interactiva en HTML.
- **Sensibilidad:** `reg_param`, `shrinkage`.

> **Nota sobre QDA:** en Breast Cancer la covarianza de una clase no es de rango completo, por lo que `reg_param=0.0` falla numéricamente. Por eso la búsqueda y el análisis de sensibilidad comienzan en `0.001`. Esto es una limitación real de QDA y se documenta explícitamente.

---

## 7. Resultados reales

Todos los valores provienen de `python -m src.run_all`. Pueden variar ligeramente por versión de scikit-learn, hardware y disponibilidad de los datos, pero la semilla está fijada.

### 7.1 Clasificación de texto (20 Newsgroups, 4 clases, test = 780 documentos)

| Modelo | Accuracy | F1 macro | F1 ponderado | ROC-AUC OvR macro | Entrenamiento (s) | Inferencia (s) |
|---|---:|---:|---:|---:|---:|---:|
| MultinomialNB | 0.9333 | 0.9336 | 0.9333 | 0.9952 | 1.334 | 0.155 |
| ComplementNB | **0.9423** | **0.9426** | **0.9424** | **0.9957** | 1.282 | 0.166 |

**Conjunto desbalanceado** (clase 0 reducida al 10 % de la mayoritaria):

| Modelo | Accuracy | F1 macro | F1 ponderado | F1 clase minoritaria |
|---|---:|---:|---:|---:|
| MultinomialNB | 0.9224 | 0.8946 | 0.9203 | 0.7723 |
| ComplementNB | **0.9301** | **0.9089** | **0.9324** | **0.7971** |

ComplementNB mejora tanto el F1 macro como el F1 de la clase minoritaria, consistente con su diseño para datos desbalanceados.

### 7.2 Regresión (California Housing, test = 4 128 muestras)

| Modelo | RMSE | MAE | R² | Entrenamiento (s) | Inferencia (s) |
|---|---:|---:|---:|---:|---:|
| MLPRegressor | 0.5080 | 0.3370 | 0.8031 | 33.273 | 0.031 |

(La variable objetivo de California Housing está expresada en cientos de miles de dólares; RMSE 0.508 ≈ 50 800 USD.)

### 7.3 LDA / QDA (Breast Cancer, test = 114 muestras)

| Modelo | Accuracy | F1 macro | ROC-AUC | Entrenamiento (s) | Inferencia (s) |
|---|---:|---:|---:|---:|---:|
| LDA | 0.9649 | 0.9619 | 0.9957 | 0.0160 | 0.0050 |
| QDA | 0.9649 | 0.9619 | 0.9954 | 0.0125 | 0.0064 |

En este dataset ambos empatan en test, aunque QDA obtiene mejor F1 macro en validación cruzada (0.9691 vs 0.9615) gracias a `reg_param=0.5`.

**Dataset sintético con covarianzas distintas** (clases con la misma media y covarianzas muy diferentes):

| Modelo | Accuracy | F1 macro | ROC-AUC OvR macro |
|---|---:|---:|---:|
| LDA | 0.3300 | 0.3140 | 0.5232 |
| QDA | **0.7067** | **0.7022** | **0.9063** |

Este es el escenario teórico donde QDA supera a LDA: LDA impone una covarianza compartida y una frontera lineal, mientras que QDA modela la forma de cada clase.

### 7.4 Comparación global (cualitativa)

| Familia | Modelo | Tarea | Métrica principal | Valor | Entrenamiento (s) |
|---|---|---|---|---:|---:|
| Naive Bayes | MultinomialNB | Clasificación | F1 macro | 0.9336 | 1.334 |
| Naive Bayes | ComplementNB | Clasificación | F1 macro | 0.9426 | 1.282 |
| Discriminante | LDA | Clasificación | F1 macro | 0.9619 | 0.016 |
| Discriminante | QDA | Clasificación | F1 macro | 0.9619 | 0.012 |
| Red neuronal | MLPRegressor | Regresión | RMSE | 0.5080 | 33.273 |

> **Aclaración:** Naive Bayes y LDA/QDA son **clasificadores**; MLPRegressor es un **regresor**. No es válido comparar directamente F1 con RMSE/R². La comparación es cualitativa y por costo computacional.

### 7.5 Figuras generadas

- `fronteras_breast_cancer_pca2.png`, `fronteras_sintetico_covarianzas.png`
- `breast_cancer_pca3d.html`
- `lda_proyeccion_wine.png`, `lda_qda_sensibilidad.png`
- `nb_desbalanceado_comparacion.png`, `sensibilidad_alpha_nb.png`, `palabras_*_*.png`
- `confusion_*.png`, `roc_*.png`
- `real_vs_pred_MLPRegressor.png`, `residuos_MLPRegressor.png`, `loss_curve_MLPRegressor.png`
- `mlp_learning_curve.png`, `mlp_permutation_importance.png`, `mlp_sensibilidad_arquitectura.png`

---

## 8. Decisiones y limitaciones

1. **4 categorías en 20 Newsgroups** para mantener el tiempo de búsqueda razonable. Con 6 categorías el experimento es más costoso y no cambia las conclusiones metodológicas.
2. **Conteos vs TF-IDF** se tratan como hiperparámetro mediante `VectorizadorSeleccionable`, de modo que `GridSearchCV` decide con validación. En ambos modelos ganó TF-IDF.
3. **Sin escalado en NB.** Es una restricción del modelo, no un olvido; está comentado en el código.
4. **Búsqueda del MLP sobre el 50 % del entrenamiento** para acotar el costo (10 candidatos × 5 folds). El modelo final se reentrena con todo el entrenamiento. Es una concesión práctica documentada.
5. **QDA y `reg_param=0`.** Se excluye porque una covarianza por clase no es de rango completo en Breast Cancer. Se usa un rango `0.001–0.5`.
6. **Dataset sintético con medias iguales.** Se diseñó así para aislar el efecto de las covarianzas: cualquier ventaja de QDA proviene de modelar la forma de cada clase, no de separar medias.
7. **Visualización 3D.** Se entrega una dispersión PCA-3D interactiva (HTML). No se dibuja una superficie de decisión 3D porque el espacio original tiene 30 dimensiones y una superficie 3D sobre PCA sería una simplificación engañosa.
8. **Comparación entre modelos.** Es cualitativa porque las tareas son distintas; la tabla global prioriza tiempos y tipo de tarea.
9. **Reproducibilidad.** `random_state=42` en divisiones, modelos, búsquedas y gráficas. Los tiempos dependen del hardware.
10. **Fallback de datos.** Si falla la descarga de 20 Newsgroups o California Housing, el pipeline continúa con datos sintéticos o Diabetes; el README y los mensajes de consola lo advierten. En esta ejecución se usaron los datasets reales.
11. **Streamlit entrena el MLP sobre una submuestra de 6 000 filas** para que la interacción sea fluida; los resultados de la simulación no pretenden replicar exactamente el experimento completo.

---

## 9. Checklist de requisitos prácticos

| Requisito | ¿Dónde se cumple? |
|---|---|
| División 80/20 estratificada | `src/preprocessing.py` → `dividir_datos`; usado en `run_all.py` |
| Escalado sin fuga (Pipeline) | `StandardScaler` dentro de pipelines en `models_mlp.py` y `models_discriminant.py` |
| NB sin escalado (comentado) | `src/models_nb.py`, docstring y comentarios |
| Categóricas con OneHotEncoder | `src/preprocessing.py` → `crear_preprocesador` (no aplica a los datasets actuales) |
| Datasets con fallback | `src/data_loader.py` |
| GridSearchCV / RandomizedSearchCV con 5 folds | `src/run_all.py` → `ejecutar_naive_bayes`, `ejecutar_mlp`, `ejecutar_lda_qda` |
| `cv_results_` en CSV | `results/tables/*_cv_results.csv` |
| Matriz de confusión | `results/figures/confusion_*.png`; `src/evaluation.py` |
| Precision, Recall, F1 macro/ponderado | `results/tables/classification_report_*.csv` y `*_resultados.csv` |
| ROC-AUC y curvas ROC | `results/tables/*_resultados.csv`, `results/figures/roc_*.png` |
| RMSE, MAE, R² | `results/tables/mlp_resultados.csv` |
| Real vs predicho y residuos | `results/figures/real_vs_pred_MLPRegressor.png`, `residuos_MLPRegressor.png` |
| Fronteras 2D LDA vs QDA | `results/figures/fronteras_*.png` |
| Fronteras en dataset sintético | `results/figures/fronteras_sintetico_covarianzas.png` |
| Versión 3D con Plotly | `results/figures/breast_cancer_pca3d.html` |
| Top palabras por clase | `results/tables/nb_*_palabras_informativas.csv`, `palabras_*.png` |
| Multinomial vs Complement en desbalance | `results/tables/nb_comparacion_desbalanceado.csv`, `nb_desbalanceado_comparacion.png` |
| Curvas de aprendizaje MLP | `results/figures/mlp_learning_curve.png` |
| Curva de pérdida MLP | `results/figures/loss_curve_MLPRegressor.png` |
| Permutation importance MLP | `results/tables/mlp_permutation_importance.csv`, PNG |
| Proyección discriminante LDA | `results/figures/lda_proyeccion_wine.png` |
| Varianza por componente | `results/tables/lda_wine_varianza_proyeccion.csv` |
| Comparación global con tiempos | `results/tables/comparacion_modelos.csv` |
| Sensibilidad a alpha | `results/tables/nb_sensibilidad_alpha.csv`, `sensibilidad_alpha_nb.png` |
| Sensibilidad a tamaño de red | `results/tables/mlp_sensibilidad_arquitectura.csv`, PNG |
| Sensibilidad a reg_param/shrinkage | `results/tables/lda_qda_sensibilidad.csv`, `lda_qda_sensibilidad.png` |
| Simulación Streamlit (NB) | `app/streamlit_app.py`, pestaña Naive Bayes |
| Simulación Streamlit (MLP) | pestaña MLP Regressor |
| Simulación Streamlit (LDA/QDA) | pestaña LDA / QDA |
| Simulación Streamlit (comparación) | pestaña Comparación global |
| Notas matemáticas con LaTeX | `docs/fundamentos_matematicos.md` |
| Ejecución integral | `python -m src.run_all` (verificado) |
| Simulación funcional | `python -m streamlit run app/streamlit_app.py` (verificado con `AppTest`, 0 excepciones) |

---

## 10. Comandos exactos de entrega

```powershell
# 1. Instalar
cd "C:\Users\Josh\Desktop\Opencode\Comparativo de Familias\proyecto_equipo6"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Reproducir todos los resultados
python -m src.run_all

# 3. Abrir la simulación interactiva
python -m streamlit run app/streamlit_app.py
```

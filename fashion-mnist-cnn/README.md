# 👕 Clasificador de prendas Fashion-MNIST (CRISP-ML + Scrum)

Proyecto de IA que clasifica imágenes de prendas del dataset **Fashion-MNIST** usando una
**Red Neuronal Convolucional (CNN)**, desarrollado siguiendo las **6 fases de CRISP-ML(Q)**
con metodología ágil **Scrum**, y con una **interfaz gráfica en Streamlit** que visualiza
cada fase.

## Fases CRISP-ML cubiertas

| # | Fase | Pestaña de la GUI |
|---|------|-------------------|
| 1 | Comprensión del Negocio | Objetivos, KPIs, Product/Sprint Backlog, DoD, riesgos |
| 2 | Comprensión de los Datos | EDA, muestras, balance de clases, estadísticas |
| 3 | Preparación de los Datos | Normalización, split, data augmentation |
| 4 | Modelado | Arquitectura CNN, hiperparámetros, entrenamiento en vivo |
| 5 | Evaluación | Curvas, matriz de confusión, métricas, errores, baseline |
| 6 | Despliegue | Inferencia sobre imagen subida / muestra del test set |

> CRISP-ML(Q) incluye además una fase transversal de **Monitorización y Mantenimiento**,
> documentada en la pestaña de Despliegue.

## Requisitos

- **Python 3.12** (TensorFlow no soporta Python 3.14 todavía).
- Dataset en `archive/fashion-mnist_train.csv` y `archive/fashion-mnist_test.csv`.

## Instalación

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

## Entrenar el modelo (opcional pero recomendado)

```powershell
.venv\Scripts\python.exe train.py
```

Genera `models/fashion_cnn.keras` (CNN) y `models/baseline_mlp.keras` (MLP de comparación).

## Ejecutar la interfaz gráfica

```powershell
.venv\Scripts\streamlit run app.py
```

Se abre en el navegador (por defecto http://localhost:8501). Recorre las 6 pestañas para
ver cada fase. Si no has entrenado antes, puedes hacerlo desde la pestaña **4️⃣ Modelado**.

## Despliegue en Streamlit Community Cloud (URL pública)

La app está preparada para desplegarse en la nube y ser accesible desde cualquier PC:

1. El código se publica en el repositorio de GitHub (carpeta `fashion-mnist-cnn/`).
2. En [share.streamlit.io](https://share.streamlit.io) → **New app**:
   - **Repository**: `andreleyends/GRUPO-IA-Los-Cheros`
   - **Branch**: `main`
   - **Main file path**: `fashion-mnist-cnn/app.py`
   - **Advanced settings → Python version**: `3.12` (¡importante para TensorFlow!)
3. **Deploy**. Se genera una URL pública tipo `https://<app>.streamlit.app`.

Notas para la nube:
- No se sube el dataset: si no existen los CSV en `archive/`, la app lo descarga
  automáticamente con `keras.datasets.fashion_mnist` (~30 MB, primera carga).
- `requirements.txt` usa `tensorflow-cpu` (evita dependencias CUDA en Linux).
- La primera carga tarda unos segundos mientras se descarga el dataset.

## Estructura del proyecto

```
app.py                      # GUI Streamlit (6 pestañas = 6 fases)
train.py                    # Entrenamiento por línea de comandos
requirements.txt
src/
  config.py                 # Clases, hiperparámetros, rutas, KPIs
  data_loader.py            # Carga de los CSV de Fashion-MNIST
  business.py               # Fase 1: negocio + Scrum
  data_understanding.py     # Fase 2: EDA
  data_preparation.py       # Fase 3: preprocesado y aumento
  modeling.py               # Fase 4: CNN + entrenamiento
  evaluation.py             # Fase 5: métricas y visualizaciones
  deployment.py             # Fase 6: inferencia
archive/                    # Dataset (CSV)
models/                     # Modelos entrenados (.keras)
```

## Resultados esperados

- Accuracy en test: **~91–92%** (objetivo ≥ 90%).
- Latencia de inferencia: **< 50 ms** por imagen.

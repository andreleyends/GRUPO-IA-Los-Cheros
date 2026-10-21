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
.venv\Scripts\python.exe train.py          # modelo base
.venv\Scripts\python.exe train_final.py    # modelo final robusto (recomendado para desplegar)
```

Genera `models/fashion_cnn.keras` (CNN) y `models/baseline_mlp.keras` (MLP de comparación).
`train_final.py` entrena un candidato con **data augmentation de producción**
(geometría + volteo horizontal + brillo suave), lo compara contra el mejor
modelo de `results/` y **despliega el mejor** en `models/fashion_cnn.keras`
(respaldando el anterior en `.bak`).

## Preprocesamiento de imágenes subidas (inferencia)

El modelo se entrenó con Fashion-MNIST: **prenda clara sobre fondo oscuro, centrada,
con margen y buen contraste**. Para que una foto real se acerque a esa distribución,
`preprocess_uploaded` (en `src/deployment.py`) aplica:

1. Escala de grises.
2. **Normalización de polaridad**: si el fondo (estimado por los píxeles del borde) es claro, o
   si la prenda es más oscura que el fondo, la imagen se invierte.
3. **Limpieza de fondo** para eliminar ruido.
4. **Recorte al contorno** de la prenda y **centrado con margen** (la prenda ocupa ~80% del
   lienzo; si toca los bordes el modelo tiende a confundir siluetas alargadas con "Bolso").
5. **Normalización de contraste** del objeto (percentiles 2–98) para que las fotos de bajo
   contraste produzcan siluetas visibles.
6. Redimensionado a 28×28 y normalización a `[0, 1]`.

La interfaz muestra además el **top-3 de predicciones** y marca **"No reconocido"** cuando la
confianza máxima es < 50%, e incluye un **control de rotación** (±45°) para prendas
fotografiadas en diagonal (p. ej. sandalias de tacón).

> Limitación: las fotos reales son un dominio distinto al del dataset. La precisión mejora
> notablemente con la prenda centrada y con buen contraste; en fotos con fondos muy complejos
> o de bajo contraste la predicción puede no ser fiable.

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

## Contexto completo del proyecto

El archivo [`CONTEXTO_PROYECTO.md`](CONTEXTO_PROYECTO.md) documenta **todo** el trabajo:
decisiones, experimentos, problemas encontrados (confusión de calzado con "Bolso" en fotos
reales, domain shift), soluciones aplicadas, resultados y siguientes pasos. Pensado como
documento de traspaso.

## Entregables del taller

En la carpeta [`ENTREGABLES/`](ENTREGABLES/) se encuentran los entregables del taller
**«Diseño y Optimización de una CNN para Clasificación de Imágenes Complejas»**:

- `Notebook_CNN_FashionMNIST.ipynb`: notebook con código limpio, documentado y celdas ejecutadas.
- `Informe_CNN_FashionMNIST.docx`: informe editable con curvas Loss/Accuracy, matriz de
  confusión y cálculo de la precisión mediante la fórmula de Accuracy.
- `figures/`: figuras generadas por el notebook.

Se regeneran con:

```powershell
.venv\Scripts\python.exe build_notebook_entregable.py
.venv\Scripts\python.exe build_informe_entregable.py
```

## Estructura del proyecto

```
app.py                      # GUI Streamlit (6 pestañas = 6 fases CRISP-ML)
train.py                    # Entrenamiento por línea de comandos (modelo base)
train_final.py              # Candidato de producción + comparación y despliegue del mejor
validate_fixes.py           # Validación de mejoras de preprocesado y modelo
experiments.py              # Experimentación comparativa (Fase C del taller)
build_notebook_entregable.py # Genera y ejecuta el notebook entregable
build_informe_entregable.py  # Genera el informe Word entregable
requirements.txt
CONTEXTO_PROYECTO.md        # Documento de traspaso: todo el trabajo, problemas y decisiones
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
results/                    # Resultados de experiments.py
ENTREGABLES/                # Notebook, informe Word, figuras y resultados del taller
```

## Resultados esperados

- Accuracy en test: **~92.4%** (objetivo ≥ 90%).
- Latencia de inferencia: **< 50 ms** por imagen.
- Calzado del test set clasificado como "Bolso" vía pipeline de subida: **0%**.

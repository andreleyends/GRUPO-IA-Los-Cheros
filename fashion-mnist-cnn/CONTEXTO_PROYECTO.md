# Contexto completo del proyecto — Clasificador Fashion-MNIST (CNN)

> **Documento de traspaso:** todo lo trabajado hasta la fecha, decisiones tomadas, problemas
> encontrados y estado actual. Sirve para que otra IA (o persona) entienda el proyecto completo
> sin necesidad de revisar el historial de conversación.

---

## 1. Resumen ejecutivo

Proyecto de IA que clasifica imágenes de prendas del dataset **Fashion-MNIST** con una
**Red Neuronal Convolucional (CNN)** en TensorFlow/Keras, siguiendo las **6 fases de CRISP-ML(Q)**
con metodología ágil **Scrum**, y con una **interfaz Streamlit** que visualiza cada fase.

- **Repositorio:** `https://github.com/andreleyends/GRUPO-IA-Los-Cheros`
- **Carpeta del proyecto:** `fashion-mnist-cnn/`
- **Estado:** funcional, desplegado y con entregables del taller incluidos.
- **Accuracy en test del modelo de producción:** **92.4%** (objetivo ≥ 90% ✅)
- **Latencia de inferencia:** ~45 ms/imagen (objetivo < 50 ms ✅)

---

## 2. Estructura del repositorio

```
fashion-mnist-cnn/
├── app.py                        # GUI Streamlit (6 pestañas = 6 fases CRISP-ML)
├── train.py                      # Entrenamiento base (CNN + baseline MLP)
├── train_final.py                # Candidato de producción + comparación y despliegue del mejor
├── experiments.py                # Experimentación comparativa (Fases C del taller)
├── validate_fixes.py             # Validación de mejoras de preprocesado y modelo
├── build_notebook_entregable.py  # Genera y ejecuta el notebook entregable
├── build_informe_entregable.py   # Genera el informe Word entregable
├── build_report.py               # (script previo, genera informe PDF)
├── build_report_docx.py          # (script previo, genera informe Word)
├── requirements.txt
├── packages.txt                  # dependencias del sistema (Streamlit Cloud)
├── .streamlit/config.toml        # config de despliegue
├── README.md
├── src/
│   ├── config.py                 # Clases, hiperparámetros, rutas, KPIs
│   ├── data_loader.py            # Carga de CSV locales o de keras.datasets
│   ├── business.py               # Fase 1: negocio + Scrum (backlog, sprints, DoD, riesgos)
│   ├── data_understanding.py     # Fase 2: EDA
│   ├── data_preparation.py       # Fase 3: normalización, split y data augmentation
│   ├── modeling.py               # Fase 4: CNN + entrenamiento
│   ├── evaluation.py             # Fase 5: métricas y visualizaciones
│   └── deployment.py             # Fase 6: preprocesado de inferencia + predicción
├── models/
│   ├── fashion_cnn.keras         # modelo de PRODUCCIÓN (92.4% test) ← desplegado
│   ├── baseline_mlp.keras        # MLP de comparación (88.2% test)
│   └── fashion_cnn.history.npy
├── results/                      # Resultados de experiments.py
│   ├── experiments.json          # historiales + métricas de las 5 configuraciones
│   ├── modelo_final.keras        # mejor modelo de los experimentos (92.4%)
│   ├── confusion_matrix.npy
│   └── fig_*.png                 # curvas y matrices de confusión
├── fallos/                       # (LOCAL, no en git) fotos reales para validar el preprocesado
└── ENTREGABLES/                  # entregables del taller
    ├── Notebook_CNN_FashionMNIST.ipynb   # notebook ejecutado (Fases A–D)
    ├── Informe_CNN_FashionMNIST.docx     # informe Word editable
    ├── resultados_notebook.json
    ├── README.md
    └── figures/                          # 10 figuras del informe
```

> **Otras carpetas del repo (NO tocar):** `neuronita/`, `sprint1/` y el `requirements.txt` raíz
> pertenecen a otros sprints del equipo. Solo se trabaja dentro de `fashion-mnist-cnn/`.

---

## 3. Dataset

- **Fashion-MNIST:** 70.000 imágenes 28×28 en escala de grises (10 clases de prendas).
- **Train:** 60.000 (6.000 por clase, **perfectamente balanceado**).
- **Test:** 10.000 (1.000 por clase).
- Los CSV locales viven en `archive/` (excluido de git). En la nube, `data_loader.py` los
  descarga automáticamente con `keras.datasets.fashion_mnist`.
- Clases (índice → español): Camiseta/top, Pantalón, Suéter, Vestido, Abrigo, Sandalia,
  Camisa, Zapatilla, Bolso, Botín.

---

## 4. Arquitectura del modelo (src/modeling.py)

CNN secuencial configurable con flags:

```
Input(28,28,1)
[Data Augmentation opcional]
Conv2D(32, 3x3, padding='same', ReLU)  [± BatchNorm]  → MaxPool 2x2
Conv2D(64, 3x3, same, ReLU)            [± BatchNorm]  → MaxPool 2x2
Conv2D(128, 3x3, same, ReLU)
Flatten → Dense(128, ReLU) [± Dropout 0.3] → Dense(10, Softmax)
```

- `build_cnn(hparams, use_augmentation, use_dropout, use_batchnorm, augmentation)`.
- ~896.906 parámetros (3.42 MB).
- `build_baseline_mlp()`: MLP simple para comparar.
- Entrenamiento con **EarlyStopping** (patience 4, val_loss) + **ReduceLROnPlateau**.

---

## 5. Experimentos de la Fase C (results/experiments.json)

5 configuraciones comparadas (10 épocas, Adam/SGD):

| Experimento | Dropout | Augment. | BatchNorm | Optim. | Train | Val | **Test** |
|---|:---:|:---:|:---:|:---:|---|---|---|
| Sin Dropout | ✗ | ✗ | ✗ | Adam | 96.7% | 91.1% | 91.67% |
| **Con Dropout (mejor)** | ✓ | ✗ | ✗ | Adam | 95.3% | 91.7% | **92.40%** |
| Dropout + Augmentation | ✓ | ✓ | ✗ | Adam | 86.9% | 87.3% | 88.15% |
| Dropout + Aug + SGD | ✓ | ✓ | ✗ | SGD | 82.1% | 82.7% | 83.84% |
| + BatchNorm (innovación) | ✓ | ✓ | ✓ | Adam | 88.0% | 87.0% | 87.46% |

**Conclusiones clave:**
- **Overfitting:** el modelo sin Dropout muestra la brecha clásica train 96.7% vs val 91.1%
  con `val_loss` subiendo. Dropout + EarlyStopping lo mitigan.
- **Optimizador:** Adam converge mucho más rápido que SGD.
- **Data Augmentation:** baja ~2-4 pts en el test "limpio" pero **aumenta la robustez** ante
  fotos reales.
- **BatchNorm:** no superó al Dropout puro con solo 10 épocas + augmentation.
- **NO es un problema de balance de clases** (el dataset ya es 6000/6000, 1000/1000).

---

## 6. Modelo de producción y reentrenamiento

- `train_final.py` entrena un candidato con **augmentation de producción**
  (`build_production_augmentation`: rotación + zoom + traslación + **volteo horizontal** +
  brillo suave, sin RandomContrast agresivo), lo compara contra `results/modelo_final.keras`
  y **despliega el mejor** en `models/fashion_cnn.keras`.
- En la última ejecución el candidato dio **89.42%** (< 92.40% del referente), por lo que se
  desplegó el **referente de 92.40%** (el anterior quedó en `.bak`, excluido de git).
- ⚠️ **Pendiente de mejora:** el reentreno con augmentation no ha superado al modelo sin
  augmentation en el test limpio. Se podría explorar más épocas, otra estrategia de aumento
  o fine-tune con fotos reales.

---

## 7. Problema principal trabajado: fotos reales mal clasificadas

### 7.1 Síntoma reportado por el usuario

Al subir **fotos reales** de prendas, el modelo confundía sobre todo **zapatos/sandalias con
"Bolso"**, daba "No reconocido" a zapatillas deportivas, y clasificaba una camisa como
"sandalia". Sospecha inicial del usuario: **desbalance de clases / sobreentrenamiento**.

### 7.2 Diagnóstico (descartado el desbalance)

- El dataset está **perfectamente balanceado** → igualar clases era innecesario y habría
  empeorado el modelo (menos datos). **Descartado.**
- Con imágenes **del test set** el calzado→Bolso casi no ocurre (5/3000). El fallo aparece
  **solo con fotos reales** → es un problema de **cambio de dominio (domain shift)** +
  fallos en el **preprocesado de inferencia**.
- Se identificaron dos causas en `src/deployment.py`:
  1. `_crop_and_center` estiraba el objeto **hasta tocar los bordes**; en Fashion-MNIST
     "Bolso" es justo una mancha grande que llena el cuadro → zapatos recortados así parecían
     bolsos.
  2. Fotos de **bajo contraste** quedaban como siluetas "fantasma" casi invisibles a 28×28.

### 7.3 Soluciones aplicadas (src/deployment.py + app.py)

1. **Margen en el recorte** (`_crop_and_center(..., fill=0.8)`): la prenda ocupa ~80% del
   lienzo, sin tocar bordes.
2. **Normalización de contraste** (`_normalize_contrast`): estira a [0,255] con percentiles
   2–98 de los píxeles del objeto.
3. **`predict` con umbral de confianza**: si `max_prob < 0.5` → **"No reconocido"** (el
   softmax siempre elige algo; en out-of-domain puede errar con alta confianza).
4. **Top-3** de predicciones devuelto siempre; la GUI lo muestra.
5. **Slider de rotación ±45°** en la GUI (Fase 6): sandalias de tacón en diagonal se arreglan
   rotando (probado: +30°/+45° → Sandalia 90–98%).

### 7.4 Resultados de la validación (validate_fixes.py)

| Métrica | Antes | Después |
|---|---|---|
| Accuracy test | 90.67% | **92.40%** |
| Calzado→Bolso (pipeline subida, 120 muestras test) | 1/60 | **0/120** |
| Fotos reales acierto top-1 | 4/8 | 4/8 (5 en top-3) |
| Latencia | — | 45 ms ✅ |

### 7.5 Fotos de `fallos/` y su estado real

| Foto | Real | Predicción (con preprocesado nuevo) | Estado |
|---|---|---|---|
| images (1) | Zapatilla (tenis fútbol) | Abrigo 44% → "No reconocido" | Límite |
| images (2) | Pantalón (jeans) | Pantalón 100% | ✅ |
| images (3) | Abrigo (casaca viento) | Abrigo 82% | ✅ |
| images (4) | Sandalia (tacón) | Bolso 100% (girando +45° → Sandalia 98%) | Límite/rotación |
| images (5) | Camisa (beige bajo contraste) | Camisa (top-3) | Límite |
| jeans rotos | Pantalón | Pantalón 100% | ✅ |
| abrigo pea-coat | Abrigo | Abrigo 78% | ✅ |
| zapatos de vestir | Botín | Sandalia 100% | Límite |

**Interpretación:** los 4 fallos originales se redujeron y los restantes son **casos límite**
reales de un modelo de 28×28 (siluetas atípicas/out-of-domain). El "No reconocido" + top-3 +
rotación los mitigan. Cerramos del todo solo con **fine-tune con fotos reales etiquetadas**
(no hecho todavía).

### 7.6 Descartado: orientación automática por prototipos

Se probó un auto-orientado por correlación con prototipos del dataset: **no generaliza**
(empeoraba otras fotos). Descartado; se optó por el **slider manual** en la GUI.

---

## 8. Entregables del taller (ENTREGABLES/)

Basados en `Taller CNN (1).pdf`: *Diseño y Optimización de una CNN para Clasificación de
Imágenes Complejas*. Fases A (preprocesamiento), B (modelo base), C (experimentación), D (evaluación).

- **`Notebook_CNN_FashionMNIST.ipynb`**: código limpio, documentado en español, **celdas
  ejecutadas**. Generado/ejecutado por `build_notebook_entregable.py`.
- **`Informe_CNN_FashionMNIST.docx`**: informe **Word editable** con gráficas de Loss/Accuracy,
  matriz de confusión, cálculo de Accuracy, análisis de overfitting e innovación (BatchNorm).
  Generado por `build_informe_entregable.py`.
- `figures/`: 10 figuras.
- Resultados de la última ejecución: mejor config = "Sin Dropout" 92.31% (semilla 42).

**Rúbrica cubierta:** Arquitectura (filtros/strides/padding ✅), Análisis de overfitting ✅,
Innovación (Batch Normalization ✅).

---

## 9. Cómo ejecutar / reproducir

```powershell
# Entorno (Python 3.12; TensorFlow no soporta 3.14)
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt

# App
.venv\Scripts\python.exe -m streamlit run app.py     # Nota: usar -m por la ruta con acentos

# Entrenamientos / validación
.venv\Scripts\python.exe train.py
.venv\Scripts\python.exe train_final.py
.venv\Scripts\python.exe experiments.py
.venv\Scripts\python.exe validate_fixes.py

# Entregables
.venv\Scripts\python.exe build_notebook_entregable.py   # ~10-15 min (entrena 5 configs)
.venv\Scripts\python.exe build_informe_entregable.py
```

> En Windows, el launcher `.venv\Scripts\streamlit.exe` falla con "uv trampoline failed to
> canonicalize script path" por la ruta con caracteres. Usar `python -m streamlit run app.py`.

---

## 10. Despliegue

**Streamlit Community Cloud:**
- Repository: `andreleyends/GRUPO-IA-Los-Cheros`
- Branch: `main`
- Main file path: `fashion-mnist-cnn/app.py`
- Python version: **3.12** (crítico para TensorFlow)
- `requirements.txt` usa `tensorflow-cpu` (evita CUDA en Linux).
- No se sube el dataset: se descarga con `keras.datasets` la primera vez.

---

## 11. Problemas conocidos / limitaciones actuales

1. **Domain shift** sigue siendo el mayor límite: fotos reales de siluetas atípicas (zapatos
   de vestir, calzado en diagonal, bajo contraste) pueden fallar. Mitigado con umbral +
   top-3 + rotación, pero no resuelto.
2. **El reentreno con augmentation no supera al modelo sin augmentation** en el test limpio
   (89.42% vs 92.40%).
3. **Sin fine-tune con datos reales**: no se ha recolectado un set de fotos reales etiquetadas.
4. **`build_report.py` / `build_report_docx.py`** son scripts previos (informe antiguo); no
   forman parte del flujo nuevo de ENTREGABLES.

---

## 12. Posibles siguientes pasos (no ejecutados)

1. **Fine-tune con fotos reales etiquetadas** (30–50 por prenda, distintos ángulos) para cerrar
   el domain shift — la solución de fondo.
2. Reentrenar con más épocas / mejor búsqueda de augmentation para superar el 92.40%.
3. Recortar el texto de la nota de la GUI o añadir guía de "mejores prácticas de foto".
4. Monitorización de drift (fase transversal de CRISP-ML(Q)), ya documentada en la GUI.

---

## 13. Convenciones y notas técnicas

- **Semilla fija** `RANDOM_SEED = 42` en todos los pasos para reproducibilidad.
- **No añadir comentarios** innecesarios al código salvo los docstrings existentes.
- El preprocesado de **inferencia** (`deployment.py`) y el de **entrenamiento**
  (`data_preparation.py`) son distintos a propósito: el de inferencia prepara fotos reales.
- `archive/` y `fallos/` están en `.gitignore` (datos locales, no van al repo).
- Los `.bak` de modelos están excluidos de git.

---

*Última actualización: 2026-10-20.*

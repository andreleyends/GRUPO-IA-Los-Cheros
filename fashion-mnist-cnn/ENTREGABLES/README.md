# Entregables del Taller — CNN Fashion-MNIST

Esta carpeta contiene los entregables solicitados en el taller
**«Diseño y Optimización de una CNN para Clasificación de Imágenes Complejas»**.

## Contenido

| Archivo | Descripción |
|---------|-------------|
| `Notebook_CNN_FashionMNIST.ipynb` | Notebook con **código limpio, documentado y celdas ejecutadas** (Fases A–D del taller). |
| `Informe_CNN_FashionMNIST.docx` | Informe en **Word editable** con gráficas de Loss/Accuracy, matriz de confusión y cálculo de Accuracy. |
| `resultados_notebook.json` | Resultados crudos (métricas, matriz de confusión, reporte) que alimentan el informe. |
| `figures/` | Figuras generadas por el notebook (curvas, matriz de confusión, comparativa, fórmula de Accuracy…). |

## Qué cubre el notebook

- **Fase A — Preparación y preprocesamiento:** carga del dataset, balance de clases,
  normalización, split estratificado y Data Augmentation.
- **Fase B — Modelo base:** CNN con Conv 3×3 (ReLU), MaxPooling y salida Softmax.
- **Fase C — Experimentación:** tabla comparativa con/sin Dropout, SGD vs Adam,
  Data Augmentation e innovación (Batch Normalization).
- **Fase D — Evaluación:** curvas Loss/Accuracy, matriz de confusión y fórmula de Accuracy.

## Qué incluye el informe (Word editable)

- Gráficas de **Loss y Accuracy** (entrenamiento vs. validación).
- **Matriz de confusión** para identificar las clases más confundidas.
- **Cálculo de la precisión final** mediante la fórmula de Accuracy.
- Análisis del **overfitting** y de las técnicas aplicadas.
- **Innovación**: Batch Normalization.

## Cómo reproducirlos

Desde la carpeta `fashion-mnist-cnn/` del repositorio:

```powershell
# Genera y ejecuta el notebook (entrena 5 configuraciones; ~10-15 min)
.venv\Scripts\python.exe build_notebook_entregable.py

# Genera el informe Word a partir de los resultados del notebook
.venv\Scripts\python.exe build_informe_entregable.py
```

> Todos los resultados usan la semilla fija `RANDOM_SEED = 42` para ser reproducibles.

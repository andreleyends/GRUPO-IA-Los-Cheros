"""Fase 1 - Comprensión del Negocio (CRISP-ML) y artefactos Scrum."""
from __future__ import annotations

PROJECT_TITLE = "Clasificador de prendas Fashion-MNIST con CNN"

PROBLEM_STATEMENT = (
    "Una plataforma de comercio electrónico recibe miles de fotografías de productos sin "
    "etiquetar y necesita clasificarlas automáticamente en 10 categorías de prendas para "
    "automatizar su catálogo, mejorar las búsquedas y reducir el trabajo manual de etiquetado. "
    "Se requiere un modelo preciso (>90% de exactitud) y rápido (<50 ms por imagen)."
)

OBJECTIVES: list[str] = [
    "Desarrollar un modelo de Deep Learning (CNN) que clasifique prendas en 10 categorías.",
    "Alcanzar al menos 90% de exactitud en el conjunto de prueba.",
    "Garantizar una inferencia rápida (<50 ms/imagen) para uso en producción.",
    "Entregar una aplicación visual (GUI) que documente el ciclo CRISP-ML completo.",
    "Trabajar con metodología ágil Scrum mediante sprints incrementales.",
]

STAKEHOLDERS: list[dict] = [
    {"rol": "Product Owner", "interes": "Catálogo clasificado automáticamente, ROI del etiquetado"},
    {"rol": "Equipo de Data Science", "interes": "Modelo preciso y reproducible"},
    {"rol": "Operaciones / Negocio", "interes": "Reducir coste y tiempo de etiquetado manual"},
    {"rol": "Equipo de TI", "interes": "Despliegue mantenible y ligero"},
]

PRODUCT_BACKLOG: list[dict] = [
    {"id": "US-01", "rol": "Product Owner", "historia": "Como PO quiero definir objetivos, KPIs y alcance del modelo", "prioridad": "Alta", "puntos": 3, "fase": "Comprensión del Negocio", "estado": "Hecho"},
    {"id": "US-02", "rol": "Data Scientist", "historia": "Como DS quiero explorar y comprender el dataset Fashion-MNIST", "prioridad": "Alta", "puntos": 5, "fase": "Comprensión de los Datos", "estado": "Hecho"},
    {"id": "US-03", "rol": "Data Scientist", "historia": "Como DS quiero limpiar, normalizar y aumentar los datos", "prioridad": "Alta", "puntos": 8, "fase": "Preparación de los Datos", "estado": "Hecho"},
    {"id": "US-04", "rol": "ML Engineer", "historia": "Como MLE quiero diseñar y entrenar una CNN de clasificación", "prioridad": "Alta", "puntos": 13, "fase": "Modelado", "estado": "Hecho"},
    {"id": "US-05", "rol": "Data Scientist", "historia": "Como DS quiero evaluar el modelo con métricas y matriz de confusión", "prioridad": "Alta", "puntos": 8, "fase": "Evaluación", "estado": "Hecho"},
    {"id": "US-06", "rol": "ML Engineer", "historia": "Como MLE quiero desplegar el modelo en una app interactiva", "prioridad": "Media", "puntos": 8, "fase": "Despliegue", "estado": "Hecho"},
    {"id": "US-07", "rol": "Product Owner", "historia": "Como PO quiero visualizar las 6 fases CRISP-ML en una GUI", "prioridad": "Media", "puntos": 5, "fase": "Transversal", "estado": "Hecho"},
    {"id": "US-08", "rol": "ML Engineer", "historia": "Como MLE quiero monitorizar el modelo y reentrenarlo periódicamente", "prioridad": "Baja", "puntos": 5, "fase": "Monitorización y Mantenimiento", "estado": "Backlog"},
]

SPRINTS: list[dict] = [
    {"sprint": "Sprint 0", "fase_crisp": "Comprensión del Negocio", "objetivo": "Definir problema, KPIs, backlog y arquitectura de la solución", "entregable": "Documento de objetivos + Product Backlog", "duracion": "1 semana", "story_points": 3},
    {"sprint": "Sprint 1", "fase_crisp": "Comprensión de los Datos", "objetivo": "Analizar el dataset: distribución, calidad y estadísticas", "entregable": "Informe EDA con visualizaciones", "duracion": "1 semana", "story_points": 5},
    {"sprint": "Sprint 2", "fase_crisp": "Preparación de los Datos", "objetivo": "Normalizar, dividir y aumentar los datos", "entregable": "Pipelines de datos listos", "duracion": "1 semana", "story_points": 8},
    {"sprint": "Sprint 3", "fase_crisp": "Modelado", "objetivo": "Diseñar, entrenar y ajustar la CNN", "entregable": "Modelo CNN entrenado (.keras)", "duracion": "2 semanas", "story_points": 13},
    {"sprint": "Sprint 4", "fase_crisp": "Evaluación", "objetivo": "Evaluar con métricas, matriz de confusión y análisis de errores", "entregable": "Informe de evaluación del modelo", "duracion": "1 semana", "story_points": 8},
    {"sprint": "Sprint 5", "fase_crisp": "Despliegue", "objetivo": "Desplegar la app interactiva y documento de MVP", "entregable": "Aplicación Streamlit funcionando", "duracion": "1 semana", "story_points": 8},
]

DEFINITION_OF_DONE: list[str] = [
    "El código está versionado y reproducible (requirements + semilla fija).",
    "Cada fase queda documentada y visible en la interfaz gráfica.",
    "El modelo supera el umbral de KPI definido en la Fase 1 (accuracy ≥ 90%).",
    "Existen métricas de evaluación y matriz de confusión del modelo final.",
    "La aplicación de despliegue permite subir una imagen y obtener una predicción.",
]

RISKS: list[dict] = [
    {"riesgo": "Clases visualmente similares (Camisa/Suéter/Abrigo)", "mitigacion": "Data augmentation + análisis de errores"},
    {"riesgo": "Sobreajuste del modelo", "mitigacion": "Dropout, validación y early stopping"},
    {"riesgo": "Datos desbalanceados", "mitigacion": "Verificación de balance (dataset balanceado 6000/clase)"},
    {"riesgo": "Entorno sin soporte de TensorFlow en Python 3.14", "mitigacion": "Usar Python 3.12 en venv aislado"},
]

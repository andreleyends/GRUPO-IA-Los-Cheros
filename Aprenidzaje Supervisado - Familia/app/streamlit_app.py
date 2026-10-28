"""
Simulación interactiva del Equipo 6 - Familia 3 (Probabilísticos y Estructurados).

Ejecutar con:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permite importar el paquete src cuando Streamlit se ejecuta desde la raíz
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src import config
from src.data_loader import (
    cargar_20newsgroups,
    cargar_california_housing,
    cargar_clasificacion_lda_qda,
    crear_dataset_sintetico_covarianzas_distintas,
)
from src.models_nb import crear_pipeline_nb, obtener_palabras_informativas
from src.models_mlp import crear_pipeline_mlp

st.set_page_config(
    page_title="Equipo 6 - Familia 3",
    page_icon="🤖",
    layout="wide",
)

st.title("Despliegue y Análisis Comparativo de Familias de Aprendizaje Supervisado")
st.caption("Equipo 6 · Familia 3: Probabilísticos y Estructurados")


# ---------------------------------------------------------------------------
# Datos cacheados
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Cargando corpus de texto...")
def datos_texto():
    """Carga textos y etiquetas de 20 Newsgroups (con fallback sintético)."""
    textos, y = cargar_20newsgroups()
    nombres = [f"clase_{i}" for i in range(len(np.unique(y)))]
    try:
        from sklearn.datasets import fetch_20newsgroups
        data = fetch_20newsgroups(
            subset="all",
            categories=[
                "comp.graphics",
                "rec.sport.baseball",
                "sci.med",
                "talk.politics.mideast",
            ],
            remove=("headers", "footers", "quotes"),
            data_home=str(config.DATA_DIR),
        )
        nombres = list(data.target_names)
    except Exception:
        pass
    return textos, np.asarray(y), nombres


@st.cache_data(show_spinner="Cargando datos de vivienda...")
def datos_regresion():
    """Carga California Housing (con fallback)."""
    X, y = cargar_california_housing()
    return X, np.asarray(y)


@st.cache_data(show_spinner="Preparando datasets 2D...")
def datos_2d(nombre: str):
    """Devuelve X, y, etiquetas de clase y nombres de columnas para 2D."""
    if nombre == "Breast Cancer (PCA 2)":
        X, y = cargar_clasificacion_lda_qda("breast_cancer")
        Xp = PCA(n_components=2, random_state=config.RANDOM_STATE).fit_transform(X)
        return Xp, np.asarray(y), ["Componente 1", "Componente 2"], "PCA-2 Breast Cancer"
    if nombre == "Wine (PCA 2)":
        X, y = cargar_clasificacion_lda_qda("wine")
        Xp = PCA(n_components=2, random_state=config.RANDOM_STATE).fit_transform(X)
        return Xp, np.asarray(y), ["Componente 1", "Componente 2"], "PCA-2 Wine"
    X, y = crear_dataset_sintetico_covarianzas_distintas()
    return X, np.asarray(y), ["Característica 1", "Característica 2"], "Sintético (covarianzas distintas)"


@st.cache_resource(show_spinner="Entrenando Naive Bayes...")
def entrenar_nb(clasificador: str, alpha: float):
    """Entrena un pipeline NB con el corpus y devuelve modelo y métricas."""
    textos, y, clases = datos_texto()
    X_tr, X_te, y_tr, y_te = train_test_split(
        textos, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y
    )
    pipe = crear_pipeline_nb(
        tipo_vectorizador="tfidf",
        clasificador=clasificador,
        alpha=alpha,
    )
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    return pipe, float(accuracy_score(y_te, pred)), float(f1_score(y_te, pred, average="macro"))


@st.cache_resource(show_spinner="Entrenando MLP...")
def entrenar_mlp(capas: tuple, activacion: str, alpha: float, lr: float):
    """Entrena MLPRegressor sobre una submuestra para respuesta rápida."""
    X, y = datos_regresion()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE
    )
    # Submuestra para que la simulación sea fluida
    n = min(6000, len(X_tr))
    indices = np.random.RandomState(config.RANDOM_STATE).choice(len(X_tr), size=n, replace=False)
    X_tr_s = X_tr.iloc[indices]
    y_tr_s = y_tr[indices]
    pipe = crear_pipeline_mlp(
        hidden_layer_sizes=capas,
        activation=activacion,
        alpha=alpha,
        learning_rate_init=lr,
        solver="adam",
        early_stopping=True,
        max_iter=400,
        random_state=config.RANDOM_STATE,
    )
    pipe.fit(X_tr_s, y_tr_s)
    pred = pipe.predict(X_te)
    return pipe, X_te, y_te, pred


# ---------------------------------------------------------------------------
# Pestañas
# ---------------------------------------------------------------------------

tab_nb, tab_mlp, tab_disc, tab_comp = st.tabs(
    ["Naive Bayes", "MLP Regressor", "LDA / QDA", "Comparación global"]
)

# ------------------------------ Naive Bayes -------------------------------
with tab_nb:
    st.header("Naive Bayes para clasificación de texto")
    st.write(
        "Escribe un documento y el modelo estimará la categoría. "
        "El vectorizador usa TF-IDF; no se aplica escalado porque NB Multinomial "
        "requiere conteos o frecuencias no negativas."
    )
    try:
        textos, y, nombres_clases = datos_texto()
        col1, col2 = st.columns(2)
        with col1:
            clasificador_nb = st.selectbox(
                "Variante", ["multinomial", "complement"],
                format_func=lambda v: "MultinomialNB" if v == "multinomial" else "ComplementNB",
            )
        with col2:
            alpha_nb = st.slider("Suavizado alpha", 0.01, 5.0, 0.1, 0.01)

        modelo_nb, acc_nb, f1_nb = entrenar_nb(clasificador_nb, alpha_nb)

        col_a, col_b = st.columns(2)
        col_a.metric("Exactitud (test)", f"{acc_nb:.4f}")
        col_b.metric("F1 macro (test)", f"{f1_nb:.4f}")

        documento = st.text_area(
            "Documento a clasificar",
            value=(
                "The medical team treated the patient in the hospital and "
                "reviewed the diagnosis and treatment options."
            ),
            height=140,
        )

        if st.button("Clasificar documento", type="primary"):
            if not documento.strip():
                st.warning("Escribe algún texto antes de clasificar.")
            else:
                proba = modelo_nb.predict_proba([documento])[0]
                clases_modelo = modelo_nb.named_steps["clf"].classes_
                pred = modelo_nb.predict([documento])[0]
                nombre_pred = nombres_clases[int(pred)] if int(pred) < len(nombres_clases) else str(pred)

                st.success(f"Clase predicha: **{nombre_pred}**")

                df_proba = pd.DataFrame({
                    "clase": [nombres_clases[int(c)] if int(c) < len(nombres_clases) else str(c)
                              for c in clases_modelo],
                    "probabilidad": proba,
                }).sort_values("probabilidad", ascending=True)
                fig = px.bar(
                    df_proba, x="probabilidad", y="clase", orientation="h",
                    title="Probabilidades por clase", range_x=[0, 1],
                )
                st.plotly_chart(fig, width="stretch")

                st.subheader("Palabras más influyentes por clase")
                clase_pred_idx = int(np.where(clases_modelo == pred)[0][0])
                palabras = obtener_palabras_informativas(modelo_nb, clase_pred_idx, top_n=12)
                if palabras:
                    df_pal = pd.DataFrame(palabras, columns=["palabra", "peso"])
                    fig2 = px.bar(
                        df_pal.iloc[::-1], x="peso", y="palabra", orientation="h",
                        title=f"Top palabras para {nombre_pred}",
                    )
                    st.plotly_chart(fig2, width="stretch")
                else:
                    st.info("El modelo no expone palabras informativas.")
    except Exception as exc:
        st.error(f"No se pudo ejecutar Naive Bayes: {exc}")


# ------------------------------ MLP Regressor -----------------------------
with tab_mlp:
    st.header("MLPRegressor para regresión")
    st.write(
        "Configura la arquitectura y entrena sobre una submuestra de California "
        "Housing para mantener la interacción fluida. El escalado se ajusta "
        "solo con el conjunto de entrenamiento dentro del Pipeline."
    )
    try:
        col1, col2, col3 = st.columns(3)
        with col1:
            n_capas = st.slider("Capas ocultas", 1, 3, 2)
            neuronas = st.slider("Neuronas por capa", 16, 256, 64, 16)
        with col2:
            activacion = st.selectbox("Activación", ["relu", "tanh"])
            alpha_mlp = st.select_slider(
                "Regularización L2 (alpha)",
                options=[0.00001, 0.0001, 0.001, 0.01, 0.1],
                value=0.001,
            )
        with col3:
            lr_mlp = st.select_slider(
                "Tasa de aprendizaje inicial",
                options=[0.0001, 0.001, 0.01, 0.05],
                value=0.001,
            )
            st.write("")
            entrenar = st.button("Entrenar MLP", type="primary")

        capas = tuple([neuronas] * n_capas)

        if entrenar:
            with st.spinner("Entrenando red neuronal..."):
                modelo_mlp, X_te, y_te, pred = entrenar_mlp(
                    capas, activacion, alpha_mlp, lr_mlp
                )
            rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
            r2 = float(r2_score(y_te, pred))
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("RMSE", f"{rmse:.4f}")
            col_b.metric("R²", f"{r2:.4f}")
            col_c.metric("Iteraciones", modelo_mlp.named_steps["mlp"].n_iter_)

            mlp = modelo_mlp.named_steps["mlp"]
            fig_loss, ax_loss = plt.subplots(figsize=(7, 3))
            ax_loss.plot(mlp.loss_curve_)
            ax_loss.set_xlabel("Iteración")
            ax_loss.set_ylabel("Pérdida")
            ax_loss.set_title("Curva de pérdida")
            st.pyplot(fig_loss)
            plt.close(fig_loss)

            fig_real, ax_real = plt.subplots(figsize=(6, 4))
            ax_real.scatter(y_te[:2000], pred[:2000], alpha=0.4)
            lim = [min(y_te[:2000].min(), pred[:2000].min()),
                   max(y_te[:2000].max(), pred[:2000].max())]
            ax_real.plot(lim, lim, "r--", label="Ideal")
            ax_real.set_xlabel("Real")
            ax_real.set_ylabel("Predicho")
            ax_real.set_title("Real vs Predicho (muestra)")
            ax_real.legend()
            st.pyplot(fig_real)
            plt.close(fig_real)
        else:
            st.info("Configura los hiperparámetros y pulsa **Entrenar MLP**.")
    except Exception as exc:
        st.error(f"No se pudo ejecutar el MLP: {exc}")


# ------------------------------ LDA / QDA ---------------------------------
with tab_disc:
    st.header("LDA y QDA con frontera de decisión en vivo")
    st.write(
        "Selecciona un dataset 2D, el modelo y su regularización. "
        "Puedes añadir un punto nuevo para ver cómo lo clasifica el modelo."
    )
    try:
        nombre_dataset = st.selectbox(
            "Dataset 2D",
            ["Breast Cancer (PCA 2)", "Wine (PCA 2)", "Sintético (covarianzas distintas)"],
        )
        X2, y2, etiquetas_cols, titulo_dataset = datos_2d(nombre_dataset)

        col1, col2 = st.columns(2)
        with col1:
            modelo_disc = st.selectbox("Modelo", ["LDA", "QDA"])
        with col2:
            if modelo_disc == "LDA":
                shrinkage = st.slider("Shrinkage (LDA)", 0.0, 0.9, 0.0, 0.05)
            else:
                reg_param = st.slider("reg_param (QDA)", 0.0, 0.9, 0.05, 0.01)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X2, y2, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y2
        )

        if modelo_disc == "LDA":
            if shrinkage > 0:
                modelo = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)
            else:
                modelo = LinearDiscriminantAnalysis(solver="svd")
        else:
            modelo = QuadraticDiscriminantAnalysis(reg_param=reg_param)

        modelo.fit(X_tr, y_tr)
        pred = modelo.predict(X_te)
        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, average="macro")
        col_a, col_b = st.columns(2)
        col_a.metric("Exactitud (test)", f"{acc:.4f}")
        col_b.metric("F1 macro (test)", f"{f1:.4f}")

        # Frontera de decisión
        x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
        y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 250),
            np.linspace(y_min, y_max, 250),
        )
        Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        scatter = ax.scatter(X2[:, 0], X2[:, 1], c=y2, cmap=plt.cm.coolwarm, edgecolor="k", s=25)
        ax.set_xlabel(etiquetas_cols[0])
        ax.set_ylabel(etiquetas_cols[1])
        ax.set_title(f"{titulo_dataset} - {modelo_disc}")
        legend1 = ax.legend(*scatter.legend_elements(), title="Clase")
        ax.add_artist(legend1)
        st.pyplot(fig)
        plt.close(fig)

        # Punto nuevo
        st.subheader("Clasificar un punto nuevo")
        col_x, col_y = st.columns(2)
        x_nuevo = col_x.number_input(
            etiquetas_cols[0], value=float(np.mean(X2[:, 0])), step=0.1
        )
        y_nuevo = col_y.number_input(
            etiquetas_cols[1], value=float(np.mean(X2[:, 1])), step=0.1
        )
        if st.button("Clasificar punto"):
            clase = modelo.predict([[x_nuevo, y_nuevo]])[0]
            proba = modelo.predict_proba([[x_nuevo, y_nuevo]])[0]
            st.success(f"El punto ({x_nuevo:.2f}, {y_nuevo:.2f}) pertenece a la clase **{clase}**.")
            df_p = pd.DataFrame({
                "clase": modelo.classes_.astype(str),
                "probabilidad": proba,
            })
            st.dataframe(df_p, width="stretch")
    except Exception as exc:
        st.error(f"No se pudo ejecutar LDA/QDA: {exc}")


# ------------------------------ Comparación -------------------------------
with tab_comp:
    st.header("Comparación global de las tres familias")
    st.write(
        "Naive Bayes y LDA/QDA son clasificadores; MLPRegressor es un regresor. "
        "Por eso la comparación de métricas es cualitativa y se complementa con "
        "los tiempos de entrenamiento e inferencia."
    )
    ruta_comp = config.TABLES_DIR / "comparacion_modelos.csv"
    if ruta_comp.exists():
        df_comp = pd.read_csv(ruta_comp)
        st.dataframe(df_comp, width="stretch")

        fig = px.bar(
            df_comp,
            x="modelo",
            y="tiempo_entrenamiento_s",
            color="familia",
            title="Tiempo de entrenamiento por modelo (s)",
        )
        st.plotly_chart(fig, width="stretch")

        st.info(
            "F1 macro y exactitud no son comparables con RMSE/R². "
            "La tabla permite comparar costo computacional, no desempeño entre tareas distintas."
        )
    else:
        st.warning(
            "Aún no existen resultados. Ejecuta `python -m src.run_all` para generarlos."
        )
    st.subheader("Resumen de las familias")
    st.markdown(
        """
| Familia | Supuesto clave | Frontera / salida | Fortaleza | Limitación |
|---|---|---|---|---|
| Naive Bayes | Independencia condicional de palabras | Probabilidades por clase | Muy rápido, buen baseline de texto | Supuesto de independencia irreal |
| MLP Regressor | Composición de transformaciones no lineales | Regresión continua | Aproxima funciones complejas | Costoso, sensible a escalado e hiperparámetros |
| LDA / QDA | Gaussianidad por clase; LDA comparte covarianza | Lineal (LDA) / cuadrática (QDA) | Interpretable y eficiente | QDA requiere más datos y regularización |
        """
    )


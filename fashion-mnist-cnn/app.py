"""Interfaz gráfica (Streamlit) que recorre las 6 fases de CRISP-ML + Scrum.

Ejecutar:
    .venv\\Scripts\\streamlit run app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st

from src import business
from src.config import (
    BASELINE_PATH,
    BUSINESS_KPIS,
    CLASS_NAMES_ES,
    CLASS_NAMES_EN,
    DEFAULT_HPARAMS,
    MODEL_PATH,
)
from src.data_loader import load_fashion_mnist
from src.data_preparation import (
    get_prepared_data,
    plot_augmentation_preview,
    plot_pipeline_summary,
)
from src.data_understanding import (
    class_balance_table,
    dataset_summary,
    plot_class_distribution,
    plot_intensity_histogram,
    plot_mean_image_per_class,
    plot_one_per_class,
    plot_samples_grid,
)
from src.deployment import (
    DEPLOYMENT_NOTES,
    predict,
    preprocess_uploaded,
)
from src.evaluation import (
    compute_confusion,
    evaluate_test,
    per_class_accuracy,
    plot_confusion_matrix,
    plot_predictions_grid,
    plot_training_curves,
    top_confusions,
)
from src.modeling import (
    build_baseline_mlp,
    build_cnn,
    load_model,
    model_exists,
    save_model,
    summary_text,
    train_model,
)

st.set_page_config(
    page_title="Fashion-MNIST · CRISP-ML + Scrum",
    page_icon="👕",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Caché y utilidades
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_dataset():
    return load_fashion_mnist()


@st.cache_data(show_spinner=False)
def cached_prepared(val_size: float):
    return get_prepared_data(val_size=val_size)


def model_mtime() -> float | None:
    return MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None


@st.cache_resource(show_spinner=False)
def cached_model(_mtime: float | None):
    return load_model(MODEL_PATH)


@st.cache_data(show_spinner=False)
def cached_eval(_mtime: float | None, val_size: float):
    model = load_model(MODEL_PATH)
    prepared = cached_prepared(val_size)
    y_pred, cm, report = compute_confusion(model, prepared)
    loss, acc = evaluate_test(model, prepared)
    return y_pred, cm, report, loss, acc


def show_fig(fig) -> None:
    import matplotlib.pyplot as plt

    st.pyplot(fig, width="stretch")
    plt.close(fig)


def load_history():
    if "history" in st.session_state:
        return st.session_state["history"]
    history_path = MODEL_PATH.with_suffix(".history.npy")
    if history_path.exists():
        return np.load(history_path, allow_pickle=True).item()
    return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("👕 Fashion-MNIST")
    st.caption("CRISP-ML + Scrum · CNN")
    st.markdown("---")
    st.subheader("Estado del proyecto")
    if model_exists():
        st.success("Modelo CNN entrenado ✓")
    else:
        st.warning("Modelo sin entrenar. Ve a la pestaña **Modelado**.")
    st.metric("Modelos y datasets", "10 clases")
    st.markdown("---")
    st.subheader("Fases CRISP-ML")
    for i, fase in enumerate(
        [
            "1. Comprensión del Negocio",
            "2. Comprensión de los Datos",
            "3. Preparación de los Datos",
            "4. Modelado",
            "5. Evaluación",
            "6. Despliegue",
        ]
    ):
        st.markdown(f"- {fase}")
    st.markdown("---")
    if st.button("🧹 Limpiar caché"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

st.title("Clasificador de prendas Fashion-MNIST")
st.caption(
    "Ciclo de vida CRISP-ML(Q) completo con metodología Scrum · Red Neuronal Convolucional (CNN)"
)

tabs = st.tabs(
    [
        "1️⃣ Negocio",
        "2️⃣ Datos",
        "3️⃣ Preparación",
        "4️⃣ Modelado",
        "5️⃣ Evaluación",
        "6️⃣ Despliegue",
    ]
)

x_train_raw, y_train, x_test_raw, y_test = cached_dataset()


# ===========================================================================
# FASE 1 - COMPRENSIÓN DEL NEGOCIO
# ===========================================================================
with tabs[0]:
    st.header("Fase 1 · Comprensión del Negocio")
    st.info(business.PROBLEM_STATEMENT)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Objetivos del proyecto")
        for obj in business.OBJECTIVES:
            st.markdown(f"- {obj}")
    with col2:
        st.subheader("Stakeholders")
        st.dataframe(pd.DataFrame(business.STAKEHOLDERS), width="stretch", hide_index=True)

    st.subheader("KPIs / Métricas de éxito")
    st.dataframe(pd.DataFrame(BUSINESS_KPIS), width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("🔄 Artefactos Scrum")

    st.markdown("**Product Backlog**")
    backlog_df = pd.DataFrame(business.PRODUCT_BACKLOG)
    backlog_df.columns = ["ID", "Rol", "Historia de usuario", "Prioridad", "Puntos", "Fase CRISP-ML", "Estado"]
    st.dataframe(backlog_df, width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sprint Backlog**")
        sprint_df = pd.DataFrame(business.SPRINTS)
        sprint_df.columns = ["Sprint", "Fase CRISP-ML", "Objetivo", "Entregable", "Duración", "Puntos"]
        st.dataframe(sprint_df, width="stretch", hide_index=True)
    with c2:
        st.markdown("**Velocidad estimada por sprint**")
        vel = pd.DataFrame(business.SPRINTS).set_index("sprint")["story_points"]
        st.bar_chart(vel)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Definition of Done**")
        for item in business.DEFINITION_OF_DONE:
            st.markdown(f"- ✅ {item}")
    with c4:
        st.markdown("**Riesgos y mitigación**")
        st.dataframe(pd.DataFrame(business.RISKS), width="stretch", hide_index=True)


# ===========================================================================
# FASE 2 - COMPRENSIÓN DE LOS DATOS
# ===========================================================================
with tabs[1]:
    st.header("Fase 2 · Comprensión de los Datos (EDA)")
    summary = dataset_summary(x_train_raw, y_train, x_test_raw, y_test)

    m = st.columns(5)
    m[0].metric("Imágenes de entrenamiento", f"{summary['train_images']:,}")
    m[1].metric("Imágenes de prueba", f"{summary['test_images']:,}")
    m[2].metric("Formato de imagen", summary["image_shape"])
    m[3].metric("Clases", summary["num_classes"])
    m[4].metric("Rango de píxeles", summary["pixel_range"])

    st.markdown("#### Muestras del dataset")
    if st.button("🎲 Generar nuevas muestras aleatorias"):
        st.session_state["sample_seed"] = np.random.randint(0, 10_000)
    np.random.seed(st.session_state.get("sample_seed", 42))
    show_fig(plot_samples_grid(x_train_raw, y_train))

    st.markdown("#### Un ejemplo por clase")
    show_fig(plot_one_per_class(x_train_raw, y_train))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Distribución de clases")
        show_fig(plot_class_distribution(y_train, y_test))
    with c2:
        st.markdown("#### Intensidad de píxeles")
        show_fig(plot_intensity_histogram(x_train_raw))

    st.markdown("#### Imagen promedio por clase")
    show_fig(plot_mean_image_per_class(x_train_raw.astype("float32"), y_train))

    st.markdown("#### Balance y calidad de los datos")
    balance_df = pd.DataFrame(class_balance_table(y_train, y_test))
    balance_df["Etiqueta (EN)"] = CLASS_NAMES_EN
    st.dataframe(balance_df, width="stretch", hide_index=True)
    st.success(
        f"Dataset balanceado: ~{int(summary['train_balance'].mean())} imágenes por clase. "
        f"Sin valores nulos ({summary['missing_values']}). "
        f"Media de píxel {summary['pixel_mean']:.1f} ± {summary['pixel_std']:.1f} (escala 0-255)."
    )


# ===========================================================================
# FASE 3 - PREPARACIÓN DE LOS DATOS
# ===========================================================================
with tabs[2]:
    st.header("Fase 3 · Preparación de los Datos")
    st.markdown(
        """
        Transformaciones aplicadas antes del modelado:
        1. **Normalización**: píxeles de `[0, 255]` → `[0, 1]` (mejora la convergencia).
        2. **Reshape**: de `(28, 28)` a `(28, 28, 1)` para añadir el canal.
        3. **Split estratificado**: entrenamiento / validación / prueba.
        4. **Data Augmentation**: rotación, zoom y traslación aleatorias (solo en entrenamiento).
        5. **Etiquetas sparse**: enteros 0-9 con `sparse_categorical_crossentropy`.
        """
    )

    val_size = st.slider("Proporción de validación", 0.05, 0.30, 0.10, 0.05)
    prepared = cached_prepared(val_size)

    s = st.columns(4)
    s[0].metric("Train", f"{len(prepared['x_train']):,}")
    s[1].metric("Validación", f"{len(prepared['x_val']):,}")
    s[2].metric("Prueba", f"{len(prepared['x_test']):,}")
    s[3].metric("Shape de entrada", str(prepared["x_train"].shape[1:]))

    st.markdown("#### Pipeline de preparación")
    show_fig(plot_pipeline_summary(x_train_raw, prepared))

    st.markdown("#### Data Augmentation (ejemplos)")
    if st.button("🔄 Regenerar ejemplos aumentados"):
        st.session_state["aug_seed"] = np.random.randint(0, 10_000)
    np.random.seed(st.session_state.get("aug_seed", 0))
    show_fig(plot_augmentation_preview(prepared["x_train"], prepared["y_train"]))


# ===========================================================================
# FASE 4 - MODELADO
# ===========================================================================
with tabs[3]:
    st.header("Fase 4 · Modelado (CNN)")

    st.markdown(
        """
        **¿Por qué una CNN?** Las CNN explotan la estructura espacial de las imágenes
        mediante filtros convolucionales compartidos, superando a un MLP en tareas de
        visión y con menos parámetros efectivos.
        """
    )

    with st.expander("⚙️ Hiperparámetros", expanded=True):
        c = st.columns(3)
        hparams = dict(DEFAULT_HPARAMS)
        hparams["conv1_filters"] = c[0].selectbox("Filtros Conv1", [16, 32, 64], index=1)
        hparams["conv2_filters"] = c[1].selectbox("Filtros Conv2", [32, 64, 128], index=1)
        hparams["kernel_size"] = c[2].selectbox("Tamaño de kernel", [3, 5], index=0)
        c = st.columns(3)
        hparams["dense_units"] = c[0].selectbox("Unidades densas", [64, 128, 256], index=1)
        hparams["dropout"] = c[1].slider("Dropout", 0.0, 0.6, 0.3, 0.05)
        hparams["learning_rate"] = c[2].select_slider(
            "Learning rate", [1e-4, 5e-4, 1e-3, 2e-3], value=1e-3
        )
        c = st.columns(3)
        hparams["epochs"] = c[0].slider("Épocas", 3, 30, 20, 1)
        hparams["batch_size"] = c[1].selectbox("Batch size", [32, 64, 128, 256], index=2)
        use_aug = c[2].checkbox("Data Augmentation", value=True)

    model_preview = build_cnn(hparams, use_augmentation=use_aug)
    with st.expander("🧱 Arquitectura del modelo (summary)"):
        st.code(summary_text(model_preview), language="text")

    val_size = st.session_state.get("val_size_model", 0.10)
    prepared_men = cached_prepared(0.10)

    if st.button("🚀 Entrenar modelo CNN", type="primary"):
        model = build_cnn(hparams, use_augmentation=use_aug)
        progress = st.progress(0.0, text="Entrenando...")
        chart_placeholder = st.empty()
        log_placeholder = st.empty()
        live = {"epoch": [], "loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

        def on_epoch(epoch, logs):
            live["epoch"].append(epoch)
            live["loss"].append(logs["loss"])
            live["val_loss"].append(logs["val_loss"])
            live["accuracy"].append(logs["accuracy"])
            live["val_accuracy"].append(logs["val_accuracy"])
            progress.progress(min(epoch / hparams["epochs"], 1.0), text=f"Época {epoch}/{hparams['epochs']}")
            df = pd.DataFrame(live).set_index("epoch")
            chart_placeholder.line_chart(df[["accuracy", "val_accuracy"]])
            log_placeholder.dataframe(
                df.round(4).tail(5), width="stretch"
            )

        with st.spinner("Entrenando la CNN..."):
            history = train_model(model, prepared_men, hparams, progress_callback=on_epoch)

        save_model(model, MODEL_PATH)
        st.session_state["history"] = history.history
        st.cache_data.clear()
        st.cache_resource.clear()
        loss, acc = evaluate_test(model, prepared_men)
        progress.progress(1.0, text="Entrenamiento finalizado ✓")
        st.success(f"Modelo guardado. **Accuracy en test: {acc:.2%}** (loss {loss:.3f}).")
        st.balloons()

    st.markdown("---")
    st.subheader("Modelo baseline (MLP) para comparación")
    st.caption("Un perceptrón multicapa simple sirve de referencia para justificar la CNN.")
    if st.button("🧪 Entrenar baseline MLP"):
        baseline = build_baseline_mlp()
        base_hparams = {**DEFAULT_HPARAMS, "epochs": 8}
        with st.spinner("Entrenando MLP..."):
            train_model(baseline, prepared_men, base_hparams)
        save_model(baseline, BASELINE_PATH)
        _, b_acc = evaluate_test(baseline, prepared_men)
        st.success(f"Baseline MLP guardado. Accuracy en test: {b_acc:.2%}")


# ===========================================================================
# FASE 5 - EVALUACIÓN
# ===========================================================================
with tabs[4]:
    st.header("Fase 5 · Evaluación")

    if not model_exists():
        st.warning("Aún no hay un modelo entrenado. Ve a la pestaña **4️⃣ Modelado** y entrena la CNN.")
    else:
        mtime = model_mtime()
        val_size = 0.10
        prepared = cached_prepared(val_size)
        with st.spinner("Calculando métricas..."):
            y_pred, cm, report, loss, acc = cached_eval(mtime, val_size)

        m = st.columns(4)
        m[0].metric("Accuracy (test)", f"{acc:.2%}")
        m[1].metric("Loss (test)", f"{loss:.3f}")
        m[2].metric("F1-macro", "ver reporte")
        m[3].metric("Objetivo", "≥ 90%", delta=f"{(acc - 0.90) * 100:+.1f} pts")

        history = load_history()
        if history is not None:
            st.markdown("#### Curvas de aprendizaje")
            show_fig(plot_training_curves(type("H", (), {"history": history})))

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### Matriz de confusión")
            show_fig(plot_confusion_matrix(cm))
        with c2:
            st.markdown("#### Métricas por clase")
            st.dataframe(pd.DataFrame(per_class_accuracy(cm)), width="stretch", hide_index=True)
            st.markdown("#### Clases más confundidas")
            st.dataframe(pd.DataFrame(top_confusions(cm)), width="stretch", hide_index=True)

        with st.expander("📄 Reporte de clasificación (precision / recall / f1)"):
            st.code(report, language="text")

        st.markdown("#### Predicciones sobre el conjunto de prueba")
        show_fig(plot_predictions_grid(load_model(MODEL_PATH), prepared["x_test"], prepared["y_test"], n=12))

        st.markdown("#### Ejemplos de errores del modelo")
        show_fig(
            plot_predictions_grid(
                load_model(MODEL_PATH), prepared["x_test"], prepared["y_test"], n=12, only_errors=True
            )
        )

        if BASELINE_PATH.exists():
            st.markdown("#### Comparación con el baseline (MLP)")
            baseline = load_model(BASELINE_PATH)
            _, b_acc = evaluate_test(baseline, prepared)
            comp = pd.DataFrame(
                [
                    {"Modelo": "CNN (propuesto)", "Accuracy test": f"{acc:.2%}"},
                    {"Modelo": "MLP (baseline)", "Accuracy test": f"{b_acc:.2%}"},
                    {"Modelo": "Mejora", "Accuracy test": f"{(acc - b_acc) * 100:+.2f} pts"},
                ]
            )
            st.dataframe(comp, width="stretch", hide_index=True)
        else:
            st.info("Entrena el baseline MLP en la pestaña **Modelado** para ver la comparación.")


# ===========================================================================
# FASE 6 - DESPLIEGUE
# ===========================================================================
with tabs[5]:
    st.header("Fase 6 · Despliegue")

    if not model_exists():
        st.warning("No hay modelo para desplegar. Entrena la CNN en la pestaña **4️⃣ Modelado**.")
    else:
        model = load_model(MODEL_PATH)
        st.success("Modelo CNN cargado y listo para inferencia.")

        modo = st.radio(
            "Fuente de la imagen",
            ["🖼️ Subir una imagen", "🎲 Muestra aleatoria del test set"],
            horizontal=True,
        )

        image = None
        real_label = None
        if modo == "🖼️ Subir una imagen":
            uploaded = st.file_uploader("Sube una imagen de una prenda (PNG/JPG)", type=["png", "jpg", "jpeg"])
            if uploaded is not None:
                from PIL import Image

                pil = Image.open(uploaded)
                c1, c2 = st.columns(2)
                c1.markdown("**Imagen original**")
                c1.image(pil, caption="Entrada", width=220)
                invert = st.selectbox("Inversión de color", ["auto", "sí", "no"], index=0)
                image = preprocess_uploaded(pil, invert=invert)
                c2.markdown("**Transformada (28×28)**")
                c2.image(image[..., 0], caption="Entrada al modelo", width=140, clamp=True)
        else:
            if st.button("🎲 Nueva muestra aleatoria"):
                st.session_state["deploy_idx"] = int(np.random.randint(0, len(x_test_raw)))
            idx = st.session_state.get("deploy_idx", 0)
            image_raw = x_test_raw[idx]
            real_label = int(y_test[idx])
            image = (image_raw.astype("float32") / 255.0)[..., np.newaxis]
            st.image(
                image[..., 0],
                caption=f"Real: {real_label} - {CLASS_NAMES_ES[real_label]}",
                width=140,
                clamp=True,
            )

        if image is not None:
            result = predict(model, image)
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("### Predicción")
                st.metric("Clase predicha", f"{result['index']} - {result['label']}")
                st.metric("Confianza", f"{result['confidence']:.1%}")
                st.metric("Latencia", f"{result['latency_ms']:.1f} ms")
                if real_label is not None:
                    acierto = result["index"] == real_label
                    st.markdown("✅ Correcto" if acierto else "❌ Incorrecto")
            with c2:
                st.markdown("### Probabilidad por clase")
                probs = pd.Series(result["probabilities"], index=CLASS_NAMES_ES, name="Probabilidad")
                st.bar_chart(probs)

        st.markdown("---")
        st.subheader("Notas de despliegue y mantenimiento (CRISP-ML/Q)")
        nc = st.columns(3)
        nc[0].markdown(f"**Local**\n\n`{DEPLOYMENT_NOTES['local']}`")
        nc[1].markdown(f"**Nube**\n\n{DEPLOYMENT_NOTES['cloud']}")
        nc[2].markdown(f"**Contenedor**\n\n{DEPLOYMENT_NOTES['contenedor']}")
        st.markdown("**Monitorización y Mantenimiento (fase transversal):**")
        for note in DEPLOYMENT_NOTES["monitorizacion"]:
            st.markdown(f"- {note}")

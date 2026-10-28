"""
Ejecuta todos los experimentos del Equipo 6 (Familia 3) y guarda resultados.

Uso:
    python -m src.run_all

Genera figuras en results/figures, tablas en results/tables y modelos en
results/models.
"""

from __future__ import annotations

import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    KFold,
    learning_curve,
    cross_val_score,
)
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from . import config
from .data_loader import (
    cargar_20newsgroups,
    crear_conjunto_desbalanceado_nb,
    cargar_california_housing,
    cargar_clasificacion_lda_qda,
    crear_dataset_sintetico_covarianzas_distintas,
)
from .preprocessing import dividir_datos
from .models_nb import crear_pipeline_nb, obtener_palabras_informativas
from .models_mlp import crear_pipeline_mlp
from .models_discriminant import crear_pipeline_lda, crear_pipeline_qda
from . import evaluation as ev

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _guardar_tabla(df: pd.DataFrame, nombre: str) -> Path:
    """Guarda un DataFrame como CSV en results/tables."""
    ruta = config.TABLES_DIR / nombre
    df.to_csv(ruta, index=False, encoding="utf-8-sig")
    return ruta


def _medir_tiempos(modelo, X_train, y_train, X_test) -> Tuple[float, float]:
    """Devuelve (tiempo_entrenamiento, tiempo_inferencia) en segundos."""
    t0 = time.perf_counter()
    modelo.fit(X_train, y_train)
    t_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    modelo.predict(X_test)
    t_inf = time.perf_counter() - t0
    return t_train, t_inf


# ---------------------------------------------------------------------------
# Fase 3.1 - Naive Bayes
# ---------------------------------------------------------------------------

def ejecutar_naive_bayes() -> Dict[str, Any]:
    """Experimentos completos de MultinomialNB y ComplementNB."""
    print("\n" + "=" * 70)
    print("FASE 3.1 - NAIVE BAYES (MultinomialNB / ComplementNB)")
    print("=" * 70)

    textos, y = cargar_20newsgroups()
    nombres_clases = [f"clase_{i}" for i in range(len(np.unique(y)))]

    # Categorías legibles si se usó el dataset real
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
        nombres_clases = list(data.target_names)
    except Exception:
        pass

    X_train, X_test, y_train, y_test = dividir_datos(
        textos, y, estratificar=True
    )
    print(f"Entrenamiento: {len(X_train)} docs | Prueba: {len(X_test)} docs")

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    espacio_grid = {
        "vectorizer__tipo": ["count", "tfidf"],
        "vectorizer__ngram_range": [(1, 1), (1, 2)],
        "vectorizer__min_df": [1, 3],
        "clf__alpha": [0.1, 0.5, 1.0],
        "clf__fit_prior": [True, False],
    }

    resultados_modelos: list[Dict[str, Any]] = []
    mejores: Dict[str, Any] = {}

    for clasificador, etiqueta in [
        ("multinomial", "MultinomialNB"),
        ("complement", "ComplementNB"),
    ]:
        print(f"\n--- Búsqueda de hiperparámetros: {etiqueta} ---")
        pipeline = crear_pipeline_nb(clasificador=clasificador)
        grid = GridSearchCV(
            pipeline,
            param_grid=espacio_grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        t0 = time.perf_counter()
        grid.fit(X_train, y_train)
        t_busqueda = time.perf_counter() - t0

        _guardar_tabla(
            pd.DataFrame(grid.cv_results_),
            f"nb_{clasificador}_cv_results.csv",
        )

        mejor = grid.best_estimator_
        mejores[clasificador] = mejor

        t0 = time.perf_counter()
        y_pred = mejor.predict(X_test)
        t_inf = time.perf_counter() - t0
        y_proba = mejor.predict_proba(X_test)

        # Tiempo de entrenamiento del mejor modelo (refit aproximado)
        t0 = time.perf_counter()
        mejor.fit(X_train, y_train)
        t_train = time.perf_counter() - t0

        metricas = ev.evaluar_clasificacion(y_test, y_pred, y_proba, etiqueta)
        metricas.update({
            "mejores_parametros": json.dumps(grid.best_params_, ensure_ascii=False),
            "f1_cv": float(grid.best_score_),
            "tiempo_busqueda_s": float(t_busqueda),
            "tiempo_entrenamiento_s": float(t_train),
            "tiempo_inferencia_s": float(t_inf),
        })
        resultados_modelos.append(metricas)

        print(f"Mejores parámetros: {grid.best_params_}")
        print(f"F1 macro CV: {grid.best_score_:.4f} | F1 macro test: {metricas['f1_macro']:.4f}")

        ev.guardar_matriz_confusion(
            y_test, y_pred, etiqueta, labels=list(range(len(nombres_clases)))
        )
        ev.guardar_curvas_roc(
            y_test, y_proba, etiqueta, clases=list(range(len(nombres_clases)))
        )
        ev.guardar_classification_report(
            y_test, y_pred, etiqueta, etiquetas=nombres_clases
        )

        # Top palabras por clase
        filas_palabras = []
        for clase_idx, nombre_clase in enumerate(nombres_clases):
            palabras = obtener_palabras_informativas(mejor, clase_idx, top_n=15)
            for rango, (palabra, peso) in enumerate(palabras, start=1):
                filas_palabras.append({
                    "modelo": etiqueta,
                    "clase": nombre_clase,
                    "rango": rango,
                    "palabra": palabra,
                    "peso_log": peso,
                })
            if palabras:
                fig, ax = plt.subplots(figsize=(8, 5))
                pal = [p for p, _ in palabras][::-1]
                pesos = [w for _, w in palabras][::-1]
                ax.barh(pal, pesos, color=sns.color_palette("viridis", len(pal)))
                ax.set_title(f"Top palabras - {etiqueta} - {nombre_clase}")
                ax.set_xlabel("log P(palabra | clase) / peso")
                plt.tight_layout()
                plt.savefig(
                    config.FIGURES_DIR
                    / f"palabras_{clasificador}_{clase_idx}.png",
                    dpi=config.DPI_FIG,
                    bbox_inches="tight",
                )
                plt.close()

        _guardar_tabla(
            pd.DataFrame(filas_palabras),
            f"nb_{clasificador}_palabras_informativas.csv",
        )

        joblib.dump(mejor, config.MODELS_DIR / f"{clasificador}_nb.joblib")

    df_nb = pd.DataFrame(resultados_modelos)
    _guardar_tabla(df_nb, "nb_resultados.csv")
    print("\nResultados NB:")
    print(df_nb[[
        "modelo", "accuracy", "f1_macro", "f1_weighted", "roc_auc_ovr_macro"
    ]].to_string(index=False))

    # ------------------------------------------------------------------
    # Conjunto desbalanceado: MultinomialNB vs ComplementNB
    # ------------------------------------------------------------------
    print("\n--- Evaluación en conjunto desbalanceado ---")
    textos_desb, y_desb = crear_conjunto_desbalanceado_nb(
        textos, y, proporcion_clase_minoritaria=0.1
    )
    X_tr_d, X_te_d, y_tr_d, y_te_d = dividir_datos(
        textos_desb, y_desb, estratificar=True
    )
    print("Distribución de clases (entrenamiento):",
          dict(zip(*np.unique(y_tr_d, return_counts=True))))

    resultados_desb = []
    for clasificador, etiqueta in [
        ("multinomial", "MultinomialNB"),
        ("complement", "ComplementNB"),
    ]:
        mejor_alpha = 1.0
        if clasificador in mejores:
            mejor_alpha = mejores[clasificador].named_steps["clf"].alpha
        pipe = crear_pipeline_nb(
            tipo_vectorizador="tfidf",
            clasificador=clasificador,
            alpha=mejor_alpha,
        )
        pipe.fit(X_tr_d, y_tr_d)
        y_pred = pipe.predict(X_te_d)
        f1s = f1_score(y_te_d, y_pred, average=None, labels=list(range(len(nombres_clases))), zero_division=0)
        resultados_desb.append({
            "modelo": etiqueta,
            "alpha": mejor_alpha,
            "accuracy": float(accuracy_score(y_te_d, y_pred)),
            "f1_macro": float(f1_score(y_te_d, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_te_d, y_pred, average="weighted", zero_division=0)),
            "f1_clase_minoritaria": float(f1s[0]),
            "f1_por_clase": json.dumps([float(x) for x in f1s]),
        })
        joblib.dump(pipe, config.MODELS_DIR / f"desbalanceado_{clasificador}_nb.joblib")

    df_desb = pd.DataFrame(resultados_desb)
    _guardar_tabla(df_desb, "nb_comparacion_desbalanceado.csv")

    # Gráfica comparativa
    fig, ax = plt.subplots(figsize=config.FIGSIZE_STD)
    x_pos = np.arange(len(nombres_clases))
    ancho = 0.35
    for i, fila in df_desb.iterrows():
        f1s = json.loads(fila["f1_por_clase"])
        ax.bar(x_pos + (i - 0.5) * ancho, f1s, ancho, label=fila["modelo"])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(nombres_clases, rotation=25, ha="right")
    ax.set_ylabel("F1-score")
    ax.set_title("MultinomialNB vs ComplementNB en datos desbalanceados")
    ax.legend()
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "nb_desbalanceado_comparacion.png",
                dpi=config.DPI_FIG, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # Sensibilidad a alpha
    # ------------------------------------------------------------------
    print("\n--- Análisis de sensibilidad a alpha ---")
    alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    filas_sens = []
    for clasificador, etiqueta in [
        ("multinomial", "MultinomialNB"),
        ("complement", "ComplementNB"),
    ]:
        for alpha in alphas:
            pipe = crear_pipeline_nb(
                tipo_vectorizador="tfidf",
                clasificador=clasificador,
                alpha=alpha,
            )
            scores = cross_val_score(
                pipe, textos, y, cv=cv, scoring="f1_macro", n_jobs=-1
            )
            filas_sens.append({
                "modelo": etiqueta,
                "alpha": alpha,
                "f1_macro": float(scores.mean()),
                "f1_std": float(scores.std()),
            })
    df_sens = pd.DataFrame(filas_sens)
    _guardar_tabla(df_sens, "nb_sensibilidad_alpha.csv")
    ev.guardar_curva_alpha_nb(df_sens)

    return {
        "resultados": df_nb,
        "desbalanceado": df_desb,
        "sensibilidad": df_sens,
        "mejores": mejores,
    }


# ---------------------------------------------------------------------------
# Fase 3.2 - MLPRegressor
# ---------------------------------------------------------------------------

def ejecutar_mlp() -> Dict[str, Any]:
    """Experimentos completos de MLPRegressor con California Housing."""
    print("\n" + "=" * 70)
    print("FASE 3.2 - MLPREGRESSOR (California Housing)")
    print("=" * 70)

    X, y = cargar_california_housing()
    print(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} variables")
    print(f"Variables: {list(X.columns)}")

    X_train, X_test, y_train, y_test = dividir_datos(
        X, y, estratificar=False
    )

    # Submuestra para la búsqueda (decisión documentada para controlar costo)
    X_busqueda, _, y_busqueda, _ = dividir_datos(
        X_train, y_train, test_size=0.5, estratificar=False
    )
    print(f"Submuestra para búsqueda: {len(X_busqueda)} | "
          f"Entrenamiento final: {len(X_train)}")

    pipeline = crear_pipeline_mlp()
    espacio = {
        "mlp__hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [0.0001, 0.001, 0.01],
        "mlp__learning_rate_init": [0.001, 0.01],
        "mlp__solver": ["adam", "sgd"],
        "mlp__early_stopping": [True],
    }

    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    busqueda = RandomizedSearchCV(
        pipeline,
        param_distributions=espacio,
        n_iter=10,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
        refit=True,
        return_train_score=True,
    )
    t0 = time.perf_counter()
    busqueda.fit(X_busqueda, y_busqueda)
    t_busqueda = time.perf_counter() - t0
    print(f"Búsqueda completada en {t_busqueda:.1f}s")
    print(f"Mejores parámetros: {busqueda.best_params_}")
    print(f"RMSE CV: {-busqueda.best_score_:.4f}")

    _guardar_tabla(pd.DataFrame(busqueda.cv_results_), "mlp_cv_results.csv")

    mejor = busqueda.best_estimator_
    t_train, t_inf = _medir_tiempos(mejor, X_train, y_train, X_test)
    y_pred = mejor.predict(X_test)

    metricas = ev.evaluar_regresion(y_test, y_pred, "MLPRegressor")
    metricas.update({
        "mejores_parametros": json.dumps(
            {k: str(v) for k, v in busqueda.best_params_.items()}, ensure_ascii=False
        ),
        "rmse_cv": float(-busqueda.best_score_),
        "tiempo_busqueda_s": float(t_busqueda),
        "tiempo_entrenamiento_s": float(t_train),
        "tiempo_inferencia_s": float(t_inf),
        "n_iter_": int(mejor.named_steps["mlp"].n_iter_),
    })
    df_mlp = pd.DataFrame([metricas])
    _guardar_tabla(df_mlp, "mlp_resultados.csv")
    print(df_mlp[["modelo", "rmse", "mae", "r2", "tiempo_entrenamiento_s"]].to_string(index=False))

    ev.guardar_real_vs_predicho(y_test.values, y_pred, "MLPRegressor")
    ev.guardar_residuos(y_test.values, y_pred, "MLPRegressor")
    ev.guardar_curva_perdida(mejor, "MLPRegressor")

    joblib.dump(mejor, config.MODELS_DIR / "mlp_regressor.joblib")

    # Curvas de aprendizaje
    print("\n--- Curvas de aprendizaje ---")
    tamaños, train_scores, val_scores = learning_curve(
        mejor,
        X_busqueda,
        y_busqueda,
        train_sizes=np.linspace(0.1, 1.0, 6),
        cv=KFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    train_rmse = -train_scores
    val_rmse = -val_scores
    fig, ax = plt.subplots(figsize=config.FIGSIZE_STD)
    ax.plot(tamaños, train_rmse.mean(axis=1), "o-", label="Entrenamiento")
    ax.fill_between(tamaños, train_rmse.mean(axis=1) - train_rmse.std(axis=1),
                    train_rmse.mean(axis=1) + train_rmse.std(axis=1), alpha=0.2)
    ax.plot(tamaños, val_rmse.mean(axis=1), "s-", label="Validación")
    ax.fill_between(tamaños, val_rmse.mean(axis=1) - val_rmse.std(axis=1),
                    val_rmse.mean(axis=1) + val_rmse.std(axis=1), alpha=0.2)
    ax.set_xlabel("Tamaño del conjunto de entrenamiento")
    ax.set_ylabel("RMSE")
    ax.set_title("Curva de aprendizaje - MLPRegressor")
    ax.legend()
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "mlp_learning_curve.png",
                dpi=config.DPI_FIG, bbox_inches="tight")
    plt.close()

    _guardar_tabla(
        pd.DataFrame({
            "n_muestras": tamaños,
            "rmse_train_media": train_rmse.mean(axis=1),
            "rmse_train_std": train_rmse.std(axis=1),
            "rmse_val_media": val_rmse.mean(axis=1),
            "rmse_val_std": val_rmse.std(axis=1),
        }),
        "mlp_learning_curve.csv",
    )

    # Permutation importance (sobre submuestra del test por costo)
    print("\n--- Permutation importance ---")
    n_imp = min(2000, len(X_test))
    X_imp = X_test.iloc[:n_imp]
    y_imp = y_test.iloc[:n_imp]
    try:
        perm = permutation_importance(
            mejor, X_imp, y_imp,
            n_repeats=5,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            scoring="neg_root_mean_squared_error",
        )
        df_perm = pd.DataFrame({
            "variable": X.columns,
            "importancia_media": perm.importances_mean,
            "importancia_std": perm.importances_std,
        }).sort_values("importancia_media", ascending=False)
        _guardar_tabla(df_perm, "mlp_permutation_importance.csv")

        fig, ax = plt.subplots(figsize=config.FIGSIZE_STD)
        sns.barplot(data=df_perm, x="importancia_media", y="variable", ax=ax,
                    palette="viridis")
        ax.set_title("Permutation Importance - MLPRegressor")
        ax.set_xlabel("Incremento medio del RMSE")
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / "mlp_permutation_importance.png",
                    dpi=config.DPI_FIG, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"Advertencia permutation_importance: {exc}")
        df_perm = pd.DataFrame()

    # Sensibilidad al tamaño de la red
    print("\n--- Sensibilidad al tamaño de la red ---")
    X_sens, _, y_sens, _ = dividir_datos(
        X_train, y_train, test_size=0.5, estratificar=False
    )
    X_sens_tr, X_sens_val, y_sens_tr, y_sens_val = dividir_datos(
        X_sens, y_sens, test_size=0.2, estratificar=False
    )
    filas_red = []
    for capas in [(16,), (32,), (64,), (128,), (64, 64), (128, 64)]:
        pipe = crear_pipeline_mlp(
            hidden_layer_sizes=capas,
            max_iter=200,
            early_stopping=True,
            random_state=config.RANDOM_STATE,
        )
        t0 = time.perf_counter()
        pipe.fit(X_sens_tr, y_sens_tr)
        t_fit = time.perf_counter() - t0
        pred = pipe.predict(X_sens_val)
        filas_red.append({
            "arquitectura": str(capas),
            "n_parametros": int(sum(p.size for p in pipe.named_steps["mlp"].coefs_)),
            "rmse_val": float(np.sqrt(ev.mean_squared_error(y_sens_val, pred))),
            "tiempo_entrenamiento_s": float(t_fit),
        })
    df_red = pd.DataFrame(filas_red)
    _guardar_tabla(df_red, "mlp_sensibilidad_arquitectura.csv")

    fig, ax = plt.subplots(figsize=config.FIGSIZE_STD)
    ax.plot(df_red["arquitectura"], df_red["rmse_val"], "o-", color="tab:blue")
    ax.set_xlabel("Arquitectura de capas ocultas")
    ax.set_ylabel("RMSE validación")
    ax.set_title("Sensibilidad del MLP al tamaño de la red")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "mlp_sensibilidad_arquitectura.png",
                dpi=config.DPI_FIG, bbox_inches="tight")
    plt.close()

    return {
        "resultados": df_mlp,
        "permutation": df_perm,
        "sensibilidad": df_red,
        "mejor": mejor,
    }


# ---------------------------------------------------------------------------
# Fase 3.3 - LDA / QDA
# ---------------------------------------------------------------------------

def _evaluar_clasificador_discriminante(
    nombre: str,
    modelo,
    X_train,
    X_test,
    y_train,
    y_test,
    etiqueta: str,
) -> Dict[str, Any]:
    """Evalúa LDA o QDA y guarda figuras/tablas."""
    t_train, t_inf = _medir_tiempos(modelo, X_train, y_train, X_test)
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)
    metricas = ev.evaluar_clasificacion(y_test, y_pred, y_proba, etiqueta)
    metricas.update({
        "tiempo_entrenamiento_s": float(t_train),
        "tiempo_inferencia_s": float(t_inf),
    })
    ev.guardar_matriz_confusion(y_test, y_pred, etiqueta)
    ev.guardar_curvas_roc(y_test, y_proba, etiqueta)
    ev.guardar_classification_report(y_test, y_pred, etiqueta)
    joblib.dump(modelo, config.MODELS_DIR / f"{nombre}.joblib")
    return metricas


def ejecutar_lda_qda() -> Dict[str, Any]:
    """Experimentos completos de LDA y QDA."""
    print("\n" + "=" * 70)
    print("FASE 3.3 - LDA / QDA")
    print("=" * 70)

    X, y = cargar_clasificacion_lda_qda("breast_cancer")
    print(f"Dataset: Breast Cancer | {X.shape[0]} muestras, {X.shape[1]} variables")
    X_train, X_test, y_train, y_test = dividir_datos(X, y, estratificar=True)

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)

    resultados: list[Dict[str, Any]] = []
    mejores: Dict[str, Any] = {}

    # ---------------- LDA ----------------
    print("\n--- Búsqueda LDA ---")
    pipe_lda = crear_pipeline_lda()
    grid_lda = GridSearchCV(
        pipe_lda,
        {
            "lda__solver": ["svd", "lsqr", "eigen"],
            "lda__shrinkage": [None, "auto", 0.1, 0.5],
        },
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        error_score=0.0,
    )
    grid_lda.fit(X_train, y_train)
    _guardar_tabla(pd.DataFrame(grid_lda.cv_results_), "lda_cv_results.csv")
    print(f"Mejores parámetros LDA: {grid_lda.best_params_}")
    print(f"F1 macro CV: {grid_lda.best_score_:.4f}")
    mejor_lda = grid_lda.best_estimator_
    mejores["lda"] = mejor_lda
    met_lda = _evaluar_clasificador_discriminante(
        "lda", mejor_lda, X_train, X_test, y_train, y_test, "LDA"
    )
    met_lda["mejores_parametros"] = json.dumps(grid_lda.best_params_, ensure_ascii=False)
    met_lda["f1_cv"] = float(grid_lda.best_score_)
    resultados.append(met_lda)

    # ---------------- QDA ----------------
    print("\n--- Búsqueda QDA ---")
    pipe_qda = crear_pipeline_qda()
    grid_qda = GridSearchCV(
        pipe_qda,
        {"qda__reg_param": [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]},
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid_qda.fit(X_train, y_train)
    _guardar_tabla(pd.DataFrame(grid_qda.cv_results_), "qda_cv_results.csv")
    print(f"Mejores parámetros QDA: {grid_qda.best_params_}")
    print(f"F1 macro CV: {grid_qda.best_score_:.4f}")
    mejor_qda = grid_qda.best_estimator_
    mejores["qda"] = mejor_qda
    met_qda = _evaluar_clasificador_discriminante(
        "qda", mejor_qda, X_train, X_test, y_train, y_test, "QDA"
    )
    met_qda["mejores_parametros"] = json.dumps(grid_qda.best_params_, ensure_ascii=False)
    met_qda["f1_cv"] = float(grid_qda.best_score_)
    resultados.append(met_qda)

    df_ldaqda = pd.DataFrame(resultados)
    _guardar_tabla(df_ldaqda, "lda_qda_resultados.csv")
    print("\nResultados LDA/QDA:")
    print(df_ldaqda[["modelo", "accuracy", "f1_macro", "roc_auc", "tiempo_entrenamiento_s"]].to_string(index=False))

    # ------------------------------------------------------------------
    # Fronteras 2D con PCA (Breast Cancer)
    # ------------------------------------------------------------------
    print("\n--- Fronteras de decisión 2D (PCA) ---")
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    X_2d = pca.fit_transform(X)
    X2_tr, X2_te, y2_tr, y2_te = dividir_datos(X_2d, y, estratificar=True)
    lda_2d = LinearDiscriminantAnalysis().fit(X2_tr, y2_tr)
    qda_2d = QuadraticDiscriminantAnalysis().fit(X2_tr, y2_tr)
    ev.guardar_fronteras_decision_2d(
        {"LDA": lda_2d, "QDA": qda_2d},
        X2_te,
        y2_te,
        "breast_cancer_pca2",
        titulos={
            "LDA": "LDA (covarianza compartida)",
            "QDA": "QDA (covarianza por clase)",
        },
    )
    print(f"Varianza explicada PCA-2: {pca.explained_variance_ratio_.sum():.3f}")
    _guardar_tabla(
        pd.DataFrame({
            "componente": ["PC1", "PC2"],
            "varianza_explicada": pca.explained_variance_ratio_,
        }),
        "pca_breast_cancer_varianza.csv",
    )

    # ------------------------------------------------------------------
    # Dataset sintético con covarianzas distintas
    # ------------------------------------------------------------------
    print("\n--- Dataset sintético con covarianzas distintas ---")
    X_syn, y_syn = crear_dataset_sintetico_covarianzas_distintas()
    Xs_tr, Xs_te, ys_tr, ys_te = dividir_datos(X_syn, y_syn, estratificar=True)
    lda_syn = LinearDiscriminantAnalysis().fit(Xs_tr, ys_tr)
    qda_syn = QuadraticDiscriminantAnalysis(reg_param=0.0).fit(Xs_tr, ys_tr)
    met_syn = []
    for nombre, modelo in [("LDA", lda_syn), ("QDA", qda_syn)]:
        pred = modelo.predict(Xs_te)
        met_syn.append(ev.evaluar_clasificacion(ys_te, pred, modelo.predict_proba(Xs_te), nombre))
    df_syn = pd.DataFrame(met_syn)
    _guardar_tabla(df_syn, "sintetico_lda_qda_resultados.csv")
    print(df_syn[["modelo", "accuracy", "f1_macro"]].to_string(index=False))

    ev.guardar_fronteras_decision_2d(
        {"LDA": lda_syn, "QDA": qda_syn},
        Xs_te,
        ys_te,
        "sintetico_covarianzas",
        titulos={
            "LDA": "LDA - frontera lineal",
            "QDA": "QDA - frontera cuadrática",
        },
    )

    # ------------------------------------------------------------------
    # Sensibilidad QDA reg_param y LDA shrinkage
    # ------------------------------------------------------------------
    print("\n--- Sensibilidad reg_param / shrinkage ---")
    filas_sens = []
    # Se excluye reg_param=0 porque en Breast Cancer la covarianza de una clase
    # no es de rango completo y QDA falla numéricamente.
    for reg in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]:
        pipe = crear_pipeline_qda(reg_param=reg)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        filas_sens.append({
            "modelo": "QDA",
            "parametro": "reg_param",
            "valor": reg,
            "f1_macro": float(scores.mean()),
            "f1_std": float(scores.std()),
        })
    for shrink in [None, "auto", 0.1, 0.3, 0.5, 0.7, 0.9]:
        pipe = crear_pipeline_lda(solver="lsqr", shrinkage=shrink)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        filas_sens.append({
            "modelo": "LDA",
            "parametro": "shrinkage",
            "valor": str(shrink),
            "f1_macro": float(scores.mean()),
            "f1_std": float(scores.std()),
        })
    df_sens = pd.DataFrame(filas_sens)
    _guardar_tabla(df_sens, "lda_qda_sensibilidad.csv")

    fig, axes = plt.subplots(1, 2, figsize=config.FIGSIZE_WIDE)
    qda_plot = df_sens[df_sens["modelo"] == "QDA"]
    axes[0].plot(qda_plot["valor"].astype(str), qda_plot["f1_macro"], "o-")
    axes[0].set_title("QDA: efecto de reg_param")
    axes[0].set_xlabel("reg_param")
    axes[0].set_ylabel("F1 macro (CV)")
    lda_plot = df_sens[df_sens["modelo"] == "LDA"]
    axes[1].plot(lda_plot["valor"], lda_plot["f1_macro"], "s-", color="tab:orange")
    axes[1].set_title("LDA: efecto de shrinkage")
    axes[1].set_xlabel("shrinkage")
    axes[1].set_ylabel("F1 macro (CV)")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "lda_qda_sensibilidad.png",
                dpi=config.DPI_FIG, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # Proyección discriminante con Wine (3 clases -> 2 componentes)
    # ------------------------------------------------------------------
    print("\n--- Proyección discriminante LDA (Wine) ---")
    Xw, yw = cargar_clasificacion_lda_qda("wine")
    Xw_train, Xw_test, yw_train, yw_test = dividir_datos(Xw, yw, estratificar=True)
    pipe_w = crear_pipeline_lda(solver="svd")
    pipe_w.fit(Xw_train, yw_train)
    lda_w = pipe_w.named_steps["lda"]
    Xw_proj = lda_w.transform(pipe_w.named_steps["scaler"].transform(Xw_test))
    var_proj = np.var(Xw_proj, axis=0)
    var_ratio = var_proj / var_proj.sum()
    _guardar_tabla(
        pd.DataFrame({
            "componente": [f"LD{i+1}" for i in range(len(var_ratio))],
            "varianza_proyeccion": var_proj,
            "proporcion": var_ratio,
        }),
        "lda_wine_varianza_proyeccion.csv",
    )

    fig, axes = plt.subplots(1, 2, figsize=config.FIGSIZE_WIDE)
    for clase in np.unique(yw_test):
        mask = yw_test == clase
        axes[0].scatter(Xw_proj[mask, 0], Xw_proj[mask, 1], label=f"Clase {clase}", alpha=0.7)
    axes[0].set_xlabel("LD1")
    axes[0].set_ylabel("LD2")
    axes[0].set_title("Proyección discriminante LDA - Wine")
    axes[0].legend()
    axes[1].bar([f"LD{i+1}" for i in range(len(var_ratio))], var_ratio, color=sns.color_palette("viridis", len(var_ratio)))
    axes[1].set_ylabel("Proporción de varianza proyectada")
    axes[1].set_title("Varianza por componente discriminante")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "lda_proyeccion_wine.png",
                dpi=config.DPI_FIG, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # Visualización 3D con Plotly (opcional)
    # ------------------------------------------------------------------
    try:
        import plotly.express as px
        pca3 = PCA(n_components=3, random_state=config.RANDOM_STATE)
        X_3d = pca3.fit_transform(X)
        fig = px.scatter_3d(
            x=X_3d[:, 0], y=X_3d[:, 1], z=X_3d[:, 2],
            color=y.astype(str),
            labels={"x": "PC1", "y": "PC2", "z": "PC3", "color": "Clase"},
            title="Breast Cancer - PCA 3D (Plotly)",
            opacity=0.7,
        )
        fig.write_html(config.FIGURES_DIR / "breast_cancer_pca3d.html")
        print("Figura 3D guardada como HTML.")
    except Exception as exc:
        print(f"Advertencia figura 3D: {exc}")

    return {
        "resultados": df_ldaqda,
        "sintetico": df_syn,
        "sensibilidad": df_sens,
        "mejores": mejores,
    }


# ---------------------------------------------------------------------------
# Fase 3.4 - Comparación global
# ---------------------------------------------------------------------------

def generar_comparacion(res_nb, res_mlp, res_disc) -> pd.DataFrame:
    """Construye tabla comparativa entre las tres familias."""
    filas = []
    for _, fila in res_nb["resultados"].iterrows():
        filas.append({
            "familia": "Naive Bayes",
            "modelo": fila["modelo"],
            "tipo_tarea": "Clasificación",
            "metrica_principal": "F1 macro",
            "valor_metrica": fila["f1_macro"],
            "accuracy_o_r2": fila["accuracy"],
            "tiempo_entrenamiento_s": fila["tiempo_entrenamiento_s"],
            "tiempo_inferencia_s": fila["tiempo_inferencia_s"],
            "mejores_parametros": fila["mejores_parametros"],
        })
    for _, fila in res_disc["resultados"].iterrows():
        filas.append({
            "familia": "Discriminante",
            "modelo": fila["modelo"],
            "tipo_tarea": "Clasificación",
            "metrica_principal": "F1 macro",
            "valor_metrica": fila["f1_macro"],
            "accuracy_o_r2": fila["accuracy"],
            "tiempo_entrenamiento_s": fila["tiempo_entrenamiento_s"],
            "tiempo_inferencia_s": fila["tiempo_inferencia_s"],
            "mejores_parametros": fila["mejores_parametros"],
        })
    for _, fila in res_mlp["resultados"].iterrows():
        filas.append({
            "familia": "Red neuronal",
            "modelo": fila["modelo"],
            "tipo_tarea": "Regresión",
            "metrica_principal": "RMSE",
            "valor_metrica": fila["rmse"],
            "accuracy_o_r2": fila["r2"],
            "tiempo_entrenamiento_s": fila["tiempo_entrenamiento_s"],
            "tiempo_inferencia_s": fila["tiempo_inferencia_s"],
            "mejores_parametros": fila["mejores_parametros"],
        })
    df = pd.DataFrame(filas)
    _guardar_tabla(df, "comparacion_modelos.csv")
    print("\n=== COMPARACIÓN GLOBAL ===")
    print(df[["familia", "modelo", "tipo_tarea", "metrica_principal",
              "valor_metrica", "tiempo_entrenamiento_s"]].to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main() -> None:
    """Ejecuta todas las fases y guarda resultados."""
    print("=" * 70)
    print("PROYECTO EQUIPO 6 - FAMILIA 3 (Probabilísticos y Estructurados)")
    print("=" * 70)

    t_global = time.perf_counter()

    res_nb = ejecutar_naive_bayes()
    res_mlp = ejecutar_mlp()
    res_disc = ejecutar_lda_qda()
    generar_comparacion(res_nb, res_mlp, res_disc)

    print("\n" + "=" * 70)
    print(f"TODO COMPLETADO en {time.perf_counter() - t_global:.1f}s")
    print(f"Figuras: {config.FIGURES_DIR}")
    print(f"Tablas:  {config.TABLES_DIR}")
    print(f"Modelos: {config.MODELS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

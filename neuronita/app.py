# ============================================================================
#  APLICACIÓN WEB INTERACTIVA: IMPLEMENTACIÓN Y SIMULACIÓN DEL
#  PERCEPTRÓN SIMPLE DESDE CERO (STREAMLIT)
#
#  - Lógica matemática, entrenamiento y predicción: SOLO numpy.
#  - Visualización: SOLO matplotlib (mostrada en la web con st.pyplot).
#  - Sin scikit-learn, tensorflow, pytorch u otros frameworks de IA.
#  - Para ejecutar:  streamlit run app.py
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# ============================================================================
#  PARTE 1: CLASE Perceptron (Perceptrón de Rosenblatt, implementado desde cero)
# ============================================================================

class Perceptron:
    """
    Neurona artificial de un solo nivel: Perceptrón de Rosenblatt.

    Parámetros
    ----------
    input_size : int
        Número de entradas (dimensión del vector x).
    learning_rate : float
        Tasa de aprendizaje eta (0 < eta <= 1).
    epochs : int
        Número máximo de épocas de entrenamiento.
    """

    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.lr = learning_rate                 # eta: tasa de aprendizaje
        self.epochs = epochs                    # máximo de épocas permitidas
        self.weights = np.zeros(input_size)     # vector de pesos w (inicializado en ceros)
        self.bias = 0.0                         # sesgo b (bias)
        self.errors_history = []                # errores por época (para la gráfica de convergencia)
        self.convergence_epoch = None           # época en la que convergió (None si no convergió)

    def activation_function(self, z):
        """
        Función de activación: escalón de Heaviside.
        Devuelve 1 si z >= 0, 0 en caso contrario.
        """
        return 1.0 if z >= 0 else 0.0

    def predict(self, x):
        """
        Predice la clase de una muestra x.

        z = w^T x + b  ->  f(z)
        """
        z = np.dot(x, self.weights) + self.bias
        return self.activation_function(z)

    def fit(self, X, y):
        """
        Entrena el Perceptrón aplicando la regla de aprendizaje de Rosenblatt:

            w_j  <-  w_j + eta * (y - y_hat) * x_j
            b    <-  b   + eta * (y - y_hat)

        X : matriz de entrenamiento (cada fila es una muestra).
        y : vector de etiquetas reales (0 o 1).

        Aplica parada temprana si el error total de una época es cero.
        """
        for epoch in range(1, self.epochs + 1):
            total_errors = 0  # contador de errores dentro de la época actual

            for xi, target in zip(X, y):
                prediction = self.predict(xi)   # y_hat: predicción del modelo
                error = target - prediction     # e = y - y_hat

                # Si hay error (e != 0) se aplica la regla de actualización
                if error != 0:
                    self.weights += self.lr * error * np.asarray(xi)
                    self.bias += self.lr * error
                    total_errors += 1

            self.errors_history.append(total_errors)

            # Criterio de parada temprana: error cero -> convergencia
            if total_errors == 0:
                self.convergence_epoch = epoch
                break
            # Si se agota el máximo de épocas sin error cero, convergence_epoch
            # permanece en None (el modelo NO convergió).


# ============================================================================
#  PARTE 2: DATASETS DE LAS COMPUERTAS LÓGICAS
# ============================================================================

def crear_dataset(compuerta):
    """
    Devuelve la matriz de entradas X y el vector de etiquetas y según la
    compuerta lógica seleccionada (AND, OR o XOR).
    """
    # Matriz de entradas (idéntica para las tres compuertas)
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    if compuerta == "AND":
        y = np.array([0, 0, 0, 1])   # y solo es 1 cuando x1 y x2 son 1
    elif compuerta == "OR":
        y = np.array([0, 1, 1, 1])   # y es 1 si al menos una entrada es 1
    else:                            # compuerta XOR
        y = np.array([0, 1, 1, 0])   # y es 1 únicamente si las entradas difieren

    return X, y


def tabla_verdad(compuerta):
    """Devuelve la tabla de verdad de la compuerta elegida, en formato Markdown."""
    return {
        "AND": """
| $x_1$ | $x_2$ | $y$ |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |
""",
        "OR": """
| $x_1$ | $x_2$ | $y$ |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |
""",
        "XOR": """
| $x_1$ | $x_2$ | $y$ |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |
""",
    }[compuerta]


# ============================================================================
#  PARTE 3: VISUALIZACIÓN (convergencia + límite de decisión)
# ============================================================================

def crear_figura(perceptron, X, y, compuerta, lr, epocas):
    """
    Genera una figura de matplotlib con dos paneles:

    1) Convergencia: Épocas vs. Número de errores.
    2) Límite de decisión: puntos de las clases en 2D y la recta divisoria
       despejando x2 de la ecuación w1*x1 + w2*x2 + b = 0:
           x2 = -(w1*x1 + b) / w2
    """
    w1, w2 = perceptron.weights
    b = perceptron.bias

    # ---- Configuración general de la figura ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle(f"Simulación del Perceptrón - Compuerta {compuerta} "
                 f"(η = {lr}, épocas máx. = {epocas})",
                 fontsize=13, fontweight="bold")

    # ==========================================================
    # Panel 1: Gráfica de convergencia (épocas vs errores)
    # ==========================================================
    epochs = np.arange(1, len(perceptron.errors_history) + 1)
    errors = np.array(perceptron.errors_history)

    ax1.plot(epochs, errors, marker="o", color="#1f77b4",
             linewidth=2, label="Errores por época")
    ax1.set_title("Convergencia: Épocas vs. Número de errores")
    ax1.set_xlabel("Épocas")
    ax1.set_ylabel("Número de errores")
    ax1.set_xticks(epochs)
    ax1.set_ylim(-0.2, None)
    ax1.grid(True, linestyle="--", alpha=0.6)

    if perceptron.convergence_epoch is not None:
        ax1.axvline(perceptron.convergence_epoch, color="green",
                    linestyle=":", linewidth=2, alpha=0.8)
        ax1.text(perceptron.convergence_epoch, ax1.get_ylim()[1] * 0.9,
                 f" Converge en\n época {perceptron.convergence_epoch}",
                 color="green", fontsize=9, va="top")
    else:
        ax1.text(0.5, 0.9, "No converge (se agotaron las épocas)",
                 transform=ax1.transAxes, color="red", fontsize=9)
    ax1.legend(loc="upper right")

    # ==========================================================
    # Panel 2: Límite de decisión (decision boundary) en 2D
    # ==========================================================
    # Rejilla fina para sombrear la región donde el modelo predice la clase 1
    x_lin = np.linspace(-0.4, 1.4, 150)
    X1, X2 = np.meshgrid(x_lin, x_lin)
    Z = np.array([perceptron.predict(np.array([xx, yy]))
                  for xx, yy in zip(X1.ravel(), X2.ravel())]).reshape(X1.shape)
    ax2.contourf(X1, X2, Z, levels=[-0.5, 0.5],
                 colors=["#cfe8ff", "#ffe0c0"], alpha=0.6)

    # Puntos del dataset: clase 1 en rojo, clase 0 en azul
    for xi, yi in zip(X, y):
        color = "#d62728" if yi == 1 else "#1f77b4"
        ax2.scatter(xi[0], xi[1], c=color, s=180, edgecolors="k",
                    linewidths=1.2, zorder=3)

    # Recta divisoria: x2 = -(w1*x1 + b) / w2
    if abs(w2) > 1e-12:
        xx = np.linspace(-0.4, 1.4, 200)
        xx2 = -(w1 * xx + b) / w2
        ax2.plot(xx, xx2, color="green", linewidth=2.2,
                 label=f"Frontera: $x_2 = -({w1:.2f}\\,x_1 {b:+.2f})\\,/\\,{w2:.2f}$")
    else:
        # Caso degenerado w2 = 0 (frontera vertical): x1 = -b / w1
        x1b = -b / w1 if abs(w1) > 1e-12 else 0.0
        ax2.axvline(x1b, color="green", linewidth=2.2,
                    label=f"Frontera: $x_1 = {x1b:.2f}$")

    ax2.set_title("Límite de decisión")
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
    ax2.set_xlim(-0.4, 1.4)
    ax2.set_ylim(-0.4, 1.4)
    ax2.set_aspect("equal")
    ax2.legend(loc="best")
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ============================================================================
#  INTERFAZ GRÁFICA (STREAMLIT)
# ============================================================================

def main():
    # ---- Configuración de la página ----
    st.set_page_config(page_title="Perceptrón Simple desde Cero",
                       layout="centered")

    # ---- Encabezado y descripción ----
    st.title("Perceptrón simple desde cero")
    st.markdown(
        """
        Esta aplicación implementa y simula el **Perceptrón de Rosenblatt**
        (neurona artificial 2D de un solo nivel) **desde cero**, usando
        únicamente `numpy` para la lógica y `matplotlib` para las gráficas.

        El Perceptrón calcula una combinación lineal ponderada más un sesgo y
        la pasa por una función escalón de **Heaviside**:

        $$
        z = w_1 x_1 + w_2 x_2 + b, \\qquad
        \\hat{y} = f(z) = \\begin{cases} 1 & \\text{si } z \\geq 0 \\\\ 0 & \\text{si } z < 0 \\end{cases}
        $$

        Regla de actualización de pesos: \\( w_j \\leftarrow w_j + \\eta (y - \\hat{y}) x_j \\)
        y \\( b \\leftarrow b + \\eta (y - \\hat{y}) \\).
        """
    )

    # ---- Panel lateral (sidebar) con los controles ----
    with st.sidebar:
        st.header("Configuración")
        compuerta = st.selectbox(
            "Compuerta lógica a simular",
            options=["AND", "OR", "XOR"],
            help="AND, OR y XOR son problemas de clasificación binaria. "
                 "AND y OR son linealmente separables; XOR no lo es, por lo "
                 "que el Perceptrón simple NO podrá converger.",
        )
        lr = st.slider(
            "Tasa de aprendizaje (η)",
            min_value=0.01, max_value=1.0, value=0.1, step=0.01, format="%.2f",
            help="Controla el tamaño de cada corrección de pesos. "
                 "Valores muy pequeños convergen lento; valores grandes "
                 "pueden 'rebasar' la frontera óptima.",
        )
        epocas = st.slider(
            "Épocas máximas",
            min_value=1, max_value=500, value=100, step=1,
            help="Número máximo de pasadas completas sobre el dataset.",
        )
        entrenar = st.button("Entrenar Perceptrón", type="primary", use_container_width=True)

        st.divider()
        st.caption("Lógica: `numpy`  |  Gráficas: `matplotlib`")

    # ---- Tabla de verdad de la compuerta seleccionada (informativa) ----
    with st.expander(f"Ver tabla de verdad de la compuerta {compuerta}"):
        st.markdown(tabla_verdad(compuerta))

    # ---- Ejecución del entrenamiento al presionar el botón ----
    if entrenar:
        X, y = crear_dataset(compuerta)

        # Instancia y entrena el perceptrón con los parámetros del sidebar
        perceptron = Perceptron(input_size=2, learning_rate=lr, epochs=epocas)
        perceptron.fit(X, y)

        # ------------------------------------------------------------
        # Mensaje de éxito o advertencia según la convergencia
        # ------------------------------------------------------------
        if perceptron.convergence_epoch is not None:
            st.success(
                f"El Perceptrón **convergió** en la época {perceptron.convergence_epoch} "
                f"para la compuerta **{compuerta}** con η = {lr}."
            )
        else:
            st.warning(
                f"El Perceptrón **NO convergió** tras {epocas} épocas para la "
                f"compuerta **{compuerta}** con η = {lr}. "
                "(Resultado esperado: la XOR no es linealmente separable, "
                "una sola recta no puede separar sus clases.)"
            )

        # ------------------------------------------------------------
        # Gráficas: convergencia + límite de decisión en una misma figura
        # ------------------------------------------------------------
        figura = crear_figura(perceptron, X, y, compuerta, lr, epocas)
        st.pyplot(figura)

        st.divider()

        # ------------------------------------------------------------
        # Pesos finales del modelo entrenado
        # ------------------------------------------------------------
        st.subheader("Parámetros finales del modelo")
        col1, col2, col3 = st.columns(3)
        col1.metric("Peso w1", f"{perceptron.weights[0]:.4f}")
        col2.metric("Peso w2", f"{perceptron.weights[1]:.4f}")
        col3.metric("Sesgo b (bias)", f"{perceptron.bias:.4f}")

        st.markdown(
            f"La neurona implementa la regla de decisión "
            f"**z = {perceptron.weights[0]:.3f}·x₁ + {perceptron.weights[1]:.3f}·x₂ "
            f"{perceptron.bias:+.3f}**; emite clase **1** si z ≥ 0 y clase **0** si z < 0."
        )

        # ------------------------------------------------------------
        # Verificación: comparación entre valores reales y predicciones
        # ------------------------------------------------------------
        st.subheader("Verificación sobre la tabla de verdad")
        predicciones = [int(perceptron.predict(xi)) for xi in X]
        aciertos = sum(1 for p, yi in zip(predicciones, y) if p == yi)
        st.table({
            "x1": X[:, 0].tolist(),
            "x2": X[:, 1].tolist(),
            "y (real)": y.tolist(),
            "ŷ (predicción)": predicciones,
        })
        st.info(
            f"El modelo acierta **{aciertos} de {len(y)}** muestras "
            f"({aciertos / len(y) * 100:.0f} % de precisión)."
        )


# Punto de entrada: Streamlit ejecuta el script como __main__
if __name__ == "__main__":
    main()
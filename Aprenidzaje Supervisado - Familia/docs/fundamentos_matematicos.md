# Fundamentos matemáticos — Familia 3: Probabilísticos y Estructurados

**Equipo 6**  
Curso de Inteligencia Artificial  
Modelos: Naive Bayes Multinomial/Complement, Perceptrón Multicapa para regresión (MLPRegressor) y Análisis Discriminante Lineal/Cuadrático (LDA/QDA).

---

## 1. Naive Bayes Multinomial y Complement

### 1.1 Formulación e hipótesis

Sea un documento representado por el vector de conteos (o frecuencias TF-IDF no negativas)  
\(\mathbf{x} = (x_1, \dots, x_d)\), donde \(x_j\) es el número de apariciones del término \(j\), y sea \(y \in \{1, \dots, K\}\) la clase.

El teorema de Bayes establece

\[
P(y=k \mid \mathbf{x}) = \frac{P(y=k)\,P(\mathbf{x} \mid y=k)}{P(\mathbf{x})}.
\]

La hipótesis *naive* es la independencia condicional de los términos dada la clase:

\[
P(\mathbf{x} \mid y=k) \approx \prod_{j=1}^{d} P(x_j \mid y=k).
\]

En el modelo multinomial, la verosimilitud de un documento con \(n = \sum_j x_j\) tokens es

\[
P(\mathbf{x} \mid y=k) = \frac{n!}{\prod_j x_j!} \prod_{j=1}^{d} \theta_{kj}^{x_j},
\]

donde \(\theta_{kj} = P(\text{término } j \mid y=k)\) y \(\sum_j \theta_{kj} = 1\).

La regla de decisión en log-probabilidades (numéricamente estable) es

\[
\hat{y} = \arg\max_{k} \left[ \log P(y=k) + \sum_{j=1}^{d} x_j \log \theta_{kj} \right].
\]

### 1.2 Criterio de optimización

Máxima verosimilitud (equivalentemente, minimización de la entropía cruzada entre la distribución empírica y el modelo). Para cada clase se maximiza

\[
\ell(\boldsymbol{\theta}_k) = \sum_{i:\, y_i=k} \sum_{j=1}^{d} x_{ij} \log \theta_{kj}.
\]

Con el suavizado de Laplace/Lidstone de parámetro \(\alpha > 0\):

\[
\hat{\theta}_{kj} = \frac{\alpha + \sum_{i:\, y_i=k} x_{ij}}{\alpha d + \sum_{j=1}^{d}\sum_{i:\, y_i=k} x_{ij}}.
\]

Los priors se estiman como \(\hat{P}(y=k) = N_k / N\) si `fit_prior=True`; en caso contrario se usa \(1/K\).

### 1.3 Algoritmo de aprendizaje

1. Vectorizar los documentos (CountVectorizer o TfidfVectorizer).
2. Contar la frecuencia de cada término por clase.
3. Estimar \(\hat{\theta}_{kj}\) con suavizado \(\alpha\).
4. Estimar los priors.
5. Clasificar maximizando el log-score anterior.

Complejidad: \(O(N\,d)\) para entrenar y \(O(d)\) por documento para predecir, con \(N\) documentos y \(d\) términos del vocabulario.

### 1.4 Complement Naive Bayes

ComplementNB estima los parámetros a partir del **complemento** de cada clase, es decir, de todos los documentos que **no** pertenecen a \(k\):

\[
\hat{\theta}_{kj}^{\text{comp}} = \frac{\alpha + \sum_{i:\, y_i \neq k} x_{ij}}{\alpha d + \sum_{j=1}^{d}\sum_{i:\, y_i \neq k} x_{ij}}.
\]

El peso de decisión se define como

\[
w_{kj} = \log \hat{\theta}_{kj}^{\text{comp}}, \qquad
\text{score}_k(\mathbf{x}) = \sum_j x_j w_{kj}.
\]

Si `norm=True`, los pesos se normalizan:

\[
w_{kj} \leftarrow \frac{w_{kj}}{\sum_{j'} |w_{kj'}|}.
\]

La clase predicha es \(\hat{y} = \arg\min_k \text{score}_k(\mathbf{x})\), porque un documento de la clase \(k\) debe parecerse **poco** al complemento de \(k\).

**¿Por qué mejora con datos desbalanceados?** En MultinomialNB, los términos de una clase mayoritaria dominan la estimación y los priors favorecen esa clase. ComplementNB estima cada clase a partir de su complemento: la clase minoritaria se compara contra un complemento grande y estable, lo que reduce el sesgo hacia la clase mayoritaria. Además, la normalización de pesos evita que las clases con más documentos acumulen magnitudes mayores.

### 1.5 Supuestos y limitaciones

- **Supuesto fuerte:** independencia condicional de los términos; en texto real las palabras están correlacionadas.
- Solo admite características no negativas (conteos o TF-IDF); **no se debe aplicar StandardScaler**.
- Ignora el orden de las palabras (salvo que se usen n-gramas).
- Muy sensible a la tokenización, al vocabulario y al parámetro \(\alpha\).
- Es un modelo lineal en el espacio de log-conteos: su frontera de decisión es lineal.
- Excelente baseline por su bajo costo, pero puede quedar por debajo de modelos que capturan interacciones.

---

## 2. Perceptrón Multicapa para regresión (MLPRegressor)

### 2.1 Formulación e hipótesis

Un MLP con \(L\) capas ocultas define la función

\[
\mathbf{h}^{(0)} = \mathbf{x},
\]
\[
\mathbf{z}^{(\ell)} = \mathbf{W}^{(\ell)} \mathbf{h}^{(\ell-1)} + \mathbf{b}^{(\ell)}, \qquad
\mathbf{h}^{(\ell)} = \sigma\!\left(\mathbf{z}^{(\ell)}\right), \quad \ell = 1, \dots, L,
\]
\[
\hat{y} = \mathbf{w}^{(L+1)\top} \mathbf{h}^{(L)} + b^{(L+1)}.
\]

La salida es **lineal** (regresión). Las activaciones típicas son

\[
\sigma_{\text{ReLU}}(z) = \max(0, z), \qquad
\sigma_{\tanh}(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}.
\]

### 2.2 Criterio de optimización

Se minimiza el error cuadrático medio con regularización L2 (parámetro `alpha`):

\[
J(\Theta) = \frac{1}{2N} \sum_{i=1}^{N} \left( \hat{y}_i - y_i \right)^2
+ \frac{\alpha}{2} \sum_{\ell} \left\| \mathbf{W}^{(\ell)} \right\|_F^2.
\]

### 2.3 Algoritmo de aprendizaje

**Forward pass:** se calculan las activaciones capa por capa y la predicción \(\hat{y}\).

**Backpropagation:** para la capa de salida,

\[
\delta^{(L+1)} = \hat{y} - y,
\]

y para las capas ocultas,

\[
\delta^{(\ell)} = \left( \mathbf{W}^{(\ell+1)\top} \delta^{(\ell+1)} \right) \odot
\sigma'\!\left(\mathbf{z}^{(\ell)}\right).
\]

Los gradientes son

\[
\frac{\partial J}{\partial \mathbf{W}^{(\ell)}} =
\delta^{(\ell)} \left(\mathbf{h}^{(\ell-1)}\right)^{\top} + \alpha \mathbf{W}^{(\ell)},
\qquad
\frac{\partial J}{\partial \mathbf{b}^{(\ell)}} = \delta^{(\ell)}.
\]

**Optimizadores:**

- *SGD:* \(\theta \leftarrow \theta - \eta \nabla_\theta J\) (eventualmente con momento).
- *Adam:* mantiene medias móviles del gradiente y del gradiente al cuadrado y adapta la tasa por parámetro.

**Early stopping:** se reserva una fracción de los datos (`validation_fraction`) y se detiene el entrenamiento cuando la pérdida de validación no mejora durante `n_iter_no_change` épocas.

**Aproximación universal:** un MLP con una capa oculta suficientemente ancha y una activación no lineal puede aproximar cualquier función continua en un compacto (teorema de aproximación universal).

### 2.4 Supuestos y limitaciones

- Requiere **escalado** de las variables (media 0, desviación 1); de lo contrario el descenso de gradiente es inestable.
- No hay supuesto distribucional fuerte, pero sí necesita suficientes datos.
- Sensible a la inicialización, la arquitectura, la tasa de aprendizaje y la regularización.
- Alto costo computacional frente a modelos lineales.
- Difícil de interpretar; propenso a sobreajuste sin regularización ni early stopping.
- La pérdida MSE penaliza más los errores grandes; los outliers influyen mucho.

---

## 3. Análisis Discriminante Lineal y Cuadrático

### 3.1 Formulación e hipótesis

Modelo generativo gaussiano por clase:

\[
P(\mathbf{x} \mid y=k) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
= \frac{1}{(2\pi)^{d/2} |\boldsymbol{\Sigma}_k|^{1/2}}
\exp\!\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu}_k)^{\top}
\boldsymbol{\Sigma}_k^{-1} (\mathbf{x} - \boldsymbol{\mu}_k) \right).
\]

Aplicando Bayes y tomando logaritmos, la función discriminante es

\[
\delta_k(\mathbf{x}) = \log P(y=k) - \frac{1}{2} \log |\boldsymbol{\Sigma}_k|
- \frac{1}{2} (\mathbf{x} - \boldsymbol{\mu}_k)^{\top}
\boldsymbol{\Sigma}_k^{-1} (\mathbf{x} - \boldsymbol{\mu}_k) + C,
\]

y se predice \(\hat{y} = \arg\max_k \delta_k(\mathbf{x})\).

**LDA:** asume covarianza compartida \(\boldsymbol{\Sigma}_k = \boldsymbol{\Sigma}\) para todas las clases. Los términos cuadráticos se cancelan y la frontera es **lineal**:

\[
\delta_k^{\text{LDA}}(\mathbf{x}) =
\mathbf{x}^{\top} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k
- \frac{1}{2} \boldsymbol{\mu}_k^{\top} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k
+ \log P(y=k).
\]

**QDA:** permite \(\boldsymbol{\Sigma}_k\) distinta por clase; la frontera es **cuadrática** porque aparece el término
\(-\frac{1}{2}\mathbf{x}^{\top}\boldsymbol{\Sigma}_k^{-1}\mathbf{x}\).

### 3.2 Criterio de optimización

Máxima verosimilitud gaussiana. Las medias y covarianzas se estiman como

\[
\hat{\boldsymbol{\mu}}_k = \frac{1}{N_k} \sum_{i:\, y_i=k} \mathbf{x}_i,
\qquad
\hat{\boldsymbol{\Sigma}}_k = \frac{1}{N_k - 1} \sum_{i:\, y_i=k}
(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_k)(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_k)^{\top}.
\]

Los priors son \(\hat{P}(y=k) = N_k / N\). Para LDA la covarianza compartida es la media ponderada de las covarianzas por clase.

**Regularización:**

- LDA con `shrinkage` (con `solver="lsqr"` o `"eigen"`):
  \[
  \hat{\boldsymbol{\Sigma}} \leftarrow (1-\gamma)\hat{\boldsymbol{\Sigma}} + \gamma \nu \mathbf{I},
  \]
  donde \(\gamma \in [0,1]\) y \(\nu\) es la varianza media de las variables. `shrinkage="auto"` usa el lema de Ledoit-Wolf.
- QDA con `reg_param` \(r\):
  \[
  \hat{\boldsymbol{\Sigma}}_k \leftarrow (1-r)\hat{\boldsymbol{\Sigma}}_k + r \mathbf{I}.
  \]

### 3.3 Algoritmo de aprendizaje

1. Calcular medias y priors por clase.
2. Calcular la covarianza compartida (LDA) o una por clase (QDA).
3. Aplicar regularización si corresponde.
4. Evaluar \(\delta_k(\mathbf{x})\) y elegir el máximo.

Complejidad: LDA \(O(Nd^2 + d^3)\); QDA \(O(Nd^2 + Kd^3)\). La inferencia es \(O(Kd^2)\).

**LDA como proyección discriminante:** LDA busca una proyección \(\mathbf{W}\) que maximice la razón de dispersión entre clases e intra clases:

\[
J(\mathbf{W}) = \frac{\left| \mathbf{W}^{\top} \mathbf{S}_B \mathbf{W} \right|}
{\left| \mathbf{W}^{\top} \mathbf{S}_W \mathbf{W} \right|},
\]

donde

\[
\mathbf{S}_B = \sum_{k=1}^{K} N_k (\boldsymbol{\mu}_k - \boldsymbol{\mu})(\boldsymbol{\mu}_k - \boldsymbol{\mu})^{\top},
\qquad
\mathbf{S}_W = \sum_{k=1}^{K} \sum_{i:\, y_i=k}
(\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^{\top}.
\]

La solución se obtiene de los vectores propios generalizados de \(\mathbf{S}_W^{-1}\mathbf{S}_B\). El número máximo de componentes discriminantes es \(\min(K-1, d)\).

### 3.4 Supuestos y limitaciones

- **Gaussianidad** de las variables dentro de cada clase.
- LDA: covarianzas iguales; si no se cumple, la frontera lineal es subóptima.
- QDA: más flexible, pero estima \(O(Kd^2)\) parámetros; requiere más datos y es propenso a sobreajuste con \(d\) grande o clases pequeñas.
- Sensible a outliers (las medias y covarianzas no son robustas).
- QDA puede fallar si una covarianza no es de rango completo; se corrige con `reg_param`.
- LDA es más estable (menor varianza) cuando el supuesto de covarianza compartida es razonable; QDA reduce el sesgo pero aumenta la varianza.
- Muy interpretable: medias, covarianzas, priors y proyecciones.

---

## 4. Tabla comparativa

| Criterio | Naive Bayes (Multinomial/Complement) | MLPRegressor | LDA / QDA |
|---|---|---|---|
| **Supuesto principal** | Independencia condicional de los términos | Composición de transformaciones no lineales; sin supuesto distribucional fuerte | Gaussianidad por clase; LDA comparte covarianza, QDA no |
| **Criterio de optimización** | Máxima verosimilitud multinomial con suavizado \(\alpha\) (Complement: verosimilitud del complemento) | Minimización de MSE + regularización L2 | Máxima verosimilitud gaussiana (medias, priors y covarianzas) |
| **Tipo de frontera / función** | Lineal en log-conteos; salida probabilística | No lineal y arbitrariamente compleja; salida continua | LDA: lineal; QDA: cuadrática |
| **Complejidad de entrenamiento** | \(O(Nd)\) | \(O(N \cdot \text{parámetros} \cdot \text{épocas})\) | LDA \(O(Nd^2 + d^3)\); QDA \(O(Nd^2 + Kd^3)\) |
| **Complejidad de inferencia** | \(O(d)\) por documento | \(O(\text{parámetros})\) por muestra | \(O(Kd^2)\) por muestra |
| **Sensibilidad a outliers** | Media (los conteos extremos afectan \(\theta\)) | Alta (MSE) | Alta (medias y covarianzas) |
| **Interpretabilidad** | Alta (palabras por clase, priors) | Baja (caja negra) | Alta (medias, covarianzas, proyecciones) |
| **Datos necesarios** | Pocos; funciona bien en texto | Muchos | LDA moderados; QDA más datos por clase |
| **Ventaja típica** | Velocidad y baseline de texto; ComplementNB en desbalance | Captura relaciones no lineales | Eficiencia, interpretabilidad y buen desempeño con supuestos válidos |
| **Limitación típica** | Supuesto de independencia irreal | Costo, ajuste de hiperparámetros y escalado obligatorio | Supuesto gaussiano; QDA inestable con \(d\) grande |

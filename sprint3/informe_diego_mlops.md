# Versionado, contenedor y pruebas automáticas (MLOps)

**Responsable:** Huamantalla Ríos, Diego Miguel · **Sprint 3**

## Objetivo
Dejar el sistema reproducible: que cualquier integrante pueda clonar el repositorio, recuperar los datos y modelos exactos del Sprint 2, levantar la web en un contenedor y comprobar con pruebas automáticas que el recomendador sigue funcionando y respetando las reglas.

## Versionado de código, datos y modelos
El código vive en Git (`main`). Los archivos pesados no caben en Git, así que se versionan con **DVC**: en Git queda solo un archivo de texto pequeño (`.dvc`) con la huella del contenido, y los archivos reales se guardan en un almacenamiento remoto. Se versionan dos carpetas: `sprint2/02_DATASET` (instancias, embeddings, estilos, perfiles sintéticos y el archivo de recortes de 1,7 GB) y `sprint2/05_CODIGO/models` (`compat_head.pt`, `taste_head.pt`). Cada commit de Git apunta a una versión exacta de datos y modelos; volver a una versión anterior es `git checkout` y `dvc checkout`. Las dependencias están en `sprint3/requirements.txt`. *(completar: nombre del remoto DVC y tamaño total subido)*

## Contenedor
El `Dockerfile` (en la raíz del repo) parte de Python 3.11, instala PyTorch en versión CPU y las dependencias, copia solo el código de `sprint3/` y levanta Streamlit en el puerto 8501 con `STYLEMATE_BACKEND=real`, de modo que si faltan datos o modelos la aplicación falla a la vista en lugar de caer en silencio al mock. Los datos, los modelos y `rules.py` no se incluyen en la imagen: `docker-compose.yml` los monta como volúmenes de solo lectura. Así la imagen es pequeña, se construye en minutos y los datos pesados siguen controlados por DVC. El contenedor incluye una comprobación de salud sobre `/_stcore/health`.

## Pruebas automáticas e integración continua
Las pruebas de Marcela (`test_recomendador.py`) y de Andre (`test_interfaz.py`) verifican que `recomendar()` devuelve resultados, que el JSON respeta el contrato, que la salida está ordenada por puntaje y que no hay reglas violadas con el filtro activo. Se añadió `test_artefactos.py`, que comprueba que datos, modelos, `rules.py` y recortes están presentes: en modo estricto (`STYLEMATE_EXIGIR_ARTEFACTOS=1`) falla si falta alguno.

El flujo de **GitHub Actions** (`.github/workflows/tests.yml`) se ejecuta en cada `push` a `main` y en cada `pull request`: instala las dependencias, corre `pytest`, construye la imagen y hace una prueba de humo del contenedor en modo mock. Las pruebas que necesitan los datos pesados se omiten en el CI (no están en GitHub) y se ejecutan en la prueba final del contenedor, con los datos recuperados por DVC. *(completar: resultado de `pytest -q` y captura de la ejecución verde en GitHub Actions)*

## Limitaciones
- El CI no ejecuta el recomendador real porque los datos no están en GitHub; esa verificación depende de la prueba final con datos.
- Sin `rules.py` del Sprint 1 el sistema usa reglas simplificadas; `test_artefactos.py` lo detecta en modo estricto.
- El almacenamiento remoto de DVC depende de una cuenta externa y de un token por integrante.

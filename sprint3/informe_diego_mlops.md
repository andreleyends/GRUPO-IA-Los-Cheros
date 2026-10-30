# Versionado, contenedor y pruebas automáticas (MLOps)

**Responsable:** Huamantalla Ríos, Diego Miguel · **Sprint 3**

## Objetivo
Dejar el sistema reproducible: que cualquier integrante pueda clonar el repositorio, recuperar los datos y modelos exactos del Sprint 2, levantar la web en un contenedor y comprobar con pruebas automáticas que el recomendador sigue funcionando y respetando las reglas.

## Versionado de código, datos y modelos
Git guarda el código, las reglas del Sprint 1 (`rules.py` y su léxico), los datos livianos del Sprint 2 y los modelos entrenados (`compat_head.pt`, `taste_head.pt`, `garmentnet.pt`), de modo que un `git clone` ya trae todo lo necesario para el recomendador. Lo único que no cabe en Git es el archivo de recortes de las prendas (`recortes_144.npy`, 1,7 GB); ese se versiona con **DVC**: en Git queda solo un archivo de texto pequeño (`recortes_144.npy.dvc`) con la huella del contenido, y el archivo real se guarda en un almacenamiento remoto compatible con S3 (DagsHub, repositorio `stylemate-data`). Cada commit apunta a una versión exacta del archivo; recuperarlo es `dvc pull` y volver a una versión anterior es `git checkout` y `dvc checkout`. Las credenciales del remoto se guardan en un archivo local que Git ignora (`.dvc/config.local`), por lo que no están en el repositorio. Las dependencias están en `sprint3/requirements.txt`.

## Contenedor
El `Dockerfile` (en la raíz del repo) parte de Python 3.11, instala PyTorch en versión CPU y las dependencias, copia solo el código de `sprint3/` y levanta Streamlit en el puerto 8501 con `STYLEMATE_BACKEND=real`, de modo que si faltan datos o modelos la aplicación falla a la vista en lugar de caer en silencio al mock. Los datos, los modelos y `rules.py` no se incluyen en la imagen: `docker-compose.yml` los monta como volúmenes de solo lectura. Así la imagen es pequeña, se construye en minutos y los datos pesados siguen controlados por DVC. El contenedor incluye una comprobación de salud sobre `/_stcore/health`.

## Pruebas automáticas e integración continua
Las pruebas de Marcela (`test_recomendador.py`) y de Andre (`test_interfaz.py`) verifican que `recomendar()` devuelve resultados, que el JSON respeta el contrato, que la salida está ordenada por puntaje y que no hay reglas violadas con el filtro activo. Se ajustaron para elegir una prenda base que tenga combinaciones posibles, porque con las reglas del Sprint 1 algunas prendas (por ejemplo, un vestido que ya trae dos estampados) quedan sin candidatas. Se añadió `test_artefactos.py`, que comprueba que datos, modelos, `rules.py` y recortes están presentes: en modo estricto (`STYLEMATE_EXIGIR_ARTEFACTOS=1`) falla si falta alguno.

El flujo de **GitHub Actions** (`.github/workflows/tests.yml`) se ejecuta en cada `push` a `main` y en cada `pull request`: instala las dependencias, corre `pytest`, construye la imagen y hace una prueba de humo del contenedor en modo mock. Como los datos y modelos están en Git, el CI ejecuta también el recomendador real; solo se omite la comprobación de los recortes, que viven en DVC. En la máquina local, `pytest -q` dio 28 pruebas pasadas y 1 omitida (los recortes). *(completar: resultado de las pruebas dentro del contenedor y captura de la ejecución verde en GitHub Actions)*

## Limitaciones
- El CI no tiene los recortes (están en DVC), por lo que no verifica la parte de imágenes; esa verificación se hace en la prueba final con el contenedor.
- Datos y modelos livianos viven en Git; si crecen, habrá que pasarlos a DVC.
- El almacenamiento remoto de DVC depende de una cuenta externa y de un token por integrante.

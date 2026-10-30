# Mensaje para Joshua 2 · Qué es cada archivo y cómo trabajan juntos · Sprint 3

Joshua, este documento reúne **todos** los archivos del Sprint 3 (de Marcela, Andre y Diego) en un solo lugar: qué hace cada uno, cómo se conectan y qué te sirve para el informe. Complementa `MENSAJE_PARA_JOSHUA.md` (el de Andre), que sigue siendo válido para la parte de la interfaz.

## 1. Cómo funciona el sistema, en una línea
La CNN (GarmentNet) ya convirtió cada prenda del catálogo en un vector en el Sprint 2 (`embeddings.npy`). Al pulsar **Recomendar**, el sistema calcula la compatibilidad (MLP de compatibilidad), la ajusta al gusto del perfil (MLP de gusto), descarta o penaliza lo que rompe las reglas de estilo y ordena. La interfaz web solo muestra el resultado, y todo corre dentro de un contenedor Docker.

**Cadena de llamadas:**
`app.py` (pantalla) → `backend.py` (elige real o mock) → `recomendador.py` (el real) **o** `mock_recomendador.py` (datos inventados con el mismo JSON).

`recomendador.py` toma sus insumos de estos sitios del repo:
| Insumo | Dónde está | Cómo llega |
|---|---|---|
| Datos (instancias, embeddings, estilos, perfiles, pares, preferencias) | `sprint2/02_DATASET/` | Git |
| Modelos (`compat_head.pt`, `taste_head.pt`) | `sprint2/05_CODIGO/models/` | Git |
| Reglas de estilo y su léxico | `sprint1/E4_reglas_gusto/` (`rules.py`, `lexico_moda.json`) | Git |
| Recortes de las prendas (`recortes_144.npy`, 1,7 GB) | `sprint2/02_DATASET/` | **DVC** (`recortes_144.npy.dvc` en Git; el archivo real en DagsHub) |

## 2. Mapa de archivos

### Carpeta `sprint3/`
| Archivo | Responsable | Para qué sirve |
|---|---|---|
| `recomendador.py` | Marcela | `recomendar(perfil, prenda_base, top_k)`: compatibilidad → gusto → reglas → ranking, salida JSON. Con `--verificar` comprueba que reproduce los resultados del Sprint 2. |
| `test_recomendador.py` | Marcela | Pruebas del pipeline (formato JSON, orden por puntaje, sin reglas violadas, sin categorías redundantes). |
| `informe_marcela_6.1.md` | Marcela | Informe de 1 página: reglas y ranking. |
| `app.py` | Andre | Interfaz Streamlit. Solo habla con `backend.py`. |
| `backend.py` | Andre | Elige el recomendador real o el mock según `STYLEMATE_BACKEND`. |
| `mock_recomendador.py` | Andre | Mock con el mismo JSON (datos inventados). Solo para desarrollar y probar la interfaz. |
| `test_interfaz.py` | Andre | Pruebas del contrato JSON y de la interfaz (sobre el mock y, si hay datos, sobre el real). |
| `informe_andre_interfaz.md` | Andre | Informe de 1 página: interfaz y capturas. |
| `capturas/` | Andre | Capturas de la app. Las `MOCK_*` son solo de prueba de diseño y **no van en el informe**; las del informe son `real_*.png`. |
| `MENSAJE_PARA_JOSHUA.md` | Andre | Qué usar de la parte de la interfaz. |
| `README.md` | Andre | Tabla de archivos y cómo ejecutar. |
| `requirements.txt` | Andre / Diego | Dependencias del sprint (`pip install -r sprint3/requirements.txt`). Ojo: el `requirements.txt` de la raíz del repo es de otro proyecto. |
| `test_artefactos.py` | Diego | Comprueba que datos, modelos, `rules.py` y recortes estén presentes. Con `STYLEMATE_EXIGIR_ARTEFACTOS=1` falla si falta algo; sin la variable, se omite. |
| `informe_diego_mlops.md` | Diego | Informe de 1 página: versionado, contenedor, pruebas y CI. |
| `plan_monitoreo_reentrenamiento.md` | Diego | Plan de monitoreo y reentrenamiento (1 página). |
| `MENSAJE_PARA_JOSHUA_2.md` | Diego | Este documento. |

### Raíz del repo (MLOps, de Diego)
| Archivo | Para qué sirve |
|---|---|
| `Dockerfile` | Imagen de la web: Python 3.11, torch CPU, dependencias y el código de `sprint3/`. Puerto 8501. |
| `docker-compose.yml` | Levanta la web con el recomendador **real** y monta datos, modelos y reglas como carpetas de solo lectura. |
| `.dockerignore` | Evita copiar a la imagen lo pesado o innecesario (`.git`, `.venv`, `.npy`, `.pt`, etc.). |
| `.github/workflows/tests.yml` | GitHub Actions: corre `pytest`, construye la imagen y hace una prueba de humo del contenedor en cada `push` y `pull request`. |
| `.dvc/config`, `.dvc/.gitignore`, `.dvcignore` | Configuración de DVC (solo la URL del remoto; **sin credenciales**). |
| `sprint2/02_DATASET/recortes_144.npy.dvc` | Puntero a los recortes. El archivo real se baja con `dvc pull`. |

### Del Sprint 1 y 2 que usa el sistema
| Archivo | Para qué sirve |
|---|---|
| `sprint1/E4_reglas_gusto/rules.py` y `lexico_moda.json` | Motor de reglas del Sprint 1 y su léxico de 40 dimensiones. Se necesitan los dos juntos. |
| `sprint2/02_DATASET/*` y `sprint2/05_CODIGO/models/*` | Datos y modelos entrenados del Sprint 2. |

## 3. Cómo se ejecuta
Desde la raíz del repo:
```bash
# A) Todo en contenedor (la demo)
docker compose up --build            # http://localhost:8501 · aviso verde «Modo real»
docker compose down                  # apagar

# B) Pruebas dentro del contenedor, en modo estricto (nada puede faltar)
docker compose run --rm -w /app/sprint3 -e STYLEMATE_EXIGIR_ARTEFACTOS=1 stylemate pytest -q -rs

# C) Sin Docker
pip install -r sprint3/requirements.txt
cd sprint3
pytest -q -rs
STYLEMATE_BACKEND=real streamlit run app.py     # Windows PowerShell: $env:STYLEMATE_BACKEND="real"; streamlit run app.py
```
Antes de la demo, si no tienes `recortes_144.npy` (1,7 GB), se baja con `dvc pull` (necesita acceso al remoto en DagsHub: usuario colaborador de `stylemate-data` y token propio) o se copia a `sprint2/02_DATASET/`. Sin ese archivo la web funciona, pero muestra siluetas rotuladas «sin recorte» en lugar de fotos.

La web muestra siempre un aviso arriba: **verde** («Modo real») o **amarillo** («MODO DEMO (mock)»). **No tomen capturas del informe con el aviso amarillo.**

## 4. Estado de las pruebas
- Local: `pytest -q -rs` → **28 pasadas, 1 omitida** (los recortes, que no están en Git).
- Contenedor, modo estricto: **29 pasadas, 0 omitidas, 0 fallos**.
- GitHub Actions: las ejecuciones de los últimos commits salieron **en verde** (`pytest` y `docker`).

## 5. Qué usar para el informe del Sprint 3
Una página de cada uno, listas para pegar:
1. `informe_marcela_6.1.md` — reglas y ranking.
2. `informe_andre_interfaz.md` — interfaz y capturas.
3. `informe_diego_mlops.md` — versionado, contenedor y CI. Además, `plan_monitoreo_reentrenamiento.md`.

Pendiente: las capturas reales `capturas/real_1_inicio.png`, `real_2_looks.png` y `real_3_reglas.png` (Andre), y los datos entre paréntesis en los informes de Andre y de Diego (resultado de `pytest` y capturas), que cada uno completa. Para esas capturas, usen como prenda base un suéter o un reloj, no el vestido que se explica en el punto 6.

## 6. Cosas que conviene saber (para el Daily y la Review)
- **Reglas reales activadas.** `recomendador.py` ya usa `rules.py` del Sprint 1 (6 reglas: shorts con frío, deportivo en oficina, denim formal, doble estampado, límite de colores vivos, calzado abierto con lluvia). Antes se usaba una versión simplificada.
- **Algunas prendas base se quedan sin looks.** Con la regla de «doble estampado», un vestido de prueba que ya trae dos señales de estampado no tiene ninguna candidata válida, y la web muestra «No hay combinaciones que cumplan las reglas». Es la lógica del Sprint 2, no un error del sistema; con un suéter o un reloj casi no se descarta nada. Si el equipo quiere que esa regla cuente el estampado por prenda y no promediado, es una decisión de Marcela y cambiaría los números del Sprint 2.
- **Ajustes que hizo Diego en archivos de otros** (para que todos lo sepan al hacer `git pull`):
  - `recomendador.py`: línea 208 (compatibilidad con pandas 3) y línea 272 (el motivo de cada regla ahora sale como nombre corto y no como diccionario).
  - `app.py`: se agregaron al diccionario `TIPOS_REGLA` los nombres de las reglas del Sprint 1.
  - `test_recomendador.py` y `test_interfaz.py`: ahora eligen una prenda base que sí tenga combinaciones válidas.
- **Qué no verifica el CI.** Los recortes están en DVC y no en GitHub, así que el CI no prueba la parte de imágenes; eso se comprueba con el contenedor.

## 7. Lista de verificación antes de presentar (prueba final desde cero)
- [ ] `git clone` limpio + `dvc pull` baja `recortes_144.npy`
- [ ] `docker compose up --build` muestra el aviso verde «Modo real» y fotos reales
- [ ] Elegir un perfil `test_*` y un suéter o reloj, pulsar **Recomendar**: salen looks con recorte
- [ ] Oficina + frío: salen descartadas por reglas y ningún look viola reglas
- [ ] El comando B del punto 3 termina sin `failed` ni `skipped`
- [ ] GitHub Actions en verde
- [ ] Los tres informes (1 página cada uno) y el plan de monitoreo entregados

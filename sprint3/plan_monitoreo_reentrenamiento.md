# Plan de monitoreo y reentrenamiento

**Responsable:** Huamantalla Ríos, Diego Miguel · **Sprint 3**

## Qué se monitorea y qué se hace si falla
Línea base del Sprint 2: violación de reglas en el top-10 de 0,0 %, NDCG@10 de 0,961 y puntaje de gusto con AUC alto sobre preferencias sintéticas. Los umbrales de la tabla son **propuestos por el equipo** y pueden ajustarse al tener datos de uso.

| Señal | Cómo se mide | Umbral propuesto | Acción |
|---|---|---|---|
| Reglas violadas | `pytest` (CI) en cada cambio | Cualquier violación | Se bloquea la integración; se revisa `rules.py` o el filtro |
| Reproducibilidad de modelos | `python recomendador.py --verificar` tras cada `dvc pull` o cambio de modelo | Error de compatibilidad ≥ 1e-3 o AUC de gusto ≤ 0,95 | Se descarta la versión de modelo y se vuelve a la anterior |
| Calidad del ranking | NDCG@10 sobre el conjunto de prueba | Caída > 0,02 respecto a 0,961 | Se reentrena y se compara |
| Disponibilidad | Healthcheck del contenedor (`/_stcore/health`) | 3 fallos seguidos | Se reinicia el contenedor y se revisan los registros |
| Errores en la interfaz | Mensajes de error de `recomendar()` (prenda o perfil inexistente, archivos faltantes) | Cualquier error repetido | Se corrige y se agrega una prueba |
| Uso | Distribución de ocasión, clima y categoría base elegidos | Categorías o contextos sin cobertura en los resultados | Se amplía el catálogo o se ajustan las reglas |
| Satisfacción | Estudio con usuarios (tarea 3.15) y, si se agrega, botón «me gusta» | Aceptación menor a la de los baselines | Se reentrena el gusto con preferencias reales |

## Cuándo y cómo se reentrena
**Disparadores:** (1) llegan preferencias reales de usuarios; (2) cambia el catálogo (nuevas prendas); (3) alguna señal de la tabla supera su umbral; (4) cambian las reglas de estilo.

**Alcance según el disparador:** preferencias reales → solo `TasteMLP`; catálogo nuevo → recalcular embeddings y estilos y volver a entrenar `CompatMLP` y `TasteMLP`; reglas nuevas → solo `rules.py` y sus pruebas. `GarmentNet` no se reentrena salvo un cambio de dominio grande (por ejemplo, fotos propias de usuarios).

**Procedimiento:**
1. Crear una rama y recuperar los recortes con `dvc pull`.
2. Entrenar con los scripts del Sprint 2 (`4_2_entrenar_compat.py`, `4_3_entrenar_taste.py`) con semilla fija.
3. Evaluar con `--verificar`, NDCG@10 y tasa de violación, y comparar con el modelo vigente (prueba de Wilcoxon, como en la tarea 3.13).
4. Si mejora: commit de los modelos nuevos (los archivos pesados, con `dvc add` y `dvc push`), etiqueta de versión (`modelo-v2`) y reconstrucción de la imagen.
5. Si empeora: se conserva la versión vigente. **Retroceso:** `git checkout <etiqueta>` y `dvc checkout`.

**Frecuencia:** revisión de las señales al cierre de cada sprint y, en operación, una vez al mes o al acumular un lote nuevo de preferencias reales.

## Limitaciones
- Hoy no hay usuarios reales: el gusto se entrenó con preferencias sintéticas, por lo que la señal de satisfacción solo existirá cuando haya estudio de usuarios o retroalimentación en la interfaz.
- La aplicación no registra consultas todavía; las señales de uso y errores requieren agregar ese registro.

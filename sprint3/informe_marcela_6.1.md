# Integración de modelos en el pipeline de recomendación (reglas y ranking)

**Responsable:** Marcela Leonor Ordoñez Paniura · **Actividad:** 6.1 · Sprint 3

## Objetivo
Integrar los modelos entrenados en el Sprint 2 en un único punto de entrada, `recomendar(perfil, prenda_base, top_k)`, que recibe una prenda del catálogo y el perfil de estilo del usuario y devuelve las combinaciones recomendadas en formato JSON. La interfaz web consume esta función.

## Flujo del pipeline
1. **Candidatas:** a partir de la prenda base se toman las prendas principales del catálogo de otra categoría, excluyendo los pares redundantes (por ejemplo, vestido + pantalón).
2. **Compatibilidad:** el MLP de compatibilidad puntúa cada par con los embeddings de GarmentNet, el vector de estilo (40D) y la categoría.
3. **Personalización:** el MLP de gusto ajusta el puntaje según el perfil del usuario (vector de afinidad de 40 dimensiones).
4. **Reglas de estilo:** se descartan las combinaciones que incumplen las cinco reglas definidas en el Sprint 1: shorts con frío, deportivo en oficina, denim en contexto formal, doble estampado y exceso de colores vivos. Las reglas se aplican con el motor del Sprint 1 (`rules.py`) y dependen de la ocasión y el clima que elija el usuario.
5. **Ranking:** se ordena por puntaje final y se muestra la mejor opción por categoría para dar variedad.

## Salida
Cada recomendación incluye la prenda, su categoría y color, el puntaje de compatibilidad, el puntaje de gusto, la penalización y el puntaje final. Esto permite explicar en la interfaz por qué se recomendó cada prenda.

## Relación con los resultados del Sprint 2
En el Sprint 2 las reglas redujeron la violación en el top-10 de 10.0 % a 0.0 %, con una caída mínima de NDCG@10 (0.964 a 0.961). El pipeline aplica las reglas como filtro previo al ranking, por lo que la salida del sistema no contiene combinaciones que las incumplan.

## Verificación
Las pruebas automáticas (`test_recomendador.py`) comprueban que la salida respeta el formato JSON, que está ordenada por puntaje y que no hay reglas violadas. Estas pruebas se ejecutan en el pipeline de CI/CD.

## Limitaciones
- El MLP de gusto se entrenó con preferencias sintéticas. Su desempeño mide consistencia con el diseño, no satisfacción de usuarios reales.
- El catálogo proviene de recortes de Fashionpedia, que muestran prendas puestas sobre personas.
- Las reglas son un conjunto fijo y se calibraron con criterio del equipo; no se validaron con usuarios.

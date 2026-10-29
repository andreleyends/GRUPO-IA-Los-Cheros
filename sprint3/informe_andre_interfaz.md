# Interfaz web de StyleMate y capturas de funcionamiento

**Responsable:** Andre Ismael Latorre Campos · **Sprint 3**

## Objetivo
Ofrecer una interfaz web en Streamlit que permita probar el sistema sin escribir código: el usuario elige un perfil de estilo y una prenda base, pulsa «Recomendar» y obtiene los looks sugeridos con la imagen de cada prenda. La interfaz consume la función `recomendar(perfil, prenda_base, top_k)` del pipeline de integración (informe 6.1).

## Diseño de la interfaz
La barra lateral reúne los controles y el área central muestra el resultado.
1. **Perfil y contexto:** perfil de estilo (con su arquetipo), ocasión (diario, oficina, deporte, noche, viaje) y clima (frío, entretiempo, calor). Si no se elige ocasión o clima, se usan los del perfil.
2. **Prenda base:** filtro por categoría y lista de prendas del catálogo, con vista previa del recorte.
3. **Opciones:** cantidad de looks (1 a 10), una prenda por categoría y ocultar las combinaciones que rompen reglas.
4. **Resultado:** la prenda base, el contexto aplicado, el número de combinaciones descartadas por reglas y una tarjeta por look con su recorte, categoría, color, puntaje de compatibilidad, puntaje de gusto, puntaje final y estado frente a las reglas. Un panel desplegable muestra el JSON devuelto.

## Integración con el pipeline
La interfaz solo se comunica con el módulo `backend.py`, que usa el recomendador real y, si faltan los modelos o los datos, un mock con el mismo formato JSON. Esto permitió desarrollar la interfaz mientras se terminaba el pipeline y reemplazar el mock sin modificar la interfaz. Un aviso en la parte superior indica siempre qué modo está activo (real o demostración), para evitar presentar resultados simulados como reales. Los recortes de 144×144 píxeles se obtienen con `recorte(pos)`; si no hay imagen disponible, se muestra un marcador gris y la recomendación continúa. Los errores del pipeline se muestran como mensaje en pantalla, sin cerrar la aplicación.

## Verificación
`test_interfaz.py` comprueba que la salida respeta el contrato JSON (claves, orden por puntaje final, numeración, contexto elegido, error ante una prenda inexistente y ausencia de violaciones con el filtro activo) y que la interfaz carga, recomienda y avisa cuando no hay combinaciones válidas. Las pruebas de contrato se ejecutan tanto sobre el mock como sobre `recomendar()` real cuando hay datos y modelos disponibles. *(completar con el resultado de `pytest -q` en la máquina con los datos reales)*

## Capturas
- **Figura 1.** Pantalla inicial con los controles. *(insertar `capturas/real_1_inicio.png`)*
- **Figura 2.** Looks recomendados para un perfil y una prenda base, con sus recortes y puntajes. *(insertar `capturas/real_2_looks.png`)*
- **Figura 3.** Contexto «oficina» y «frío», con las combinaciones descartadas por reglas. *(insertar `capturas/real_3_reglas.png`)*

## Limitaciones
- El puntaje de gusto proviene de preferencias sintéticas; la interfaz muestra consistencia con el diseño, no satisfacción de usuarios reales.
- Los recortes son prendas puestas sobre personas (Fashionpedia), por lo que no tienen fondo uniforme.
- La interfaz no incluye autenticación, historial ni subida de fotos propias; el catálogo se limita al conjunto de prueba.

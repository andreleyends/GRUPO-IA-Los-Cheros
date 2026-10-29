# Entrega de Andre (CNN / interfaz) para Joshua · Sprint 3

Joshua, esto es lo que necesitas de mi parte para el documento del Sprint 3.

## Qué hice
Interfaz web en Streamlit que consume `recomendar()` de Marcela: se elige perfil, ocasión, clima y
prenda base, se pulsa **Recomendar** y se ven los looks con su recorte, puntajes de compatibilidad
y gusto, puntaje final y reglas que cumple o rompe. Mientras Marcela terminaba usé un mock con el
mismo JSON (`mock_recomendador.py`); `backend.py` cambia entre mock y real sin tocar la interfaz.

## Qué debes poner en el documento
1. **Una página:** `informe_andre_interfaz.md` (interfaz y capturas). Está listo para pegar.
2. **Capturas reales:** van en `capturas/real_*.png`. Las `MOCK_*.png` son solo de prueba de diseño,
   **no deben ir en el informe**.
3. **Evidencia de pruebas:** salida de `pytest -q` (ver README).

## Cómo correrlo (para la demo de mañana)
Ver `README.md` de esta carpeta. Resumen: `STYLEMATE_BACKEND=real streamlit run app.py` (puerto 8501 en el contenedor).

## Dependencias entre compañeros
- **Marcela:** falta subir `sprint1/E4_reglas_gusto/rules.py` al repo; sin él se usan reglas simplificadas.
- **Recortes:** `recortes_144.npy` (1,7 GB) no está en el repo; sin él la interfaz muestra siluetas rotuladas «sin recorte» en lugar de fotos.
- **Diego:** agregar `streamlit` al `requirements.txt`; el contenedor debe tener los datos y modelos
  (`embeddings.npy`, `compat_head.pt`, `taste_head.pt`, etc.) y exponer el puerto 8501.

## Para el Daily / Review
- Hecho: interfaz, mock, integración con `recomendar()`, pruebas de interfaz.
- Pendiente de mi parte: capturas con los datos reales y revisión final desde cero con el contenedor.

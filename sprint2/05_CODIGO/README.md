# StyleMate - Sprint 2

Sprint 2 del sistema de recomendacion de outfits **StyleMate**. Esta entrega usa el
**escenario B: 4.000 imagenes** de Fashionpedia y deja preparado el pipeline para
escalar al dataset completo en una entrega posterior.

## Objetivo

Construir y evaluar el pipeline de:

1. limpieza y muestreo del dataset;
2. pares de compatibilidad positivos, negativos y sinteticos;
3. particiones train/val/test a nivel de imagen;
4. CNN multi-tarea de categoria y atributos;
5. MLP de compatibilidad entre prendas;
6. MLP de personalizacion con perfiles sinteticos;
7. reglas de gusto y metricas de ablacion.

## Resultados principales

| Etapa | Metrica | Valor |
|---|---|---:|
| Dataset | Imagenes / instancias | 4.000 / 28.565 |
| Pares | Positivos / negativos | 24.861 / 49.462 |
| CNN | Accuracy top-1 / top-3 | 0,624 / 0,806 |
| CNN | Macro F1 / weighted F1 | 0,321 / 0,615 |
| Compatibilidad | AUC / AP / F1 | 0,866 / 0,726 / 0,705 |
| Taste | NDCG@10 sin reglas | 0,964 |
| Taste + reglas | NDCG@10 / violacion top-10 | 0,961 / 0,0 % |
| Ablacion | AUC completo vs sin CNN | 0,867 vs 0,825 |

Las preferencias de taste son **sinteticas**; miden consistencia con el diseno, no
satisfaccion de usuarios reales.

## Estructura

```
sprint2/
  config.py                 constantes, rutas y umbrales
  data/processed/           CSV, recortes, embeddings y pares
  models/                   checkpoints .pt
  reports/                  informes Markdown y figuras
  reports/metricas/         JSON/CSV de metricas
  docs/                     documentos Word finales
  scrum/                    planificacion, dailies y burndown
  src/                      pipeline ejecutable
  _generadores/              conversion Markdown -> DOCX
```

## Reproduccion

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

.\.venv\Scripts\python.exe -X utf8 src\3_1_limpiar_dataset.py
.\.venv\Scripts\python.exe -X utf8 src\3_3_splits.py
.\.venv\Scripts\python.exe -X utf8 src\3_2_generar_pares.py
.\.venv\Scripts\python.exe -X utf8 src\4_1_entrenar_cnn.py
.\.venv\Scripts\python.exe -X utf8 src\4_2_entrenar_compat.py
.\.venv\Scripts\python.exe -X utf8 src\4_3_entrenar_taste.py
.\.venv\Scripts\python.exe -X utf8 src\5_1_metricas.py
.\.venv\Scripts\python.exe -X utf8 src\5_2_ablacion.py
.\.venv\Scripts\python.exe -X utf8 _generadores\generar_docs.py
```

La ruta de Fashionpedia se configura en `config.py`. El pipeline asume que los
indices de Sprint 1 existen en `../sprint1/salidas/indices`.

## Entregables

- `scrum/S2.0_Sprint_Goal_Y_Sprint_Backlog.md`
- `reports/3.1_Dataset_limpio.md`
- `reports/3.2_Dataset_de_pares_etiquetados.md`
- `reports/3.3_Conjuntos_de_entrenamiento_validación_y_prueba.md`
- `reports/4.1_Modelo_CNN_entrenado.md`
- `reports/4.2_Modelo_MLP_de_compatibilidad.md`
- `reports/4.3_MLP_de_personalización_y_filtros_de_reglas.md`
- `reports/5.1_Reporte_de_métricas.md`
- `reports/5.2_Reporte_comparativo_y_de_ablación.md`
- `scrum/S2.1_Registro_breve_de_dailies.md`
- `scrum/S2.2_Burndown_y_Backlog_actualizados.md`
- `scrum/S2.3_Acta_de_Review.md`
- `scrum/S2.4_Acta_de_Retrospective.md`
- Versión Word de cada documento en `docs/`.

## Notas

- `S2.3 Sprint Review` y `S2.4 Retrospective` no se generan en esta entrega por
  decision del equipo.
- El dataset completo se procesara despues; el pipeline ya contempla los mismos
  pasos y umbrales.
- Los negativos aleatorios pueden ser combinaciones validas; el AUC mide separacion
  respecto al muestreo, no verdad absoluta.

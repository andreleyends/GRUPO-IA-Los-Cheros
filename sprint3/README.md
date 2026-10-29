# StyleMate · Sprint 3

Interfaz web (Andre) sobre el pipeline de recomendación (Marcela).

| Archivo | Responsable | Para qué sirve |
|---|---|---|
| `recomendador.py` | Marcela | `recomendar(perfil, prenda_base, top_k)`: compatibilidad → gusto → reglas → ranking, salida JSON. |
| `test_recomendador.py` | Marcela | Pruebas del pipeline. |
| `informe_marcela_6.1.md` | Marcela | Informe de 1 página del pipeline. |
| `app.py` | Andre | Interfaz Streamlit. Solo habla con `backend.py`. |
| `backend.py` | Andre | Elige recomendador real o mock. |
| `mock_recomendador.py` | Andre | Mock con el mismo JSON (datos inventados). |
| `test_interfaz.py` | Andre | Pruebas del contrato JSON y de la interfaz. |
| `informe_andre_interfaz.md` | Andre | Informe de 1 página: interfaz y capturas. |
| `capturas/` | Andre | Capturas de la app (`MOCK_*` = solo prueba de diseño). |
| `MENSAJE_PARA_JOSHUA.md` | Andre | Qué usar de esta carpeta en el documento del Sprint 3. |

## De dónde toma los datos

`recomendador.py` busca solo, dentro del repo, estos archivos del Sprint 2:

- Datos: `sprint2/02_DATASET/` (`instancias_split.csv`, `embeddings.npy`, `estilos_instancias.npy`, `perfiles_sinteticos.json`, `pares_scored.csv`, `preferencias_sinteticas.csv`).
- Modelos: `sprint2/05_CODIGO/models/` (`compat_head.pt`, `taste_head.pt`).
- Reglas: `sprint1/E4_reglas_gusto/rules.py` (**no está en el repo**; ver abajo).
- Recortes: `recortes_144.npy` (1,7 GB, **no está en el repo**; ver abajo).

Si algo está en otra carpeta: variables `STYLEMATE_DATA`, `STYLEMATE_MODELS`, `STYLEMATE_RULES`.

## Dos archivos que faltan en el repo

1. **`rules.py` del Sprint 1** (`sprint1/E4_reglas_gusto/rules.py`). Sin él, `recomendador.py` usa una versión simplificada de las 5 reglas y avisa con una advertencia. Copiarlo a `sprint1/E4_reglas_gusto/` desde la máquina de Marcela.
2. **`recortes_144.npy`** (está en `sprint2\data\processed` de la máquina del equipo). Copiarlo a `sprint2/02_DATASET/`; el `.gitignore` ya lo excluye para no subirlo. Sin él la interfaz funciona, pero en lugar de la foto muestra una silueta de la categoría en su color, rotulada «sin recorte».

## Cómo ejecutar (desde la raíz del repo)

```bash
pip install -r sprint3/requirements.txt
cd sprint3
python recomendador.py --verificar          # comprueba que reproduce los resultados del Sprint 2
STYLEMATE_BACKEND=real streamlit run app.py # exige el recomendador real (demo y capturas)
streamlit run app.py                        # auto: real si puede, mock si falta algo
STYLEMATE_BACKEND=mock streamlit run app.py # fuerza el mock
pytest -q                                   # pruebas de Marcela y de Andre
```
Windows (PowerShell): `$env:STYLEMATE_BACKEND="real"; streamlit run app.py`

La app muestra arriba un aviso verde «Modo real» o amarillo «MODO DEMO (mock)». **No tomar capturas del informe con el aviso amarillo.**

## Capturas reales (pendientes)

Con `STYLEMATE_BACKEND=real`: `capturas/real_1_inicio.png` (pantalla inicial), `real_2_looks.png` (perfil `test_*` + prenda base + Recomendar) y `real_3_reglas.png` (oficina + frío, con descartadas por reglas).

## Docker (Diego)

Puerto 8501. Comando: `streamlit run sprint3/app.py --server.address 0.0.0.0 --server.port 8501` con `STYLEMATE_BACKEND=real`, para que falle a la vista si faltan modelos o datos. Ojo: el `requirements.txt` de la raíz es de otro proyecto (Streamlit Cloud); el de este sprint es `sprint3/requirements.txt`.

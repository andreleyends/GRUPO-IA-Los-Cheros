# -*- coding: utf-8 -*-
"""
StyleMate · Sprint 3 · Interfaz web (Andre)

Ejecutar:   streamlit run app.py
Backend:    STYLEMATE_BACKEND=auto|real|mock  (ver backend.py)
"""
import json

import numpy as np
import streamlit as st

import backend as bk

st.set_page_config(page_title="StyleMate", page_icon="👗", layout="wide")

TIPOS_REGLA = {
    "shorts_con_frio": "Shorts con clima frío",
    "deportivo_en_oficina": "Prenda deportiva en la oficina",
    "denim_formal": "Denim en contexto formal",
    "doble_estampado": "Doble estampado",
    "exceso_colores_vivos": "Exceso de colores vivos",
}


def _placeholder(texto="sin imagen") -> np.ndarray:
    img = np.full((144, 144, 3), 236, dtype=np.uint8)
    img[60:84, 20:124] = 214
    return img


def _imagen(p: dict) -> np.ndarray:
    """Recorte de la prenda `p` (dict con pos, categoria, color)."""
    try:
        arr = bk.recorte(p["pos"], p["categoria"], p["color"])
    except Exception:  # noqa: BLE001
        arr = None
    return arr if arr is not None else _placeholder()


def _etiqueta_prenda(p: dict) -> str:
    return f"{p['categoria']} · {p['color'] or 'sin color'} · #{p['annotation_id']}"


@st.cache_data(show_spinner=False)
def _perfiles():
    return bk.listar_perfiles()


@st.cache_data(show_spinner=False)
def _categorias():
    return bk.categorias()


@st.cache_data(show_spinner=False)
def _prendas(categoria):
    return bk.listar_prendas(categoria or None, 24)


# ---------------------------------------------------------------- encabezado
st.title("👗 StyleMate")
st.caption("Recomendador de combinaciones de prendas · CNN → MLP de compatibilidad → "
           "MLP de gusto → reglas de estilo")

if bk.MODO == "real":
    st.success("Modo real: usando `recomendar()` con los modelos entrenados en el Sprint 2.")
else:
    st.warning("**MODO DEMO (mock)**: los puntajes y las imágenes son inventados para probar "
               "la interfaz; no vienen de los modelos. "
               f"Motivo: {bk.MOTIVO_MOCK}")

# ---------------------------------------------------------------- controles
with st.sidebar:
    st.header("1. Perfil y contexto")
    perfiles = sorted(_perfiles(), key=lambda p: (p["split"] != "test", p["perfil"]))
    por_id = {p["perfil"]: p for p in perfiles}
    perfil = st.selectbox(
        "Perfil de estilo", list(por_id),
        format_func=lambda k: f"{k} · {por_id[k]['arquetipo']}")
    ocasion = st.selectbox("Ocasión", ["(según el perfil)"] + bk.OCASIONES)
    clima = st.selectbox("Clima", ["(según el perfil)"] + bk.CLIMAS)

    st.header("2. Prenda base")
    cat = st.selectbox("Categoría", ["(todas)"] + _categorias())
    prendas = _prendas(None if cat == "(todas)" else cat)
    prenda = st.selectbox("Prenda", prendas, format_func=_etiqueta_prenda)
    st.image(_imagen(prenda), width=144, caption=_etiqueta_prenda(prenda))

    st.header("3. Opciones")
    top_k = st.slider("Cantidad de looks", 1, 10, 5)
    una = st.checkbox("Una prenda por categoría", value=True)
    solo = st.checkbox("Ocultar combinaciones que rompen reglas", value=True,
                       help="Si lo desmarcas, se muestran también las prendas que violan "
                            "alguna regla, con su penalización.")
    pedir = st.button("Recomendar", type="primary", width="stretch")

# ---------------------------------------------------------------- consulta
if pedir:
    try:
        with st.spinner("Calculando recomendaciones…"):
            st.session_state["salida"] = bk.recomendar(
                perfil, prenda["annotation_id"], top_k,
                ocasion=None if ocasion.startswith("(") else ocasion,
                clima=None if clima.startswith("(") else clima,
                una_por_categoria=una, solo_validas=solo)
            st.session_state["error"] = None
    except Exception as e:  # noqa: BLE001
        st.session_state["salida"] = None
        st.session_state["error"] = f"{type(e).__name__}: {e}"

if st.session_state.get("error"):
    st.error(f"No se pudo recomendar. {st.session_state['error']}")

salida = st.session_state.get("salida")
if not salida:
    st.info("Elige un perfil y una prenda base en la barra lateral y pulsa **Recomendar**.")
    st.stop()

# ---------------------------------------------------------------- resultado
base = salida["prenda_base"]
ctx = salida["contexto"]
cab1, cab2 = st.columns([1, 4])
with cab1:
    st.image(_imagen(base), width=160, caption="Prenda base")
with cab2:
    st.subheader(f"{base['categoria']} · {base['color'] or 'sin color'}")
    st.markdown(
        f"**Perfil:** {salida['perfil']} ({salida['arquetipo']})  \n"
        f"**Contexto:** {ctx['ocasion']} · {ctx['clima']}  \n"
        f"**Descartadas por reglas:** {salida['descartadas_por_reglas']}")

recs = salida["recomendaciones"]
st.divider()
if not recs:
    st.warning("No hay combinaciones que cumplan las reglas con este contexto. "
               "Prueba otra prenda base, otro clima u otra ocasión, o desmarca "
               "«Ocultar combinaciones que rompen reglas».")
    st.stop()

st.subheader(f"Looks recomendados ({len(recs)})")
cols = st.columns(min(len(recs), 5))
for i, r in enumerate(recs):
    with cols[i % len(cols)]:
        with st.container(border=True):
            st.markdown(f"**#{r['rank']} · {r['categoria']}**")
            st.image(_imagen(r), width="stretch")
            st.caption(f"Color: {r['color'] or 'sin color'} · #{r['annotation_id']}")
            st.progress(min(max(r["final"], 0.0), 1.0), text=f"Puntaje final {r['final']:.2f}")
            c1, c2 = st.columns(2)
            c1.metric("Compat.", f"{r['compat']:.2f}")
            c2.metric("Gusto", f"{r['taste']:.2f}")
            if r["reglas_violadas"]:
                st.error("⚠ " + ", ".join(TIPOS_REGLA.get(x, x) for x in r["reglas_violadas"])
                         + f" (−{r['penalizacion']:.1f})")
            else:
                st.success("Cumple las reglas")

with st.expander("Ver respuesta JSON de recomendar()"):
    st.code(json.dumps(salida, ensure_ascii=False, indent=2), language="json")

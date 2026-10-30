# -*- coding: utf-8 -*-
"""
StyleMate · Sprint 3 · 6.1 Pipeline de inferencia (reglas + ranking)
Responsable: Marcela

Flujo:  prenda base -> candidatas del catalogo -> CompatMLP (4.2)
        -> TasteMLP (4.3) -> motor de reglas de Sprint 1 -> ranking (JSON)

Usa las mismas clases, features y reglas del Sprint 2 (4_2, 4_3, estilo.py).
No necesita la CNN: los embeddings de cada prenda ya estan en embeddings.npy.

Ubicacion sugerida:  <repo>/sprint3/recomendador.py
Los datos (02_DATASET o data/processed), los modelos (compat_head.pt, taste_head.pt)
y sprint1/E4_reglas_gusto/rules.py se buscan solos dentro del repo.
Si estan en otra parte:  STYLEMATE_DATA, STYLEMATE_MODELS, STYLEMATE_RULES.

Uso rapido:
    python recomendador.py --verificar
    python recomendador.py --listar
    python recomendador.py --perfil test_0000 --prenda 123456 --top 5

Desde Python (interfaz web):
    from recomendador import recomendar, listar_prendas, listar_perfiles
    r = recomendar("test_0000", 123456, top_k=5, ocasion="oficina", clima="frio")
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download # <--- NUEVA IMPORTACIÓN

AQUI = Path(__file__).resolve().parent
SEMILLA = 42

# --- CONFIGURACIÓN PARA HUGGING FACE ---
REPO_DATASET = "DiegoMT2/stylemate-datos"
ARCHIVO_RECORTES = "recortes_144.npy"

# ----------------------------------------------------------------------------
# Constantes copiadas del Sprint 2 (config.py y 4_3_entrenar_taste.py)
# ----------------------------------------------------------------------------
EMBED_DIM = 128
PARES_REDUNDANTES = {frozenset(p) for p in [
    ("dress", "skirt"), ("dress", "pants"), ("dress", "shorts"), ("dress", "jumpsuit"),
    ("skirt", "pants"), ("skirt", "shorts"), ("pants", "shorts"),
    ("jumpsuit", "pants"), ("jumpsuit", "skirt"), ("jumpsuit", "shorts"),
]}
CONTEXTO = {  # contexto por arquetipo (4_3)
    "casual": {"clima": "entretiempo", "ocasion": "diario", "formalidad": "casual"},
    "formal": {"clima": "entretiempo", "ocasion": "oficina", "formalidad": "formal"},
    "deportivo": {"clima": "entretiempo", "ocasion": "deporte", "formalidad": "sport"},
    "elegante": {"clima": "entretiempo", "ocasion": "noche", "formalidad": "formal"},
    "bohemio": {"clima": "calor", "ocasion": "viaje", "formalidad": "casual"},
}
COLORES_VIVOS = {"rojo", "naranja", "amarillo", "verde", "morado", "rosa"}
TOKEN_CATEGORIA = {
    "shirt, blouse": "shirt", "top, t-shirt, sweatshirt": "top",
    "sweater": "sweater", "cardigan": "cardigan", "jacket": "jacket",
    "vest": "vest", "pants": "pants", "shorts": "shorts", "skirt": "skirt",
    "coat": "coat", "dress": "dress", "jumpsuit": "jumpsuit", "cape": "cape",
    "glasses": "glasses", "hat": "hat", "tie": "tie", "glove": "glove",
    "watch": "watch", "belt": "belt", "sock": "sock", "shoe": "shoe",
    "bag, wallet": "bag", "scarf": "scarf", "umbrella": "umbrella",
}
FORMALIDAD_POR_OCASION = {"oficina": "formal", "noche": "formal", "deporte": "sport"}
OCASIONES = ["diario", "oficina", "deporte", "noche", "viaje"]
CLIMAS = ["frio", "entretiempo", "calor"]


# ----------------------------------------------------------------------------
# Busqueda de archivos
# ----------------------------------------------------------------------------
def _localizar(nombre: str, env: str | None = None, obligatorio: bool = True):
    if env and os.environ.get(env):
        p = Path(os.environ[env])
        if p.is_file():
            return p
        if (p / nombre).exists():
            return p / nombre
    raiz = AQUI
    for _ in range(3):  # busca en esta carpeta y hacia arriba (repo)
        hallados = sorted(raiz.rglob(nombre), key=lambda x: len(x.parts))
        if hallados:
            return hallados[0]
        raiz = raiz.parent
    if obligatorio:
        raise FileNotFoundError(
            f"No encuentro '{nombre}'. Copialo junto a recomendador.py o define "
            f"la variable de entorno {env or 'STYLEMATE_DATA'}.")
    return None


# ----------------------------------------------------------------------------
# Modelos: identicos a 4_2_entrenar_compat.py y 4_3_entrenar_taste.py
# ----------------------------------------------------------------------------
class CompatMLP(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, cat_dim=16, style_dim=40,
                 n_cat=46, hidden=(256, 128, 64)):
        super().__init__()
        self.cat_emb = nn.Embedding(n_cat, cat_dim)
        inp = embed_dim * 3 + style_dim * 2 + cat_dim * 2
        capas, prev = [], inp
        for h in hidden:
            capas += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True),
                      nn.Dropout(0.2)]
            prev = h
        capas.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*capas)
        self.inp = inp

    def forward(self, ea, eb, sa, sb, ca, cb):
        x = torch.cat([ea, eb, torch.abs(ea - eb), sa, sb,
                       self.cat_emb(ca), self.cat_emb(cb)], dim=1)
        return self.mlp(x).squeeze(1)


class TasteMLP(nn.Module):
    def __init__(self, inp=161, hidden=(128, 64)):
        super().__init__()
        capas, prev = [], inp
        for h in hidden:
            capas += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True),
                      nn.Dropout(0.2)]
            prev = h
        capas.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*capas)

    def forward(self, x):
        return self.mlp(x).squeeze(1)


def _cargar_pt(nombre: str, modelo: nn.Module) -> nn.Module:
    ck = torch.load(_localizar(nombre, "STYLEMATE_MODELS"), map_location="cpu",
                    weights_only=False)
    modelo.load_state_dict(ck["state_dict"])
    return modelo.eval()


# ----------------------------------------------------------------------------
# Motor de reglas de Sprint 1 (rules.py) con respaldo simple
# ----------------------------------------------------------------------------
def _cargar_motor_reglas():
    ruta = _localizar("E4_reglas_gusto/rules.py", "STYLEMATE_RULES", obligatorio=False)
    if ruta is None:
        ruta = _localizar("rules.py", "STYLEMATE_RULES", obligatorio=False)
    if ruta is not None:
        try:
            spec = importlib.util.spec_from_file_location("rules_sprint1", ruta)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            lexico = mod.cargar_lexico()
            orden = mod.orden_dimensiones(lexico)
            return mod, {n: i for i, n in enumerate(orden)}
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"No pude cargar rules.py ({e}); uso reglas simplificadas.")
    else:
        warnings.warn("No encontre sprint1/E4_reglas_gusto/rules.py; uso reglas simplificadas.")
    return None, {}


def _reglas_simples(tokens: list[str], ctx: dict):
    """Respaldo si no esta rules.py. Mismas 5 reglas del Sprint 1, version basica."""
    v = []
    if ctx.get("clima") == "frio" and "shorts" in tokens:
        v.append("shorts_con_frio")
    if ctx.get("ocasion") == "oficina" and ("sport" in tokens or "shorts" in tokens):
        v.append("deportivo_en_oficina")
    if ctx.get("formalidad") == "formal" and ctx.get("material") == "denim":
        v.append("denim_formal")
    if sum(t in tokens for t in ("estampado", "floral", "rayas", "cuadros")) >= 2:
        v.append("doble_estampado")
    if ctx.get("colores_vivos", 0) >= 2:
        v.append("exceso_colores_vivos")
    return float(len(v)), v


# ----------------------------------------------------------------------------
# Sistema de recomendacion
# ----------------------------------------------------------------------------
class Recomendador:
    def __init__(self, split: str | None = "test"):
        torch.manual_seed(SEMILLA)
        self._cargar_datos(split)
        self.compat = _cargar_pt("compat_head.pt", CompatMLP(embed_dim=self.E.shape[1]))
        self.taste = _cargar_pt("taste_head.pt", TasteMLP())
        self.motor, self.DIM = _cargar_motor_reglas()

    # -- datos ---------------------------------------------------------------
    def _cargar_datos(self, split):
        df = pd.read_csv(_localizar("instancias_split.csv", "STYLEMATE_DATA"))
        df = df.sort_values("pos").reset_index(drop=True)
        E = np.load(_localizar("embeddings.npy", "STYLEMATE_DATA"))
        S = np.load(_localizar("estilos_instancias.npy", "STYLEMATE_DATA"))
        if len(E) != len(df) or len(S) != len(df):
            raise ValueError(f"Tamanos distintos: instancias={len(df)}, "
                             f"embeddings={len(E)}, estilos={len(S)}")
        self.df = df
        self.E = torch.tensor(E, dtype=torch.float32)   # fila i == instancia con pos i
        self.S = torch.tensor(S, dtype=torch.float32)
        self.cat_idx = torch.tensor(df["category_id"].astype(int).values, dtype=torch.long)
        self.nombres = df["category_name"].astype(str).values
        self.colores = df["color_nombre"].fillna("").astype(str).values
        mask = df["es_principal"].astype(bool).to_numpy(copy=True)
        if split:
            mask &= (df["split"] == split).values
        self.catalogo = np.where(mask)[0]
        self.por_ann = {int(a): i for i, a in enumerate(df["annotation_id"])}

        perfiles = json.loads(_localizar("perfiles_sinteticos.json",
                                         "STYLEMATE_DATA").read_text(encoding="utf-8"))
        self.perfiles = {p["profile_id"]: p for p in perfiles}

    # -- consultas -------------------------------------------------------------
    def listar_perfiles(self) -> list[dict]:
        return [{"perfil": k, "arquetipo": p["arquetipo"], "split": p["split"]}
                for k, p in self.perfiles.items()]

    def listar_prendas(self, categoria: str | None = None, n: int = 24) -> list[dict]:
        idx = self.catalogo
        if categoria:
            idx = idx[self.nombres[idx] == categoria]
        idx = np.random.default_rng(SEMILLA).permutation(idx)[:n]
        return [self._prenda(int(i)) for i in idx]

    def _vector_perfil(self, perfil):
        if isinstance(perfil, (list, tuple, np.ndarray)):
            return np.asarray(perfil, dtype=np.float32), "casual"
        if perfil in self.perfiles:
            p = self.perfiles[perfil]
            return np.asarray(p["vector"], dtype=np.float32), p["arquetipo"]
        grupo = [np.asarray(p["vector"], dtype=np.float32) for p in self.perfiles.values()
                 if p["arquetipo"] == perfil]            # nombre de arquetipo
        if grupo:
            return np.mean(grupo, axis=0).astype(np.float32), str(perfil)
        raise KeyError(f"Perfil desconocido: {perfil}")

    def _prenda(self, i: int) -> dict:
        r = self.df.iloc[i]
        return {"annotation_id": int(r["annotation_id"]), "categoria": str(r["category_name"]),
                "color": str(r["color_nombre"]) if pd.notna(r["color_nombre"]) else "",
                "pos": int(r["pos"]), "image_id": int(r["image_id"]),
                "file_name": str(r["file_name"]) if "file_name" in self.df.columns else ""}

    # -- reglas (misma logica que prendas_reglas de 4_3) ---------------------------
    def _penalizacion(self, ia: int, ib: int, ctx_base: dict):
        tokens = [TOKEN_CATEGORIA[n] for n in (self.nombres[ia], self.nombres[ib])
                  if n in TOKEN_CATEGORIA]
        style = 0.5 * (self.S[ia].numpy() + self.S[ib].numpy())
        d = self.DIM
        if "formalidad.sport" in d and style[d["formalidad.sport"]] > 0.25:
            tokens.append("sport")
        for k, tok in [("estampado.grafico", "estampado"), ("estampado.floral", "floral"),
                       ("estampado.rayas", "rayas"), ("estampado.cuadros", "cuadros")]:
            if k in d and style[d[k]] > 0.25:
                tokens.append(tok)
        ctx = dict(ctx_base)
        ctx["colores_vivos"] = (int(self.colores[ia] in COLORES_VIVOS)
                                + int(self.colores[ib] in COLORES_VIVOS)
                                + int("color_base.vivo" in d and style[d["color_base.vivo"]] > 0.30))
        ctx["material"] = ("denim" if "material.denim" in d
                           and style[d["material.denim"]] > 0.35 else "tela")
        if self.motor is not None:
            pen, detalle = self.motor.aplicar_reglas(tokens, ctx)
        else:
            pen, detalle = _reglas_simples(tokens, ctx)
        if isinstance(detalle, (list, tuple, set)):
            detalle = [str(x["regla"]) if isinstance(x, dict) and "regla" in x else str(x) for x in detalle]
        elif isinstance(detalle, dict):
            detalle = [str(k) for k in detalle]
        else:
            detalle = []
        return float(pen), detalle

    # -- recomendacion -------------------------------------------------------------
    @torch.no_grad()
    def recomendar(self, perfil, prenda_base, top_k: int = 5, ocasion: str | None = None,
                   clima: str | None = None, una_por_categoria: bool = True,
                   solo_validas: bool = True) -> dict:
        vec, arq = self._vector_perfil(perfil)
        ib = self.por_ann.get(int(prenda_base))
        if ib is None:
            raise KeyError(f"prenda_base {prenda_base} no existe (usa annotation_id)")

        ctx = dict(CONTEXTO.get(arq, CONTEXTO["casual"]))
        if ocasion:
            ctx["ocasion"] = ocasion
            ctx["formalidad"] = FORMALIDAD_POR_OCASION.get(ocasion, "casual")
        if clima:
            ctx["clima"] = clima

        cat_b = self.nombres[ib]
        cand = np.array([i for i in self.catalogo
                         if i != ib and self.nombres[i] != cat_b
                         and frozenset((cat_b, self.nombres[i])) not in PARES_REDUNDANTES])
        if len(cand) == 0:
            return self._salida(perfil, arq, ctx, ib, [], 0)
        n = len(cand)
        c = torch.as_tensor(cand, dtype=torch.long)

        # 1) compatibilidad (4.2): el par se arma (base, candidata)
        ea, sa = self.E[ib].expand(n, -1), self.S[ib].expand(n, -1)
        p_compat = torch.sigmoid(self.compat(ea, self.E[c], sa, self.S[c],
                                             self.cat_idx[ib].expand(n), self.cat_idx[c]))
        # 2) gusto (4.3): features identicas a features() del Sprint 2
        estilo_par = 0.5 * (sa + self.S[c])
        pv = torch.as_tensor(vec).expand(n, -1)
        x = torch.cat([pv, estilo_par, pv * estilo_par, (pv - estilo_par).abs(),
                       p_compat.unsqueeze(1)], dim=1)
        p_taste = torch.sigmoid(self.taste(x))

        # 3) reglas + ranking
        filas, descartadas = [], 0
        for j, ic in enumerate(cand):
            pen, detalle = self._penalizacion(ib, int(ic), ctx)
            if pen > 0 and solo_validas:
                descartadas += 1
                continue
            final = float(p_taste[j]) - 0.5 * pen   # misma formula que evaluar_ranking
            filas.append({**self._prenda(int(ic)),
                          "compat": round(float(p_compat[j]), 4),
                          "taste": round(float(p_taste[j]), 4),
                          "penalizacion": round(pen, 3), "final": round(final, 4),
                          "reglas_violadas": detalle})
        filas.sort(key=lambda r: r["final"], reverse=True)
        if una_por_categoria:
            vistos, unicas = set(), []
            for r in filas:
                if r["categoria"] not in vistos:
                    vistos.add(r["categoria"])
                    unicas.append(r)
            filas = unicas
        return self._salida(perfil, arq, ctx, ib, filas[:top_k], descartadas)

    def _salida(self, perfil, arq, ctx, ib, filas, descartadas) -> dict:
        for k, r in enumerate(filas, 1):
            r["rank"] = k
        return {"perfil": perfil if isinstance(perfil, str) else "personalizado",
                "arquetipo": arq,
                "contexto": {"ocasion": ctx["ocasion"], "clima": ctx["clima"]},
                "prenda_base": self._prenda(ib), "recomendaciones": filas,
                "descartadas_por_reglas": int(descartadas)}

        @lru_cache(maxsize=1)
    def _cargar_recortes(self):
        """Descarga y carga los recortes UNA sola vez para no saturar la memoria."""
        ruta = _localizar(ARCHIVO_RECORTES, "STYLEMATE_DATA", obligatorio=False)
        if ruta is None:
            print(f"No se encontró {ARCHIVO_RECORTES} localmente. Descargando desde Hugging Face (1.78 GB)...")
            try:
                # Descarga el archivo desde el Dataset público
                ruta_str = hf_hub_download(
                    repo_id=REPO_DATASET,
                    filename=ARCHIVO_RECORTES,
                    repo_type="dataset"
                )
                ruta = Path(ruta_str)
                print("¡Descarga completada!")
            except Exception as e:
                warnings.warn(f"Fallo al descargar los recortes: {e}")
                return None
                
        try:
            return np.load(ruta, mmap_mode="r")
        except Exception as e:
            warnings.warn(f"Fallo al cargar el archivo npy: {e}")
            return None

        @lru_cache(maxsize=1)
    def _cargar_recortes(self):
        """Descarga y carga los recortes UNA sola vez para no saturar la memoria."""
        ruta = _localizar(ARCHIVO_RECORTES, "STYLEMATE_DATA", obligatorio=False)
        if ruta is None:
            print(f"No se encontró {ARCHIVO_RECORTES} localmente. Descargando desde Hugging Face (1.78 GB)...")
            try:
                from huggingface_hub import hf_hub_download
                ruta_str = hf_hub_download(
                    repo_id=REPO_DATASET,
                    filename=ARCHIVO_RECORTES,
                    repo_type="dataset"
                )
                ruta = Path(ruta_str)
                print("¡Descarga completada!")
            except Exception as e:
                warnings.warn(f"Fallo al descargar los recortes: {e}")
                return None
                
        try:
            return np.load(ruta, mmap_mode="r")
        except Exception as e:
            warnings.warn(f"Fallo al cargar el archivo npy: {e}")
            return None

    def recorte(self, pos: int):
        """Recorte RGB 144x144 de la prenda, o None si falla."""
        recortes_array = self._cargar_recortes()
        if recortes_array is None:
            return None
        try:
            return np.asarray(recortes_array[int(pos)])
        except IndexError:
            return None


# ----------------------------------------------------------------------------
# Interfaz de modulo (la usan la web y las pruebas)
# ----------------------------------------------------------------------------
@lru_cache(maxsize=2)
def obtener(split: str | None = "test") -> Recomendador:
    return Recomendador(split)


def recomendar(perfil, prenda_base, top_k=5, **kw) -> dict:
    return obtener().recomendar(perfil, prenda_base, top_k, **kw)


def listar_prendas(categoria=None, n=24):
    return obtener().listar_prendas(categoria, n)


def listar_perfiles():
    return obtener().listar_perfiles()


# ----------------------------------------------------------------------------
# Verificacion contra los resultados guardados del Sprint 2
# ----------------------------------------------------------------------------
def verificar() -> bool:
    rec = obtener(None)
    ok = True
    print(f"[ok] {len(rec.df)} instancias, {len(rec.catalogo)} principales, "
          f"{len(rec.perfiles)} perfiles; reglas: "
          f"{'rules.py del Sprint 1' if rec.motor else 'SIMPLIFICADAS (no encontre rules.py)'}")

    # 1) compatibilidad: debe reproducir compat_score de pares_scored.csv
    ps = pd.read_csv(_localizar("pares_scored.csv", "STYLEMATE_DATA")).head(500)
    a, b = ps["pos_a"].values, ps["pos_b"].values
    with torch.no_grad():
        p = torch.sigmoid(rec.compat(rec.E[a], rec.E[b], rec.S[a], rec.S[b],
                                     rec.cat_idx[a], rec.cat_idx[b])).numpy()
    err = float(np.abs(p - ps["compat_score"].values).mean())
    print(f"[{'ok' if err < 1e-3 else 'FALLA'}] compatibilidad: error medio = {err:.6f}")
    ok &= err < 1e-3

    # 2) gusto: AUC sobre preferencias sinteticas de test (en 4.3 fue 1.000)
    try:
        from sklearn.metrics import roc_auc_score
        pref = pd.read_csv(_localizar("preferencias_sinteticas.csv", "STYLEMATE_DATA"))
        pref = pref[pref["split"] == "test"].head(3000)
        pares = pd.read_csv(_localizar("pares_scored.csv", "STYLEMATE_DATA")).set_index("pair_id")
        filas = pares.loc[pref["pair_id"].astype(int)]
        pa, pb = filas["pos_a"].values, filas["pos_b"].values
        sty = 0.5 * (rec.S[pa] + rec.S[pb])
        pv = torch.tensor(np.vstack([rec.perfiles[i]["vector"] for i in pref["profile_id"]]),
                          dtype=torch.float32)
        comp = torch.tensor(filas["compat_score"].values, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            x = torch.cat([pv, sty, pv * sty, (pv - sty).abs(), comp], dim=1)
            pt = torch.sigmoid(rec.taste(x)).numpy()
        auc = roc_auc_score(pref["label"].values, pt)
        print(f"[{'ok' if auc > 0.95 else 'FALLA'}] gusto: AUC preferencias test = {auc:.3f}")
        ok &= auc > 0.95
    except Exception as e:  # noqa: BLE001
        print(f"[?] no pude verificar el gusto: {e}")
    return bool(ok)


def main():
    ap = argparse.ArgumentParser(description="StyleMate - recomendador")
    ap.add_argument("--verificar", action="store_true")
    ap.add_argument("--listar", action="store_true")
    ap.add_argument("--categoria")
    ap.add_argument("--perfil")
    ap.add_argument("--prenda", type=int)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--ocasion", choices=OCASIONES)
    ap.add_argument("--clima", choices=CLIMAS)
    a = ap.parse_args()
    if a.verificar:
        raise SystemExit(0 if verificar() else 1)
    if a.listar:
        print(json.dumps({"perfiles": listar_perfiles()[:10],
                          "prendas": listar_prendas(a.categoria, 10)},
                         ensure_ascii=False, indent=2))
        return
    perfil = a.perfil or [p for p in listar_perfiles() if p["split"] == "test"][0]["perfil"]
    prenda = a.prenda or listar_prendas(a.categoria, 1)[0]["annotation_id"]
    print(json.dumps(recomendar(perfil, prenda, a.top, ocasion=a.ocasion, clima=a.clima),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

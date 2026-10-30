# -*- coding: utf-8 -*-
"""Diagnostico: que regla descarta las candidatas con el rules.py real. Solo lee, no cambia nada.
Ejecutar desde la carpeta sprint3:  python diagnostico_reglas.py
"""
import collections
import re

import recomendador as rec

r = rec.obtener()
print("Motor de reglas:", "REAL (rules.py)" if r.motor is not None else "simplificado")
print("Dimensiones cargadas:", len(r.DIM))

S = r.S.numpy()
cat = r.catalogo
print("\nFraccion de prendas del catalogo por encima del umbral que usa recomendador.py:")
for k, thr in [("formalidad.sport", .25), ("estampado.grafico", .25), ("estampado.floral", .25),
               ("estampado.rayas", .25), ("estampado.cuadros", .25),
               ("color_base.vivo", .30), ("material.denim", .35)]:
    if k in r.DIM:
        print(f"  {k:20s} > {thr}: {(S[cat, r.DIM[k]] > thr).mean():.1%}")

perfiles = [p for p in r.listar_perfiles() if p["split"] == "test"][:3]
bases = r.listar_prendas(n=3)
print("\nCandidatas con penalizacion > 0 (sin filtrar), por perfil y prenda base:")
for p in perfiles:
    for b in bases:
        o = r.recomendar(p["perfil"], b["annotation_id"], top_k=100000,
                         una_por_categoria=False, solo_validas=False)
        recs = o["recomendaciones"]
        con = [x for x in recs if x["penalizacion"] > 0]
        cnt = collections.Counter()
        for x in con:
            for s in x["reglas_violadas"]:
                m = re.search(r"'regla': '([^']+)'", s)
                cnt[m.group(1) if m else s] += 1
        print(f"- {p['perfil']} ({o['arquetipo']}) | {o['contexto']} | base: {b['categoria']}"
              f" | candidatas: {len(recs)} | con violacion: {len(con)}")
        print(f"    reglas: {dict(cnt)}")

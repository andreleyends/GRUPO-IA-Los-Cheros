# -*- coding: utf-8 -*-
"""
Tarea 1.7 - rules.py v1: lexico de moda, vectorizador de 40 dimensiones y motor
de reglas de gusto.

El vector de gusto resume las preferencias de una persona (o de una prenda) en
40 numeros entre 0 y 1, agrupados en 8 familias de 5 y 6 dimensiones. El motor
de reglas aplica restricciones duras y blandas que el modelo aprendido NO puede
garantizar por si solo (por ejemplo, no recomendar shorts en invierno o no
mezclar dos estampados fuertes).

Uso:
    python -X utf8 rules.py demo     # imprime vector y veredicto de reglas
"""

import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
SPRINT1 = AQUI.parent
IDX = SPRINT1 / "salidas" / "indices"
LEXICO = AQUI / "lexico_moda.json"


# --------------------------------------------------------------------------- #
# 1. CARGA DEL LEXICO Y DEFINICION DE LAS 40 DIMENSIONES
# --------------------------------------------------------------------------- #
def cargar_lexico():
    return json.loads(LEXICO.read_text(encoding="utf-8"))


def orden_dimensiones(lexico):
    """Lista plana de las 40 dimensiones, en orden estable."""
    orden = []
    for familia, dims in lexico["dimensiones"].items():
        for d in dims:
            orden.append(f"{familia}.{d}")
    return orden


# --------------------------------------------------------------------------- #
# 2. VECTORIZADOR A 40 DIMENSIONES
# --------------------------------------------------------------------------- #
def _normalizar(v):
    m = float(np.max(v))
    return v / m if m > 0 else v


def vector_desde_terminos(terminos, lexico):
    """Suma las contribuciones de cada termino conocido y normaliza a [0,1]."""
    orden = orden_dimensiones(lexico)
    pos = {name: i for i, name in enumerate(orden)}
    v = np.zeros(len(orden), dtype=float)
    for t in terminos:
        clave = t.strip().lower()
        registro = lexico["terminos"].get(clave)
        if not registro:                       # busqueda por subcadena
            for k, val in lexico["terminos"].items():
                if clave and (clave in k or k in clave):
                    registro = val
                    break
        if registro:
            for dim, peso in registro.items():
                if dim in pos:
                    v[pos[dim]] += peso
    return _normalizar(v)


def vector_desde_texto(texto, lexico):
    """Acepta una frase separada por comas; tambien busca terminos sueltos."""
    partes = [p for p in texto.replace(";", ",").split(",")]
    v = vector_desde_terminos(partes, lexico)
    if v.sum() == 0:                            # texto libre: rastrear palabras
        palabras = [w for w in texto.lower().split() if len(w) > 2]
        v = vector_desde_terminos(palabras, lexico)
    return v


def vector_compuesto(perfil, atas, lexico):
    """Mezcla el perfil declarado con la senal observada en la ropa del ataud."""
    base = vector_desde_terminos(perfil, lexico)
    observado = vector_desde_terminos(atas, lexico)
    return _normalizar(0.6 * base + 0.4 * observado)


# --------------------------------------------------------------------------- #
# 3. MOTOR DE REGLAS (v1)
# --------------------------------------------------------------------------- #
# Cada regla devuelve (penalizacion, motivo). Las penalizaciones se suman y
# mueven el ranking final hacia abajo; el valor absoluto no es critico.

def r_invierno_sin_short(prendas, ctx):
    if ctx.get("clima") == "frio" and "shorts" in prendas:
        return 1.0, "Shorts con clima frio"
    return 0.0, None


def r_oficina_sin_deportivo(prendas, ctx):
    if ctx.get("ocasion") == "oficina" and {"shorts", "sport"} & set(prendas):
        return 0.8, "Prenda deportiva en contexto de oficina"
    return 0.0, None


def r_formal_sin_denim(prendas, ctx):
    if ctx.get("formalidad") == "formal" and ctx.get("material") == "denim":
        return 0.9, "Denim en look formal"
    return 0.0, None


def r_estampados_fuertes(prendas, ctx):
    fuertes = [p for p in prendas if p in {"estampado", "grafico", "floral",
                                           "rayas", "cuadros"}]
    if len(fuertes) >= 2:
        return 0.6, "Dos o mas estampados fuertes en el mismo conjunto"
    return 0.0, None


def r_color_vivo(prendas, ctx):
    vivos = ctx.get("colores_vivos", 0)
    if vivos >= 3:
        return 0.5, "Muchos colores vivos compitiendo"
    return 0.0, None


def r_lluvia_calzado(prendas, ctx):
    if ctx.get("clima") == "lluvia" and "sandalia" in prendas:
        return 0.7, "Calzado abierto con lluvia"
    return 0.0, None


REGLAS = [
    ("invierno_sin_short", r_invierno_sin_short),
    ("oficina_sin_deportivo", r_oficina_sin_deportivo),
    ("formal_sin_denim", r_formal_sin_denim),
    ("sin_doble_estampado", r_estampados_fuertes),
    ("limite_color_vivo", r_color_vivo),
    ("lluvia_calzado", r_lluvia_calzado),
]


def aplicar_reglas(prendas, ctx):
    """Devuelve la penalizacion total y los motivos activados."""
    total, motivos = 0.0, []
    for nombre, fn in REGLAS:
        pen, motivo = fn(prendas, ctx)
        if pen > 0:
            total += pen
            motivos.append({"regla": nombre, "penalizacion": pen, "motivo": motivo})
    return total, motivos


def puntuar_conjunto(prendas, ctx, compatibilidad_modelo=0.0):
    """Score final = aporte del modelo menos penalizaciones de reglas."""
    pen, motivos = aplicar_reglas(prendas, ctx)
    return compatibilidad_modelo - pen, motivos


# --------------------------------------------------------------------------- #
# 4. DEMO CON EVIDENCIA
# --------------------------------------------------------------------------- #
def demo():
    lexico = cargar_lexico()
    orden = orden_dimensiones(lexico)
    assert len(orden) == 40, f"El vector debe tener 40 dims, tiene {len(orden)}"

    texto = "elegante, oficina, minimalista, colores neutros, entallado"
    v = vector_desde_texto(texto, lexico)

    casos = [
        (["shorts", "top, t-shirt, sweatshirt"], {"clima": "frio", "ocasion": "diario",
                                                  "colores_vivos": 1}, 0.9),
        (["suit", "shirt, blouse"], {"clima": "entretiempo", "ocasion": "oficina",
                                     "formalidad": "formal", "colores_vivos": 0}, 0.8),
        (["estampado", "floral", "cuadros"], {"clima": "calor", "ocasion": "noche",
                                              "colores_vivos": 3}, 0.7),
    ]
    resultados = []
    for prendas, ctx, base in casos:
        score, motivos = puntuar_conjunto(prendas, ctx, base)
        resultados.append({"prendas": prendas, "contexto": ctx,
                           "score_modelo": base, "score_final": round(score, 2),
                           "reglas": motivos})

    # Evidencia para el informe.
    evidencia = {
        "texto_preferencia": texto,
        "dimensiones": orden,
        "vector": [round(float(x), 3) for x in v],
        "dimensiones_activas": [orden[i] for i in np.nonzero(v > 0)[0]],
        "casos": resultados,
    }
    IDX.mkdir(parents=True, exist_ok=True)
    (IDX / "demo_reglas.json").write_text(
        json.dumps(evidencia, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Vector de gusto ({len(v)} dims) para: {texto}")
    print("  activas:", ", ".join(evidencia["dimensiones_activas"]))
    print("\nCasos evaluados:")
    for r in resultados:
        print(f"  {r['prendas']} -> score {r['score_modelo']} => {r['score_final']}")
        for m in r["reglas"]:
            print(f"      - {m['motivo']} ({m['regla']})")
    print(f"\nEvidencia: {IDX / 'demo_reglas.json'}")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if arg == "demo":
        demo()
    else:
        print("Uso: rules.py demo")


if __name__ == "__main__":
    main()

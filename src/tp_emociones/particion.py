"""Partición train / val / test por video (regla de oro del enunciado).

Por qué por video y no por imagen: dos cuadros del mismo plano son casi idénticos. Si uno
cae en train y otro en test, el modelo "reconoce el plano" y el accuracy se infla.

Reglas (BITACORA D-004):
  R1. Ningún video aparece a la vez en train y en (val o test). Esta regla no tiene
      excepciones y la verifica `verificar_sin_fugas`.
  R2. La partición se hace por actor: cada actor queda en train (indispensable para la
      tarea de identidad) y, si alcanza, también en val y en test.
  R3. Paso 1: por actor, se eligen los videos de train buscando ~70 % de sus imágenes
      (búsqueda exhaustiva si tiene <= 16 videos), prefiriendo -con hasta 10 puntos de
      tolerancia- dejar para evaluación videos que se puedan separar entre val y test. Paso 2: los videos restantes se reparten
      entre val y test *por video*, buscando mitades parecidas.
  R4. Si el paso 2 no puede repartir por video sin dejar a val o a test casi vacío (pasa
      cuando quedan 1 o 2 videos de evaluación muy desparejos), las imágenes de evaluación
      se cortan en dos mitades contiguas (orden de archivo ~ orden temporal). Val y test
      pueden compartir un video; train nunca. Concesión documentada: sin ella, esos
      actores no tendrían validación o test.
  R5. Actores con 1 solo video: todo a train y se reporta (no se pueden evaluar sin fuga).

Determinismo: los empates se rompen con un hash (SHA-256) del nombre del video más la
semilla, nunca con el orden de llegada de los archivos.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from . import config

SPLITS = ("train", "val", "test")
_MAX_EXHAUSTIVO = 16  # 2**16 = 65.536 combinaciones: instantáneo
MIN_FRAC_EVAL = 0.35  # val y test deben tener al menos 35 % de la evaluación cada uno
# Cuánto desvío del 70 % en train se tolera para evitar que val y test compartan video:
# 0.10 = hasta 10 puntos porcentuales. Ver BITACORA D-004.
PENALIDAD_R4 = 0.10


def _separable_por_video(tamanios: pd.Series, min_frac: float = MIN_FRAC_EVAL) -> bool:
    """¿Existe un reparto de estos videos en val/test con >= min_frac de cada lado?"""
    n, total = len(tamanios), tamanios.sum()
    if n < 2:
        return False
    if n > _MAX_EXHAUSTIVO:
        return True  # con tantos videos chicos siempre se puede balancear
    valores = tamanios.to_numpy()
    for mascara in range(1, 2 ** n - 1):
        n_val = sum(valores[i] for i in range(n) if mascara >> i & 1)
        if min(n_val, total - n_val) >= min_frac * total:
            return True
    return False


def _h(nombre: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{nombre}".encode()).hexdigest()


def _elegir_train(tamanios: pd.Series, frac_train: float, seed: int) -> set[str]:
    """Paso 1: qué videos de un actor van a train, buscando ~frac_train de sus imágenes.

    Siempre queda al menos un video fuera de train (para poder evaluar al actor).
    """
    videos = sorted(tamanios.index, key=lambda v: (-tamanios[v], _h(v, seed)))
    total = tamanios.sum()
    objetivo = frac_train * total
    if len(videos) <= _MAX_EXHAUSTIVO:
        mejor, mejor_costo = None, np.inf
        for mascara in range(1, 2 ** len(videos) - 1):  # excluye "ninguno" y "todos"
            elegidos = [v for i, v in enumerate(videos) if mascara >> i & 1]
            resto = tamanios.drop(elegidos)
            # Costo = desvío del 70 % (en fracción del actor) + penalidad si lo que queda
            # para evaluar no se puede separar por video (forzaría R4).
            costo = abs(tamanios[elegidos].sum() - objetivo) / total
            if not _separable_por_video(resto):
                costo += PENALIDAD_R4
            if costo < mejor_costo:  # ante empate gana el primero -> determinista
                mejor, mejor_costo = set(elegidos), costo
        return mejor
    # Muchos videos: los grandes primero; los chicos permiten ajustar fino.
    elegidos, acumulado = set(), 0
    for v in videos:
        if acumulado + tamanios[v] <= objetivo or not elegidos:
            elegidos.add(v)
            acumulado += tamanios[v]
    if len(elegidos) == len(videos):
        elegidos.discard(videos[-1])
    return elegidos


def _repartir_evaluacion(g: pd.DataFrame, seed: int, min_frac: float = MIN_FRAC_EVAL) -> pd.Series:
    """Paso 2: reparte entre val y test las imágenes de un actor que no son de train."""
    tamanios = g.groupby("video").size()
    videos = sorted(tamanios.index, key=lambda v: _h(v, seed))
    total = tamanios.sum()

    mejor, mejor_costo = None, np.inf
    if 2 <= len(videos) <= _MAX_EXHAUSTIVO:
        for mascara in range(1, 2 ** len(videos) - 1):
            a_val = [v for i, v in enumerate(videos) if mascara >> i & 1]
            n_val = tamanios[a_val].sum()
            if min(n_val, total - n_val) < min_frac * total:
                continue
            costo = abs(n_val - total / 2)
            if costo < mejor_costo:
                mejor, mejor_costo = set(a_val), costo
    if mejor is not None:  # R3: por video
        return pd.Series(np.where(g["video"].isin(mejor), "val", "test"), index=g.index)

    # R4: mitades contiguas. Se ordena por video (orden hash) y dentro de cada video por
    # nombre de archivo, así a lo sumo UN video queda repartido entre val y test.
    orden = g.assign(_hv=g["video"].map(lambda v: _h(v, seed))).sort_values(["_hv", "clave"]).index
    mitad = len(orden) // 2
    etiquetas = pd.Series(["val"] * mitad + ["test"] * (len(orden) - mitad), index=orden)
    return etiquetas.reindex(g.index)


def particionar(
    df: pd.DataFrame,
    fracciones: dict[str, float] | None = None,
    seed: int = config.SEED,
) -> pd.Series:
    """Devuelve una Serie (mismo índice que df) con 'train' | 'val' | 'test'.

    `split.attrs["avisos"]` lista los actores donde se aplicó R4 o R5.
    """
    fracciones = fracciones or config.FRACCIONES
    split = pd.Series(index=df.index, dtype=object)
    avisos = []

    for actor, g in df.groupby("actor", sort=True):
        tamanios = g.groupby("video").size()
        if len(tamanios) == 1:  # R5
            split.loc[g.index] = "train"
            avisos.append(f"{actor}: 1 solo video -> todo a train, no evaluable (R5)")
            continue

        es_train = g["video"].isin(_elegir_train(tamanios, fracciones["train"], seed))
        split.loc[g.index[es_train]] = "train"
        ev = _repartir_evaluacion(g[~es_train], seed)
        split.loc[ev.index] = ev

        v_val = set(g.loc[ev.index[ev == "val"], "video"])
        v_test = set(g.loc[ev.index[ev == "test"], "video"])
        if v_val & v_test:
            avisos.append(f"{actor}: val y test comparten {sorted(v_val & v_test)} (R4)")

    split.attrs["avisos"] = avisos
    return split


def verificar_sin_fugas(df: pd.DataFrame, split: pd.Series) -> None:
    """Falla (AssertionError) si algún video de train aparece en val o test (R1)."""
    videos = {s: set(df.loc[split == s, "video"]) for s in SPLITS}
    fuga = videos["train"] & (videos["val"] | videos["test"])
    assert not fuga, f"FUGA: {len(fuga)} videos en train y en evaluación: {sorted(fuga)[:5]}"
    sin_train = set(df["actor"]) - set(df.loc[split == "train", "actor"])
    assert not sin_train, f"Actores sin imágenes en train: {sin_train}"
    assert split.notna().all(), "Hay imágenes sin split asignado"


def resumen_particion(df: pd.DataFrame, split: pd.Series) -> pd.DataFrame:
    """Imágenes, videos y actores por split, y distribución de emociones (en %)."""
    d = df.assign(split=split)
    filas = []
    for s in SPLITS:
        x = d[d["split"] == s]
        fila = {"split": s, "imagenes": len(x), "%": round(100 * len(x) / len(d), 1),
                "videos": x["video"].nunique(), "actores": x["actor"].nunique()}
        dist = x["emocion"].value_counts(normalize=True)
        fila.update({e: round(100 * dist.get(e, 0.0), 1) for e in config.EMOCIONES})
        filas.append(fila)
    return pd.DataFrame(filas).set_index("split")

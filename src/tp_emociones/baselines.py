"""Baselines clásicos y las dos mediciones del "acta de límites" (NB01 y NB02 del curso).

- Trivial: predice siempre la clase más frecuente de train. Es el piso: cualquier modelo
  que no lo supere no aprendió nada.
- Random Forest y MLPClassifier sobre píxeles: 64x64 en escala de grises -> 4096 columnas.
  Se reduce porque con 224x224x3 = 150.528 columnas y ~1.900 imágenes no hay de dónde
  generalizar (y el costo se dispara). Es la misma decisión de la diapositiva 15 del TP1.
- Acta de límites (NB02): (1) cómo escala el costo de entrenar MLPs más grandes y
  (2) el experimento de permutación de píxeles.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

from . import config


def cargar_pixeles(rutas, tam: int = config.IMG_SIZE_BASELINE) -> np.ndarray:
    """Lee imágenes, las pasa a gris, las reduce a tam x tam y las aplana en [0, 1]."""
    X = np.empty((len(rutas), tam * tam), dtype=np.float32)
    for i, r in enumerate(rutas):
        with Image.open(r) as im:
            X[i] = np.asarray(im.convert("L").resize((tam, tam), Image.BILINEAR), dtype=np.float32).ravel() / 255.0
    return X


def metricas(y_true, y_pred) -> dict[str, float]:
    """Accuracy y F1 macro. F1 macro pesa igual a cada clase: con clases desbalanceadas
    (Neutral 28 % vs Tristeza 12 %) es la métrica honesta."""
    return {"accuracy": accuracy_score(y_true, y_pred), "f1_macro": f1_score(y_true, y_pred, average="macro")}


def _a_enteros(y_tr, y_ev) -> tuple[np.ndarray, np.ndarray]:
    """Etiquetas de texto -> enteros con un vocabulario común a train y evaluación.

    Motivo: scikit-learn 1.6 (la versión de Colab) falla en MLPClassifier con
    early_stopping=True si las etiquetas son strings. Con enteros funciona en todas.
    """
    clases = np.unique(np.concatenate([np.asarray(y_tr), np.asarray(y_ev)]))
    indice = {c: i for i, c in enumerate(clases)}
    return np.array([indice[v] for v in y_tr]), np.array([indice[v] for v in y_ev])


def modelos_baseline(seed: int = config.SEED) -> dict[str, object]:
    return {
        "Trivial (clase más frecuente)": DummyClassifier(strategy="most_frequent"),
        "Random Forest (píxeles 64x64)": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=seed),
        "MLP sklearn (píxeles 64x64)": MLPClassifier(
            hidden_layer_sizes=(256,), early_stopping=True, max_iter=200, random_state=seed),
    }


def evaluar_baselines(X_tr, y_tr, X_ev, y_ev, seed: int = config.SEED) -> pd.DataFrame:
    """Entrena cada baseline en train y lo mide en el conjunto de evaluación dado."""
    y_tr, y_ev = _a_enteros(y_tr, y_ev)
    filas = []
    for nombre, modelo in modelos_baseline(seed).items():
        t0 = time.perf_counter()
        modelo.fit(X_tr, y_tr)
        seg = time.perf_counter() - t0
        filas.append({"modelo": nombre, **metricas(y_ev, modelo.predict(X_ev)), "segundos_fit": round(seg, 1)})
    return pd.DataFrame(filas).set_index("modelo")


# ---------------------------------------------------------------------------
# Acta de límites (NB02)
# ---------------------------------------------------------------------------
def medir_costo_mlp(X_tr, y_tr, X_ev, y_ev, arquitecturas=((64,), (256,), (512, 256), (1024, 512)),
                    max_iter: int = 15, seed: int = config.SEED) -> pd.DataFrame:
    """Medición 1: parámetros vs tiempo vs accuracy al agrandar el MLP de sklearn."""
    y_tr, y_ev = _a_enteros(y_tr, y_ev)
    filas = []
    for arq in arquitecturas:
        m = MLPClassifier(hidden_layer_sizes=arq, max_iter=max_iter, random_state=seed)
        t0 = time.perf_counter()
        m.fit(X_tr, y_tr)
        seg = time.perf_counter() - t0
        n_params = sum(w.size for w in m.coefs_) + sum(b.size for b in m.intercepts_)
        filas.append({"arquitectura": str(arq), "parametros": n_params, "segundos": round(seg, 1),
                      "accuracy": accuracy_score(y_ev, m.predict(X_ev))})
    return pd.DataFrame(filas).set_index("arquitectura")


def permutacion_fija(n_features: int, seed: int = config.SEED) -> np.ndarray:
    """Una única permutación de posiciones, igual para todas las imágenes."""
    return np.random.default_rng(seed).permutation(n_features)


def experimento_permutacion_mlp(X_tr, y_tr, X_ev, y_ev, seed: int = config.SEED) -> dict[str, float]:
    """Medición 2: el MLP rinde igual con los píxeles mezclados -> no usa la estructura
    espacial. (La misma prueba sobre la CNN está en interpretacion.py.)"""
    y_tr, y_ev = _a_enteros(y_tr, y_ev)
    perm = permutacion_fija(X_tr.shape[1], seed)
    res = {}
    for nombre, (a, b) in {"original": (X_tr, X_ev), "permutada": (X_tr[:, perm], X_ev[:, perm])}.items():
        m = MLPClassifier(hidden_layer_sizes=(256,), early_stopping=True, max_iter=200, random_state=seed)
        m.fit(a, y_tr)
        res[nombre] = accuracy_score(y_ev, m.predict(b))
    return res

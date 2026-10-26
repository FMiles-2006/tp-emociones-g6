"""Evaluación honesta e interpretación (NB12, NB16).

- Métricas: accuracy + F1 macro + reporte por clase + matriz de confusión.
- Peores errores: las imágenes mal clasificadas con mayor confianza.
- Mapa de oclusión: tapa regiones con un parche gris y mide cuánto cae la probabilidad de
  la clase correcta. Responde la pregunta clave de este dataset: ¿el modelo mira la CARA
  o el fondo/la fuente? (en identidad, actor y fuente casi coinciden).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from . import config


def predecir(modelo, ds) -> np.ndarray:
    """Probabilidades (N, K) en el orden del dataset (val/test no se mezclan)."""
    return modelo.predict(ds, verbose=0)


def metricas(y_true, y_pred) -> dict[str, float]:
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro"))}


def reporte(y_true, y_prob, clases: list[str], titulo: str = "", mostrar: bool = True) -> dict:
    y_pred = y_prob.argmax(axis=1)
    m = metricas(y_true, y_pred)
    if mostrar:
        print(f"== {titulo}  accuracy={m['accuracy']:.3f}  F1 macro={m['f1_macro']:.3f}")
        print(classification_report(y_true, y_pred, labels=range(len(clases)), target_names=clases, zero_division=0))
    return m


def graficar_matriz_confusion(y_true, y_prob, clases: list[str], titulo: str = "", normalizar: bool = True, ax=None):
    """Matriz normalizada por fila: cada fila suma 1 = recall de cada clase."""
    y_pred = y_prob.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(clases)), normalize="true" if normalizar else None)
    if ax is None:
        lado = max(5, 0.45 * len(clases) + 2)
        _, ax = plt.subplots(figsize=(lado, lado))
    ConfusionMatrixDisplay(cm, display_labels=clases).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format=".2f" if normalizar else "d", xticks_rotation=90)
    ax.set_title(titulo)
    plt.tight_layout()
    return ax


def pares_mas_confundidos(y_true, y_prob, clases: list[str], n: int = 5) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_prob.argmax(axis=1), labels=range(len(clases)))
    np.fill_diagonal(cm, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-cm.ravel()), cm.shape))[0][:n]
    return pd.DataFrame([{"real": clases[i], "predicho": clases[j], "casos": int(cm[i, j])} for i, j in idx if cm[i, j]])


def graficar_curvas(historias: dict[str, object], titulo: str = ""):
    """Curvas loss/accuracy de train y val. Acepta varias fases concatenadas:
    historias = {"fase 1": h1, "fase 2": h2}."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    desplazamiento = 0
    for nombre, h in historias.items():
        hh = h.history
        ep = np.arange(len(hh["loss"])) + desplazamiento
        for ax, met in zip(axes, ("loss", "accuracy")):
            ax.plot(ep, hh[met], label=f"train {nombre}")
            ax.plot(ep, hh[f"val_{met}"], "--", label=f"val {nombre}")
        desplazamiento += len(hh["loss"])
    for ax, met in zip(axes, ("Pérdida", "Accuracy")):
        ax.set_title(f"{titulo} — {met}"); ax.set_xlabel("época"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def mostrar_peores_errores(rutas, y_true, y_prob, clases: list[str], n: int = 12, titulo: str = ""):
    """Errores con mayor confianza: ¿son razonables (sorpresa<->alegría) o absurdos?"""
    y_pred = y_prob.argmax(axis=1)
    err = np.where(y_pred != y_true)[0]
    if len(err) == 0:
        print("Sin errores."); return
    peores = err[np.argsort(-y_prob[err, y_pred[err]])][:n]
    cols = 6
    filas = int(np.ceil(len(peores) / cols))
    fig, axes = plt.subplots(filas, cols, figsize=(2.3 * cols, 2.7 * filas))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).ravel(), peores):
        ax.imshow(Image.open(rutas[i]))
        ax.set_title(f"real: {clases[y_true[i]]}\npred: {clases[y_pred[i]]} ({y_prob[i, y_pred[i]]:.0%})", fontsize=8, color="red")
    fig.suptitle(titulo)
    plt.tight_layout()
    return fig


def mapa_oclusion(modelo, imagen: np.ndarray, clase: int, parche: int = 32, paso: int = 16,
                  gris: float = 127.5) -> tuple[np.ndarray, float]:
    """Caída de probabilidad de `clase` al tapar cada región (imagen en [0, 255]).

    Se arman TODAS las imágenes tapadas y se predice en un solo lote (NB12).
    """
    lado = imagen.shape[0]
    pos = list(range(0, lado - parche + 1, paso))
    lote = []
    for y in pos:
        for x in pos:
            tapada = imagen.copy()
            tapada[y:y + parche, x:x + parche] = gris
            lote.append(tapada)
    prob_base = float(modelo.predict(imagen[None].astype("float32"), verbose=0)[0, clase])
    probs = modelo.predict(np.stack(lote).astype("float32"), batch_size=64, verbose=0)[:, clase]
    caidas = (prob_base - probs).reshape(len(pos), len(pos))
    # Se lleva a resolución de imagen promediando los parches que cubren cada píxel.
    mapa = np.zeros((lado, lado)); cuenta = np.zeros((lado, lado))
    for a, y in enumerate(pos):
        for b, x in enumerate(pos):
            mapa[y:y + parche, x:x + parche] += caidas[a, b]
            cuenta[y:y + parche, x:x + parche] += 1
    return mapa / np.maximum(cuenta, 1), prob_base


def graficar_oclusiones(modelo, rutas, y_true, clases: list[str], n: int = 4, titulo: str = ""):
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.2))
    for k, (ruta, y) in enumerate(list(zip(rutas, y_true))[:n]):
        img = np.asarray(Image.open(ruta).convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE)), dtype="float32")
        mapa, p = mapa_oclusion(modelo, img, int(y))
        axes[0, k].imshow(img.astype("uint8")); axes[0, k].set_title(f"{clases[y]} (p={p:.2f})", fontsize=9)
        axes[1, k].imshow(img.astype("uint8")); axes[1, k].imshow(mapa, cmap="jet", alpha=0.5)
        axes[0, k].axis("off"); axes[1, k].axis("off")
    axes[1, 0].set_title("rojo = región decisiva", fontsize=8, loc="left")
    fig.suptitle(titulo)
    plt.tight_layout()
    return fig


def guardar_resultados(resultados: dict, ruta: str | Path) -> None:
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    Path(ruta).write_text(json.dumps(resultados, indent=2, ensure_ascii=False))

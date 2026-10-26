"""Pipelines tf.data con el patrón del curso: map -> cache -> shuffle -> batch -> prefetch.

Decisiones (BITACORA D-006):
- Las imágenes se cachean como uint8 (224x224x3 = 150 KB c/u; ~400 MB el dataset entero).
  Cachear en float32 cuadruplicaría la RAM sin ganar nada.
- Ni el escalado ni la aumentación viven en el pipeline: van como **capas del modelo**
  (práctica moderna vista en la clase 17). Así el modelo guardado incluye su propio
  preprocesamiento y no hay riesgo de que entrenamiento e inferencia preprocesen distinto.
- Val y test nunca se mezclan ni se aumentan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf

from . import config

AUTOTUNE = tf.data.AUTOTUNE


def codificar_etiquetas(df: pd.DataFrame, columna: str, clases: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Etiquetas de texto -> enteros 0..K-1, con un orden de clases fijo (alfabético).

    El orden se fija una vez y se reutiliza para train/val/test: si cada split codificara
    por su cuenta, el índice 3 podría significar cosas distintas.
    """
    clases = clases or sorted(df[columna].unique())
    indice = {c: i for i, c in enumerate(clases)}
    desconocidas = set(df[columna]) - set(indice)
    if desconocidas:
        raise ValueError(f"Clases no vistas en el vocabulario: {desconocidas}")
    return df[columna].map(indice).to_numpy(dtype=np.int32), list(clases)


def _leer_imagen(ruta: tf.Tensor, etiqueta: tf.Tensor):
    img = tf.io.decode_image(tf.io.read_file(ruta), channels=3, expand_animations=False)
    img = tf.image.resize(img, (config.IMG_SIZE, config.IMG_SIZE), method="bilinear")
    img = tf.cast(tf.round(img), tf.uint8)
    img.set_shape((config.IMG_SIZE, config.IMG_SIZE, 3))
    return img, etiqueta


def _a_float(img, etiqueta):
    return tf.cast(img, tf.float32), etiqueta  # rango [0, 255]; el modelo escala adentro


def _permutar(perm: np.ndarray):
    """Aplica una permutación fija de posiciones espaciales (experimento del NB11)."""
    perm_t = tf.constant(perm, dtype=tf.int32)
    lado = config.IMG_SIZE

    def f(img, etiqueta):
        plano = tf.reshape(img, (lado * lado, 3))
        return tf.reshape(tf.gather(plano, perm_t), (lado, lado, 3)), etiqueta

    return f


def crear_dataset(
    rutas,
    etiquetas,
    entrenamiento: bool,
    batch_size: int = config.BATCH_SIZE,
    seed: int = config.SEED,
    permutacion: np.ndarray | None = None,
) -> tf.data.Dataset:
    """Dataset listo para `model.fit` / `model.evaluate`.

    `entrenamiento=True` mezcla (con semilla fija, re-mezclando en cada época).
    `permutacion`: solo para el experimento de permutación de píxeles.
    """
    ds = tf.data.Dataset.from_tensor_slices((np.asarray(rutas, dtype=str), np.asarray(etiquetas)))
    ds = ds.map(_leer_imagen, num_parallel_calls=AUTOTUNE)
    if permutacion is not None:
        ds = ds.map(_permutar(permutacion), num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    if entrenamiento:
        ds = ds.shuffle(len(rutas), seed=seed, reshuffle_each_iteration=True)
    return ds.map(_a_float, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)


def datasets_por_split(df: pd.DataFrame, tarea: str, **kwargs) -> tuple[dict[str, tf.data.Dataset], list[str], dict[str, np.ndarray]]:
    """Arma train/val/test para una tarea ('actor' o 'emocion').

    Devuelve (datasets, clases, etiquetas_enteras_por_split). El vocabulario de clases se
    toma de train (el registro garantiza que todos los actores están en train).
    """
    _, clases = codificar_etiquetas(df[df["split"] == "train"], tarea)
    dsets, ys = {}, {}
    for split in ("train", "val", "test"):
        parte = df[df["split"] == split]
        y, _ = codificar_etiquetas(parte, tarea, clases)
        ys[split] = y
        dsets[split] = crear_dataset(parte["ruta"].to_numpy(), y, entrenamiento=(split == "train"), **kwargs)
    return dsets, clases, ys


def pesos_por_clase(y: np.ndarray, n_clases: int) -> dict[int, float]:
    """Pesos 'balanced' (n / (K * n_c)): cada clase aporta lo mismo a la pérdida.

    Motivo: Neutral es 30 % y Tristeza 12 % en train; sin pesos, el modelo gana
    accuracy prediciendo de más las clases grandes (lo que el F1 macro castiga).
    """
    conteo = np.bincount(y, minlength=n_clases)
    return {i: float(len(y) / (n_clases * c)) if c else 0.0 for i, c in enumerate(conteo)}

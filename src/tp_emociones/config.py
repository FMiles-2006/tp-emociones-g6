"""Configuración central del proyecto.

Todo valor que influya en un resultado (semilla, clases, tamaños, fracciones de la
partición) vive acá y solo acá: si dos notebooks importan este módulo, están usando
exactamente la misma configuración. Cambiar algo de este archivo es una decisión que
se registra en docs/BITACORA.md.
"""
from __future__ import annotations

import os
import random

import numpy as np

SEED: int = 42

# Vocabulario de emociones cerrado por el curso (Miedo se descartó, ver BITACORA D-001).
EMOCIONES: tuple[str, ...] = ("alegria", "enojo", "neutral", "sorpresa", "tristeza")

# Las imágenes del dataset ya vienen en 224x224 RGB: es la entrada nativa de ResNet50.
IMG_SIZE: int = 224
# Resolución reducida para los baselines clásicos (RF/MLP sobre píxeles en gris).
IMG_SIZE_BASELINE: int = 64

# Fracciones objetivo de la partición (en imágenes, por actor). Ver BITACORA D-004.
FRACCIONES: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}

BATCH_SIZE: int = 32

# Nombres de archivo/carpeta del dataset del curso (DATASET_FINAL del Drive compartido).
CSV_MASTER: str = "etiquetas_master_final.csv"
CARPETA_IMAGENES: str = "Imagenes_Finales"

# Unificación de identidades que el CSV del curso todavía escribe distinto.
# El curso ya aplica un mapa equivalente; lo repetimos para no depender de que esté corrido.
MAPA_ACTORES: dict[str, str] = {
    "tini": "tini_stoessel",
    "georginabarbarossa": "georgina_barbarossa",
    "oscarmartinez": "oscar_martinez",
    "juliochavez": "julio_chavez",
    "pachupena": "pachu_pena",
    "rodrigodelaserna": "rodrigo_de_la_serna",
    "francella": "guillermo_francella",
    "darin": "ricardo_darin",
    "lali": "lali_esposito",
    "milei": "javier_milei",
    "barassi": "dario_barassi",
    "momo_benavides": "momo",
    "geronimo_benavides": "momo",
    "flavio_azzaro": "azzaro",
    "franco_colapinto": "colapinto",
    "lionel_scaloni": "scaloni",
}


def fijar_semillas(seed: int = SEED) -> None:
    """Fija todas las fuentes de azar que usamos (Python, NumPy, TensorFlow/Keras).

    También pide operaciones deterministas a TensorFlow. En GPU puede quedar una
    variación mínima entre corridas (algunas operaciones de cuDNN no son 100 %
    deterministas); se documenta en la bitácora en vez de esconderlo.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        import keras

        keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:  # versiones viejas de TF
            pass
    except ImportError:
        pass

"""Inferencia con las redes finales (E19) exportadas a ONNX.

Es lo único que necesita la aplicación: no usa TensorFlow ni el dataset. Las cuatro redes y el
manifiesto `modelos.json` se descargan del repositorio público de modelos de Hugging Face
(se generan en la celda 12.5 del notebook, verificadas contra Keras).

El preprocesamiento replica el de la celda 12.3 del notebook (`preparar_foto`): mismo recorte, mismo
redimensionado con PIL y la escala dentro de cada red. Por eso Colab, Streamlit y el frontend dan los
mismos porcentajes para la misma foto.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

REPO_MODELOS = "Franmiles/tp-g6-modelos"

# Orden en que se muestran los resultados: primero la CNN propia, después ResNet50 (transfer learning).
PANELES = [("cnn", "actor"), ("cnn", "emocion"), ("resnet", "actor"), ("resnet", "emocion")]


@dataclass
class Red:
    clave: str
    nombre: str
    tarea: str
    compite: bool
    lado: int
    entrada: str
    etiquetas: list[str]
    sesion: ort.InferenceSession


def _opciones() -> ort.SessionOptions:
    """Opciones para el servidor gratuito (2 núcleos compartidos): optimización completa del grafo y 2 hilos."""
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.intra_op_num_threads = 2
    return o


@lru_cache(maxsize=1)
def cargar_redes() -> dict[str, Red]:
    """Descarga (una sola vez, queda en caché) y abre las cuatro redes ONNX."""
    manifiesto = json.loads(open(hf_hub_download(REPO_MODELOS, "modelos.json")).read())
    redes = {}
    for clave, info in manifiesto["redes"].items():
        ruta = hf_hub_download(REPO_MODELOS, info["archivo"])
        redes[clave] = Red(clave=clave, nombre=info["nombre"], tarea=info["tarea"], compite=info["compite"],
                           lado=info["lado"], entrada=info["entrada"], etiquetas=info["etiquetas"],
                           sesion=ort.InferenceSession(ruta, sess_options=_opciones(), providers=["CPUExecutionProvider"]))
    return redes


def abrir_imagen(archivo) -> Image.Image:
    """Abre la foto en RGB y corrige la orientación de las fotos de celular (EXIF)."""
    return ImageOps.exif_transpose(Image.open(archivo)).convert("RGB")


def caja_recorte(ancho: int, alto: int, tamano: float = 1.0, centro_x: float = 0.5,
                 centro_y: float = 0.5) -> tuple[int, int, int, int]:
    """El cuadrado a recortar, (izq, arriba, der, abajo), sin salirse de la foto.

    `tamano` es el lado como fracción del lado corto de la foto; el centro va de 0 a 1. Lo usan el
    recorte y el recuadro que la app dibuja sobre la foto, así los dos coinciden siempre.
    """
    lado = max(1, round(min(ancho, alto) * tamano))
    izq = min(max(round(centro_x * ancho - lado / 2), 0), ancho - lado)
    arr = min(max(round(centro_y * alto - lado / 2), 0), alto - lado)
    return izq, arr, izq + lado, arr + lado


def recortar(img: Image.Image, tamano: float = 1.0, centro_x: float = 0.5, centro_y: float = 0.5) -> Image.Image:
    """Recorta un cuadrado alrededor de la cara.

    Por defecto (tamano=1, centro 0,5) es el cuadrado central más grande, igual que en el notebook.
    Las redes se entrenaron con recortes de rostro: cuanto más ocupe la cara, mejor.
    """
    return img.crop(caja_recorte(*img.size, tamano, centro_x, centro_y))


def preparar(recorte: Image.Image, lado: int) -> np.ndarray:
    """RGB en [0, 255], del tamaño de la red y con dimensión de lote: (1, lado, lado, 3)."""
    return np.asarray(recorte.resize((lado, lado), Image.BILINEAR), dtype="float32")[None]


def predecir(recorte: Image.Image, red: Red, top: int = 5) -> list[tuple[str, float]]:
    """Las `top` clases más probables de una red: [(etiqueta, probabilidad), ...]."""
    prob = red.sesion.run(None, {red.entrada: preparar(recorte, red.lado)})[0][0]
    orden = np.argsort(-prob)[:top]
    return [(red.etiquetas[i], float(prob[i])) for i in orden]


def frase(pred: list[tuple[str, float]], tarea: str, n: int = 3) -> str:
    """"con un 67 % de confianza es Francella, con un 30 % es Darín y con un 3 % es Colapinto." """
    verbo = "es" if tarea == "actor" else "expresa"
    partes = [f"con un {100 * pred[0][1]:.0f} % de confianza {verbo} {pred[0][0]}"]
    partes += [f"con un {100 * p:.0f} % " + ("es " if tarea == "actor" else "") + nombre for nombre, p in pred[1:n]]
    return (", ".join(partes[:-1]) + " y " + partes[-1] + ".") if len(partes) > 1 else partes[0] + "."

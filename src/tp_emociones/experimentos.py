"""Orquestación de experimentos: la misma receta para las dos tareas (identidad y emoción).

Cada función entrena con train, decide con val (early stopping) y devuelve todo lo
necesario para evaluar después. NO toca test: la evaluación en test se hace una sola vez,
al final del notebook, con `evaluar_en_test`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config, evaluacion, modelos, pipeline


@dataclass
class Experimento:
    nombre: str
    tarea: str
    modelo: object
    clases: list[str]
    datasets: dict
    ys: dict
    historias: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    val: dict = field(default_factory=dict)

    def evaluar_val(self, mostrar: bool = False) -> dict:
        prob = evaluacion.predecir(self.modelo, self.datasets["val"])
        self.val = evaluacion.reporte(self.ys["val"], prob, self.clases, f"{self.nombre} [{self.tarea}] VAL", mostrar)
        return self.val


def _preparar(df: pd.DataFrame, tarea: str, usar_pesos: bool):
    dsets, clases, ys = pipeline.datasets_por_split(df, tarea)
    pesos = pipeline.pesos_por_clase(ys["train"], len(clases)) if usar_pesos else None
    return dsets, clases, ys, pesos


def entrenar_cnn_propia(df: pd.DataFrame, tarea: str, epocas: int = 60, lr: float = 1e-3,
                        paciencia: int = 8, usar_pesos: bool = True, **kwargs_modelo) -> Experimento:
    dsets, clases, ys, pesos = _preparar(df, tarea, usar_pesos)
    m = modelos.cnn_propia(len(clases), nombre=f"cnn_propia_{tarea}", **kwargs_modelo)
    h = modelos.entrenar(m, dsets["train"], dsets["val"], lr=lr, epocas=epocas, paciencia=paciencia, pesos_clase=pesos)
    exp = Experimento("CNN propia", tarea, m, clases, dsets, ys, {"entrenamiento": h},
                      {"lr": lr, "epocas_max": epocas, "epocas_reales": len(h.history["loss"]),
                       **modelos.contar_parametros(m), **kwargs_modelo})
    exp.evaluar_val()
    return exp


def entrenar_resnet50(df: pd.DataFrame, tarea: str, epocas_f1: int = 20, lr_f1: float = 1e-3,
                      epocas_f2: int = 20, lr_f2: float = 1e-5, desde_capa: str = "conv5_block1_1_conv",
                      paciencia: int = 5, usar_pesos: bool = True, fase2: bool = True) -> Experimento:
    """Fase 1: base congelada, solo la cabeza (lr 1e-3). Fase 2: descongelar conv5 y
    reentrenar con lr 100 veces menor (1e-5). Orden y lr según NB15: al revés, los
    gradientes grandes de una cabeza aleatoria destruyen lo aprendido en ImageNet."""
    dsets, clases, ys, pesos = _preparar(df, tarea, usar_pesos)
    m, base = modelos.resnet50_transferencia(len(clases), nombre=f"resnet50_{tarea}")
    params_f1 = modelos.contar_parametros(m)
    h1 = modelos.entrenar(m, dsets["train"], dsets["val"], lr=lr_f1, epocas=epocas_f1, paciencia=paciencia, pesos_clase=pesos)
    exp = Experimento("ResNet50 fase 1 (congelada)", tarea, m, clases, dsets, ys, {"fase 1": h1},
                      {"lr_f1": lr_f1, "epocas_f1": len(h1.history["loss"]), "entrenables_f1": params_f1["entrenables"],
                       "total": params_f1["total"]})
    val_f1 = exp.evaluar_val()
    exp.params["val_f1"] = val_f1
    if fase2:
        n = modelos.descongelar_desde(base, desde_capa)
        h2 = modelos.entrenar(m, dsets["train"], dsets["val"], lr=lr_f2, epocas=epocas_f2, paciencia=paciencia, pesos_clase=pesos)
        exp.nombre = "ResNet50 fase 2 (fine-tuning conv5)"
        exp.historias["fase 2"] = h2
        exp.params.update({"lr_f2": lr_f2, "epocas_f2": len(h2.history["loss"]), "capas_descongeladas": n,
                           "entrenables_f2": modelos.contar_parametros(m)["entrenables"]})
        exp.evaluar_val()
    return exp


def tabla_val(experimentos: list[Experimento]) -> pd.DataFrame:
    return pd.DataFrame([{"tarea": e.tarea, "modelo": e.nombre, **{k: round(v, 3) for k, v in e.val.items()}}
                         for e in experimentos]).set_index(["tarea", "modelo"])


def evaluar_en_test(exp: Experimento, df: pd.DataFrame, mostrar: bool = True) -> dict:
    """ÚNICA evaluación en test (regla 3). Llamar una vez por modelo, al final."""
    prob = evaluacion.predecir(exp.modelo, exp.datasets["test"])
    m = evaluacion.reporte(exp.ys["test"], prob, exp.clases, f"{exp.nombre} [{exp.tarea}] TEST", mostrar)
    return {"metricas": m, "prob": prob, "rutas": df.loc[df["split"] == "test", "ruta"].to_numpy()}

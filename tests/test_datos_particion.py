"""Tests de las piezas que garantizan una evaluación honesta: nombres y partición.

Correr con:  python -m pytest tests/
No necesitan el dataset real: usan un registro sintético.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tp_emociones import datos, particion  # noqa: E402


def registro_sintetico():
    filas = []
    # actor con muchos videos, actor con 3 desparejos, actor con 2 videos
    for actor, videos in {"a": [10, 12, 8, 9, 11, 7], "b": [40, 30, 2], "c": [25, 20]}.items():
        for v, n in enumerate(videos):
            for f in range(n):
                filas.append({"clave": f"{actor}__v{v}__f{f:04d}.jpg", "actor": actor,
                              "emocion": ["alegria", "enojo", "neutral", "sorpresa", "tristeza"][f % 5],
                              "video": f"{actor}|v{v}"})
    return pd.DataFrame(filas)


def test_clave_archivo_decodifica_drive_y_tildes():
    assert datos.clave_archivo("PachuPe#U00f1a_Alegr#U00eda_G6_000001.jpg") == "pachupena_alegria_g6_000001.jpg"
    assert datos.clave_archivo("PachuPeña_Alegría_G6_000001.jpg") == "pachupena_alegria_g6_000001.jpg"


def test_normalizar_actor_unifica_variantes():
    assert datos.normalizar_actor("Guillermo Francella") == "guillermo_francella"
    assert datos.normalizar_actor("tini") == "tini_stoessel"


@pytest.mark.parametrize("nombre,video", [
    ("lali_esposito__v0902abc__f0000060.jpg", "v0902abc"),
    ("cande1_frame_000010.jpg", "cande1"),
    ("momo_video_1_p2_0003.jpg", "momo_video_1"),
])
def test_video_desde_nombre(nombre, video):
    assert datos.video_desde_nombre(nombre) == video


def test_particion_sin_fugas_y_completa():
    df = registro_sintetico()
    split = particion.particionar(df)
    particion.verificar_sin_fugas(df, split)  # no levanta excepción
    for actor in df["actor"].unique():
        assert set(split[df["actor"] == actor]) == {"train", "val", "test"}


def test_particion_determinista():
    df = registro_sintetico()
    assert particion.particionar(df).equals(particion.particionar(df))


def test_verificar_detecta_fuga():
    df = registro_sintetico()
    split = particion.particionar(df)
    video_train = df.loc[split == "train", "video"].iloc[0]
    fila = df.index[(df["video"] == video_train)][0]
    split.loc[fila] = "test"
    with pytest.raises(AssertionError):
        particion.verificar_sin_fugas(df, split)

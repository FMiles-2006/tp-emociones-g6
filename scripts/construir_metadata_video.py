"""Construye src/tp_emociones/recursos/video_por_imagen.csv.

Por qué existe: el CSV maestro del curso (etiquetas_master_final.csv) solo guarda
archivo;actor;emocion. Para partir train/val/test por video (regla de oro del enunciado)
necesitamos saber de qué video sale cada imagen. Esa información existe en los registros
que cada grupo subió a su carpeta (G1, G3, G4, G6, G7, G8) del Drive compartido.

Uso (sobre una copia descargada del Drive; NUNCA sobre el Drive montado con escritura):
    python scripts/construir_metadata_video.py --raiz "/ruta/a/la/copia/del/drive"

La raíz debe contener las carpetas G1, G3, G4, G6, G7, G8 tal como están en el Drive.
El resultado se versiona en el repo: así Colab no necesita las carpetas de los grupos.
Las imágenes que no figuren acá caen en el patrón del nombre de archivo
(ver datos.asignar_video), así que el dataset puede crecer sin romper nada.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tp_emociones.datos import RUTA_METADATA_VIDEO, clave_archivo  # noqa: E402

COLS_TXT = ["archivo", "actor", "emocion", "video", "escena"]


def leer_fuentes(raiz: Path) -> pd.DataFrame:
    partes = []

    def agregar(grupo: str, archivos, videos):
        partes.append(pd.DataFrame({"grupo": grupo, "archivo": list(archivos), "video": list(videos)}))

    # G3, G7, G8: etiquetas_master.txt sin encabezado: archivo;actor;emocion;video;escena
    for g in ("G3", "G7", "G8"):
        p = raiz / g / "etiquetas_master.txt"
        if p.exists():
            df = pd.read_csv(p, sep=";", header=None, names=COLS_TXT)
            agregar(g, df["archivo"], df["video"])

    # G1: registro completo con columna 'video'
    p = raiz / "G1" / "registro_grupo.csv"
    if p.exists():
        df = pd.read_csv(p)
        agregar("G1", df["archivo"], df["video"])

    # G4: columna 'video' (archivo en 'imagen')
    p = raiz / "G4" / "labels_224_final_grupo4.csv"
    if p.exists():
        df = pd.read_csv(p)
        agregar("G4", df["imagen"], df["video"])

    # G6 (nuestro grupo): 'video_origen' (archivo en 'imagen')
    p = raiz / "G6" / "DATASET_FINAL.csv"
    if p.exists():
        df = pd.read_csv(p, encoding="utf-8-sig").dropna(subset=["imagen", "video_origen"])
        # Los nombres de video de G6 son hashes larguísimos: se acortan sin perder unicidad.
        agregar("G6", df["imagen"], df["video_origen"].astype(str).str.slice(0, 24))

    if not partes:
        raise SystemExit(f"No se encontró ningún registro de grupo bajo {raiz}")
    out = pd.concat(partes, ignore_index=True)
    out["clave"] = out["archivo"].map(clave_archivo)
    out["video"] = out["video"].astype(str).str.replace(r"\.mp4$", "", regex=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", required=True, type=Path)
    ap.add_argument("--salida", type=Path, default=RUTA_METADATA_VIDEO)
    args = ap.parse_args()

    df = leer_fuentes(args.raiz)
    conflictos = df.groupby("clave")["video"].nunique()
    conflictos = conflictos[conflictos > 1]
    if len(conflictos):
        print(f"ATENCIÓN: {len(conflictos)} imágenes con más de un video declarado; se toma el primero.")
    df = df.drop_duplicates("clave", keep="first").sort_values("clave")

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    df[["clave", "video", "grupo"]].to_csv(args.salida, index=False)
    print(f"{len(df)} imágenes con video conocido -> {args.salida}")
    print(df.groupby("grupo").size().to_string())


if __name__ == "__main__":
    main()

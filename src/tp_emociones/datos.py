"""Carga, normalización y auditoría del dataset del curso.

Principio rector: el DATASET_FINAL del Drive del curso es la fuente de verdad y **no se
modifica nunca**. Este módulo solo lo lee y arma, en memoria, un "registro" (DataFrame)
con una fila por imagen utilizable:

    archivo | ruta | actor | emocion | video | origen_video

La columna `video` no está en el CSV del curso; se recupera (ver `asignar_video`) porque
es indispensable para la partición sin fugas (regla de oro del enunciado).
Si mañana se suman imágenes al Drive, este código las incorpora solo: nada de lo que hay
acá depende de conteos ni de nombres fijos.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd

from . import config

_RAIZ_PAQUETE = Path(__file__).resolve().parent
RUTA_METADATA_VIDEO = _RAIZ_PAQUETE / "recursos" / "video_por_imagen.csv"


# ---------------------------------------------------------------------------
# Normalización de texto
# ---------------------------------------------------------------------------
def clave_archivo(nombre: str) -> str:
    """Clave canónica de un nombre de archivo para poder cruzar fuentes.

    Resuelve tres problemas reales que encontramos en la auditoría:
    1. Las descargas en zip de Drive codifican los caracteres no ASCII como `#U00ed`
       (í) o `#U00f1` (ñ): se decodifican.
    2. El mismo nombre puede venir con o sin tilde ("Alegría" vs "Alegria"): se quitan.
    3. Mayúsculas/minúsculas y espacios en los bordes: se normalizan.
    """
    s = re.sub(r"#U([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), str(nombre))
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()


def normalizar_etiqueta(texto: str) -> str:
    """'Alegría ' -> 'alegria'; 'Guillermo Francella' -> 'guillermo_francella'."""
    s = clave_archivo(texto)
    s = re.sub(r"[\s\-_]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s.strip("_")


def normalizar_actor(nombre: str) -> str:
    actor = normalizar_etiqueta(nombre)
    return config.MAPA_ACTORES.get(actor, actor)


# ---------------------------------------------------------------------------
# Video de origen de cada imagen
# ---------------------------------------------------------------------------
_PATRONES_VIDEO = (
    re.compile(r"^.+?__(?P<video>.+?)__f\d+\.jpe?g$"),  # G1/G10: actor__video__f0000123.jpg
    re.compile(r"^(?P<video>.+?)_frame_\d+\.jpe?g$"),  # G4: Cande1_frame_000010.jpg
    # G7: momo_video_2_0012.jpg / momo_video_1_p2_0003.jpg ("_p2" = parte del mismo video:
    # se agrupa con el video para no partir un mismo clip entre train y test).
    re.compile(r"^(?P<video>.+?)(?:_p\d+)?_\d+\.jpe?g$"),
)


def video_desde_nombre(clave: str) -> str | None:
    for patron in _PATRONES_VIDEO:
        m = patron.match(clave)
        if m:
            return m.group("video")
    return None


def cargar_metadata_video(ruta: Path | str = RUTA_METADATA_VIDEO) -> dict[str, str]:
    """Mapa clave_archivo -> video, construido a partir de los registros de cada grupo.

    Se genera con scripts/construir_metadata_video.py y se versiona en el repo.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        return {}
    df = pd.read_csv(ruta)
    return dict(zip(df["clave"], df["video"]))


def asignar_video(claves: pd.Series, metadata: dict[str, str]) -> tuple[pd.Series, pd.Series]:
    """Devuelve (video, origen_video) para cada imagen.

    Orden de prioridad:
      1. 'registro': el registro por imagen que entregó cada grupo (lo más confiable).
      2. 'nombre':   el patrón del nombre de archivo, cuando codifica el video.
      3. 'prefijo':  último recurso conservador: el nombre sin números. Agrupa de más
                     (varios videos podrían caer juntos), lo que nunca genera fugas;
                     a lo sumo hace la partición menos fina.
    """
    videos, origenes = [], []
    for c in claves:
        if c in metadata:
            videos.append(metadata[c]); origenes.append("registro")
            continue
        v = video_desde_nombre(c)
        if v is not None:
            videos.append(v); origenes.append("nombre")
        else:
            videos.append(re.sub(r"\d+", "", Path(c).stem).strip("_")); origenes.append("prefijo")
    return pd.Series(videos, index=claves.index), pd.Series(origenes, index=claves.index)


# ---------------------------------------------------------------------------
# Registro
# ---------------------------------------------------------------------------
def leer_csv_master(ruta_csv: Path | str) -> pd.DataFrame:
    """Lee el CSV del curso tolerando separador ';' o ',' y BOM."""
    df = pd.read_csv(ruta_csv, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    faltan = {"archivo", "actor", "emocion"} - set(df.columns)
    if faltan:
        raise ValueError(f"Al CSV le faltan columnas: {faltan}")
    return df


def construir_registro(ruta_dataset: Path | str, verbose: bool = True) -> pd.DataFrame:
    """Arma el registro de imágenes utilizables a partir de la carpeta DATASET_FINAL.

    Descarta (y reporta) filas sin imagen física, emociones fuera del vocabulario y
    duplicados de archivo. Nunca escribe en `ruta_dataset`.
    """
    ruta_dataset = Path(ruta_dataset)
    df = leer_csv_master(ruta_dataset / config.CSV_MASTER)
    carpeta = ruta_dataset / config.CARPETA_IMAGENES

    # Índice clave -> ruta real de las imágenes presentes en disco.
    fisicas = {clave_archivo(p.name): p for p in carpeta.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}

    df = df.assign(
        clave=df["archivo"].map(clave_archivo),
        actor=df["actor"].map(normalizar_actor),
        emocion=df["emocion"].map(normalizar_etiqueta),
    )
    n0 = len(df)
    reporte = {"filas_csv": n0, "imagenes_en_carpeta": len(fisicas)}

    df = df[df["emocion"].isin(config.EMOCIONES)]
    reporte["descartadas_emocion_invalida"] = n0 - len(df)

    n1 = len(df)
    df = df.drop_duplicates(subset="clave", keep="first")
    reporte["descartadas_duplicadas"] = n1 - len(df)

    n2 = len(df)
    df = df[df["clave"].isin(fisicas)]
    reporte["descartadas_sin_imagen"] = n2 - len(df)
    reporte["imagenes_sin_fila_en_csv"] = len(set(fisicas) - set(df["clave"]))

    df = df.assign(ruta=df["clave"].map(lambda c: str(fisicas[c])))
    df["video"], df["origen_video"] = asignar_video(df["clave"], cargar_metadata_video())
    # El video se califica con el actor: dos grupos pueden tener un "video1".
    df["video"] = df["actor"] + "|" + df["video"].astype(str)

    df = df.sort_values("clave").reset_index(drop=True)
    reporte["imagenes_utilizables"] = len(df)
    df.attrs["reporte"] = reporte
    df.attrs["huella"] = huella_dataset(df)

    if verbose:
        print("Auditoría de carga")
        for k, v in reporte.items():
            print(f"  {k:32s} {v:>6}")
        print(f"  {'huella (versión) del dataset':32s} {df.attrs['huella']}")
        print("  origen del video:", df["origen_video"].value_counts().to_dict())
    salida = df[["archivo", "clave", "ruta", "actor", "emocion", "video", "origen_video"]].copy()
    salida.attrs = dict(df.attrs)
    return salida


def huella_dataset(df: pd.DataFrame) -> str:
    """Identificador corto de la versión del dataset (qué imágenes y con qué etiquetas).

    Si alguien agrega o corrige imágenes en el Drive, la huella cambia: así cualquier
    resultado queda atado a la versión de datos con la que se obtuvo.
    """
    contenido = "\n".join(sorted(df["clave"] + ";" + df["actor"] + ";" + df["emocion"]))
    return hashlib.sha256(contenido.encode()).hexdigest()[:12]


def tabla_distribucion(df: pd.DataFrame) -> pd.DataFrame:
    """Conteo actor x emoción con totales (la tabla que pide el enunciado)."""
    return pd.crosstab(df["actor"], df["emocion"], margins=True, margins_name="Total")

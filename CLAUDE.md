# CLAUDE.md — reglas del proyecto (leer antes de tocar cualquier cosa)

Proyecto: TP de IAyAA II (UCA Rosario 2026), Grupo 6. Dos tareas sobre recortes de rostro
de actores argentinos: **identidad** (20 clases hoy) y **emoción** (5 clases:
alegria, enojo, neutral, sorpresa, tristeza). Todo corre en Google Colab.

## Reglas de oro (no negociables)
1. **El Drive del curso es de solo lectura.** Ningún código escribe, mueve ni borra nada en
   `CARPETA COMPARTIDA PARA TODOS`. El notebook copia DATASET_FINAL a `/content/datos`
   y trabaja sobre la copia. Si una tarea parece requerir escribir ahí: parar y preguntar.
2. **Partición por video, nunca por imagen.** Se usa `particion.particionar` y después
   `particion.verificar_sin_fugas`. No crear particiones "a mano" en ningún notebook.
3. **El test se evalúa una sola vez por entrega**, al final. Toda decisión (épocas, lr,
   capas a descongelar, aumentación) se toma contra validación.
4. **Reproducibilidad:** `config.fijar_semillas()` al inicio de cada notebook y antes de
   construir cada modelo. Toda constante que afecte resultados vive en `config.py`.
5. **Sin código duplicado:** si algo se usa en dos notebooks, va a `src/tp_emociones/`.
   Los notebooks orquestan y explican; la lógica vive en el paquete.
6. **Toda decisión se registra** en `docs/BITACORA.md` (formato D-NNN: qué, por qué,
   alternativas descartadas, quién, fecha). Si cambiás una decisión, agregá una entrada
   nueva que referencie la vieja; no borres historia.
7. **Nunca se versionan imágenes, zips ni modelos** (ver .gitignore). El dataset no se
   publica fuera del curso.

## Estructura
- `src/tp_emociones/config.py` — semilla, clases, tamaños, fracciones.
- `src/tp_emociones/datos.py` — lectura del CSV del curso, normalización de nombres,
  recuperación del video de origen, huella (versión) del dataset.
- `src/tp_emociones/particion.py` — partición train/val/test por video + verificación.
- `src/tp_emociones/baselines.py` — trivial, RF y MLP sobre píxeles; acta de límites (NB02).
- `src/tp_emociones/pipeline.py` — tf.data (patrón del curso) + aumentación para rostros.
- `src/tp_emociones/modelos.py` — CNN propia y ResNet50 (transferencia en 2 fases).
- `src/tp_emociones/evaluacion.py` — métricas, matriz de confusión, errores, oclusión.
- `src/tp_emociones/recursos/video_por_imagen.csv` — generado por
  `scripts/construir_metadata_video.py` desde los registros de cada grupo.
- `notebooks/` — un notebook por bloque de la entrega; importan el paquete.

## Convenciones
- Código y comentarios en español. Funciones con docstring que explique el *por qué*.
- Cada modelo nuevo: función constructora en `modelos.py` que llame a `fijar_semillas`.
- Métricas que se reportan siempre: accuracy y **F1 macro** (las clases están desbalanceadas)
  + matriz de confusión. Para emoción, recordar el techo de acuerdo entre etiquetadores.
- Antes de commitear: correr `python -m pytest tests/` (tests de datos y partición).
- Cuando trabajes con Claude Code: un cambio lógico por commit, mensaje en español.

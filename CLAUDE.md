# CLAUDE.md — reglas del proyecto (leer antes de tocar cualquier cosa)

Proyecto: TP de IAyAA II (UCA Rosario 2026), Grupo 6. Dos tareas sobre recortes de rostro
de actores argentinos: **identidad** (20 clases hoy) y **emoción** (5 clases:
alegria, enojo, neutral, sorpresa, tristeza). Todo corre en Google Colab.

## Reglas de oro (no negociables)
1. **El Drive del curso es de solo lectura.** Ningún código escribe, mueve ni borra nada en
   `CARPETA COMPARTIDA PARA TODOS`. El notebook copia DATASET_FINAL a `/content/datos`
   y trabaja sobre la copia. Si una tarea parece requerir escribir ahí: parar y preguntar.
2. **Partición por video, nunca por imagen.** Se usa `particionar` y después
   `verificar_sin_fugas` (sección 3 del notebook). No crear particiones "a mano" en ningún notebook.
3. **El test se evalúa una sola vez por entrega**, al final. Toda decisión (épocas, lr,
   capas a descongelar, aumentación) se toma contra validación.
4. **Reproducibilidad:** `fijar_semillas()` al inicio y antes de construir cada modelo.
   Toda constante que afecte resultados vive en la celda 1.1 del notebook.
5. **Sin código duplicado:** cada función se define una sola vez en el notebook y se reutiliza
   para las dos tareas (identidad y emoción).
6. **Toda decisión se registra** en `docs/BITACORA.md` (formato D-NNN: qué, por qué,
   alternativas descartadas, quién, fecha). Si cambiás una decisión, agregá una entrada
   nueva que referencie la vieja; no borres historia.
7. **Nunca se versionan imágenes, zips ni modelos** (ver .gitignore). El dataset no se
   publica fuera del curso.

## Estructura
- `notebooks/TP2_Grupo6.ipynb` — **todo el código de la entrega**, autocontenido, en secciones:
  entorno → datos → partición → baselines → acta de límites → pipeline + alineación de rostros →
  CNN propia (v1 control, v2 multitarea) → ResNet50 → test → oclusión → conclusiones → competencia. Cada función se define una sola vez, en la
  sección donde se usa por primera vez.
- `docs/BITACORA.md` — registro de decisiones.
- GitHub es control de versiones: el notebook se edita en Colab y se guarda en el repo con
  "Archivo → Guardar una copia en GitHub".

## Convenciones
- Código y comentarios en español. Funciones con docstring que explique el *por qué*.
- Cada modelo nuevo: función constructora en el notebook que llame a `fijar_semillas`.
- Métricas que se reportan siempre: accuracy y **F1 macro** (las clases están desbalanceadas)
  + matriz de confusión. Para emoción, recordar el techo de acuerdo entre etiquetadores.
- Antes de guardar en GitHub: "Ejecutar todas" con GPU T4 y verificar que corre de punta a punta.
- Cuando trabajes con Claude Code: un cambio lógico por commit, mensaje en español.
- La competencia del curso es con la **CNN propia desde cero** (sin pesos preentrenados) sobre fotos
  cualquiera: toda mejora a la CNN tiene que pasar por `preparar_cara` (detección + alineación).

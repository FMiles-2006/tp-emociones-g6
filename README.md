# TP IAyAA II — Identidad y emoción de actores argentinos (Grupo 6)

UCA Rosario · Licenciatura en Ciencia de Datos · 2026
Martín Masuelli · Francisco Miles · Tomás Lizazu · Bautista Verna · Juan Diego Ancarani · Manuel Lafosse

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FMiles-2006/tp-emociones-g6/blob/main/notebooks/TP2_Grupo6.ipynb)

## Cómo correr el notebook de la entrega
1. En Google Drive: **Compartido conmigo → `CARPETA COMPARTIDA PARA TODOS` → clic derecho →
   Organizar → Agregar acceso directo → Todas las ubicaciones → Mi unidad**.
   (Elegí *Mi unidad*: no agregues el acceso dentro de carpetas del curso.)
2. Abrí el notebook con el botón **Abrir en Colab** de arriba.
3. **Entorno de ejecución → Cambiar tipo de entorno → GPU T4**. Sin GPU, ResNet50 tarda horas.
4. **Entorno de ejecución → Ejecutar todas** y aceptá el permiso de Drive. El Drive se monta
   en **solo lectura**: el notebook no puede modificar nada del curso.

Duración aproximada con GPU T4: 30–40 minutos.

## Qué hay en el repo
| Ruta | Contenido |
|---|---|
| `notebooks/TP2_Grupo6.ipynb` | Notebook de la entrega, de punta a punta |
| `src/tp_emociones/` | Toda la lógica (datos, partición, baselines, pipeline, modelos, evaluación) |
| `docs/BITACORA.md` | Registro de **todas** las decisiones con su porqué |
| `CLAUDE.md` | Reglas del proyecto (las lee Claude Code de cada integrante) |
| `tests/` | Tests de la partición sin fugas (`python -m pytest tests/`) |
| `scripts/construir_metadata_video.py` | Cómo se recuperó el video de origen de cada imagen |

## Cómo trabajar en grupo
- Cambios a modelos/hiperparámetros: en `src/tp_emociones/` (no copiar código en notebooks).
- Toda decisión nueva: una entrada en `docs/BITACORA.md`.
- Un cambio lógico por commit; antes de subir, correr los tests.
- Nunca se suben imágenes, zips ni modelos entrenados (ver `.gitignore`).

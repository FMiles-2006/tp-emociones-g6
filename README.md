# TP IAyAA II — Identidad y emoción de actores argentinos (Grupo 6)

UCA Rosario · Licenciatura en Ciencia de Datos · 2026
Martín Masuelli · Francisco Miles · Tomás Lizazu · Bautista Verna · Juan Diego Ancarani · Manuel Lafosse

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FMiles-2006/tp-emociones-g6/blob/cnn-v2/notebooks/TP_Grupo6_Final.ipynb)

## Cómo correr el notebook de la entrega
1. En Google Drive: **Compartido conmigo → `CARPETA COMPARTIDA PARA TODOS` → clic derecho →
   Organizar → Agregar acceso directo → Todas las ubicaciones → Mi unidad**.
   (Elegí *Mi unidad*: no agregues el acceso dentro de carpetas del curso.)
2. Abrí el notebook con el botón **Abrir en Colab** de arriba.
3. **Entorno de ejecución → Cambiar tipo de entorno → GPU T4**. Sin GPU, ResNet50 tarda horas.
4. **Entorno de ejecución → Ejecutar todas** y aceptá el permiso de Drive. El Drive se monta
   en **solo lectura**: el notebook no puede modificar nada del curso.

Duración aproximada con GPU T4: unos 45 minutos (el notebook completo, unas 2 horas).

## Qué hay en el repo
| Ruta | Contenido |
|---|---|
| `notebooks/TP_Grupo6_Final.ipynb` | **Notebook de la entrega**: CNN propia final + ResNet50, partición, test y predicción de fotos nuevas |
| `notebooks/TP2_Grupo6.ipynb` | Notebook completo con todo el camino: v1, búsqueda con KerasTuner, anexo con versiones anteriores |
| `docs/BITACORA.md` | Registro de **todas** las decisiones con su porqué |
| `docs/EXPERIMENTOS_CNN.md` | Cada configuración de la CNN propia que se probó y su resultado (insumo del informe) |
| `CLAUDE.md` | Reglas del proyecto (las lee Claude Code de cada integrante) |

## Cómo trabajar en grupo
- El código se edita en Colab y se guarda acá con "Archivo → Guardar una copia en GitHub".
- Toda decisión nueva: una entrada en `docs/BITACORA.md`.
- Un cambio lógico por commit; antes de guardar, "Ejecutar todas" con GPU.
- Nunca se suben imágenes, zips ni modelos entrenados (ver `.gitignore`).

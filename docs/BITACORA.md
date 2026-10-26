# Bitácora de decisiones — Grupo 6

Cada entrada: **qué** se decidió, **por qué**, **alternativas descartadas**, **evidencia**,
quién y cuándo. Las decisiones no se borran: si cambian, se agrega una entrada nueva.

---

## D-001 · Vocabulario de 5 emociones (sin Miedo) — 2026-10 (curso)
- **Qué:** alegria, enojo, neutral, sorpresa, tristeza.
- **Por qué:** decisión de todo el curso. En nuestro avance del TP1 Miedo tenía 26
  imágenes (3,9 %) y ninguna de Oscar Martínez: no alcanzaba para aprender la clase.
- **Descartado:** sexta clase Miedo.

## D-002 · El DATASET_FINAL del curso es la fuente de verdad y es de solo lectura — 2026-10-26
- **Qué:** trabajamos con lo que hay en `DATASET_FINAL` (hoy 2.723 imágenes, 20 actores).
  El notebook lo copia a disco local de Colab y nunca escribe en el Drive del curso.
- **Por qué:** lo administra otro compañero; tocarlo puede romper el trabajo de otros
  grupos. Además, trabajar sobre una copia hace la ejecución más rápida.
- **Cómo se mantiene reproducible si se agregan imágenes:** el código no depende de
  conteos ni de nombres fijos; cada corrida calcula una **huella** (hash) del dataset y la
  imprime. Todo resultado queda atado a la huella con que se obtuvo. Huella actual:
  `512b8d7c66ec`.
- **Limitaciones conocidas del dataset oficial (para la ficha):**
  - Pachu Peña (122 imágenes, G6) no entró al FINAL: su nombre de archivo tiene "ñ".
  - Rodrigo de la Serna (G6) no tiene imágenes subidas.
  - Grupos 2, 5 y 9 no aportaron imágenes.
  - Pendiente avisar al responsable del Drive; si se corrige, el código lo incorpora solo.

## D-003 · Recuperar el video de origen de cada imagen — 2026-10-26
- **Qué:** el CSV del curso solo tiene `archivo;actor;emocion`. Reconstruimos el video
  de cada imagen desde los registros que subió cada grupo (G1, G3, G4, G6, G7, G8) y,
  si no está ahí, desde el patrón del nombre de archivo (`actor__video__f123.jpg`,
  `Cande1_frame_10.jpg`, ...). Último recurso: el nombre sin números (agrupa de más,
  nunca genera fugas).
- **Evidencia:** 2.723 / 2.723 imágenes con video (2.287 desde registro, 436 desde nombre).
- **Detalle técnico:** los zips de Drive codifican la ñ y las tildes como `#U00f1`; los
  nombres se cruzan por una clave normalizada (`datos.clave_archivo`).

## D-004 · Partición por video, por actor, ~70/15/15 — 2026-10-26
- **Qué:** reglas R1–R5 en `src/tp_emociones/particion.py`. Resumen: ningún video de
  train aparece en val/test (sin excepciones); cada actor está en los tres splits; los
  videos se reparten buscando 70 % de imágenes en train.
- **Por qué:** regla de oro del enunciado. Encontramos ~1.500 pares de cuadros casi
  idénticos dentro de un mismo video: partir por imagen inflaría el resultado.
- **Concesión (R4):** en 9 actores con pocos videos o muy desparejos (azzaro, colapinto,
  dario_barassi, georgina_barbarossa, javier_milei, julio_chavez, momo, mora_bianchi,
  ricardo_darin) val y test comparten **un** video, cortado en mitades contiguas. Train
  sigue sin compartir nada. Alternativa descartada: dejar a esos actores sin val o sin
  test (no se podría medir identidad para ellos).
- **Resultado (huella 512b8d7c66ec):** train 1.869 (68,6 %) · val 427 (15,7 %) · test 427
  (15,7 %); 20 actores en cada split; distribución de emociones similar entre splits.

## D-005 · Baselines: trivial, Random Forest y MLP sobre píxeles 64×64 en gris — 2026-10-26
- **Por qué 64×64 gris:** 224×224×3 = 150.528 columnas para ~1.900 imágenes no
  generaliza y es lento. Misma decisión de la diapositiva 15 del TP1.
- **Resultados en VALIDACIÓN** (huella 512b8d7c66ec, semilla 42):

  | Tarea | Trivial acc / F1m | RF acc / F1m | MLP acc / F1m |
  |---|---|---|---|
  | Identidad (20) | 0,148 / 0,013 | 0,333 / 0,310 | 0,290 / 0,235 |
  | Emoción (5)    | 0,260 / 0,083 | 0,293 / 0,234 | 0,314 / 0,266 |

- **Acta de límites (NB02), permutación de píxeles con el MLP:** identidad 0,290 → 0,321;
  emoción 0,314 → 0,297. Mezclar los píxeles no cambia el resultado: el MLP no usa la
  estructura espacial. Motiva las CNN (D-006).

## Pendientes
- D-006 CNN propia (arquitectura, aumentación) — a cargo de: _completar_
- D-007 ResNet50 con transferencia en 2 fases — a cargo de: _completar_
- D-008 Evaluación final en test (una sola vez) + oclusión

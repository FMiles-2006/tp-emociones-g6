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

## D-006 · Pipeline tf.data y aumentación como capas del modelo — 2026-10-26
- **Qué:** patrón del curso `map → cache → shuffle (solo train) → batch → prefetch`, con las
  imágenes cacheadas en uint8 (~400 MB). Escalado y aumentación van **dentro del modelo**.
- **Por qué:** cachear en float32 cuadruplica la RAM sin beneficio; con el preprocesamiento
  adentro, el modelo guardado no puede ser alimentado distinto en inferencia (clase 17).
- **Aumentación elegida (válida para rostros):** flip horizontal, rotación ±8°, zoom ±10 %,
  traslación ±5 %, brillo y contraste ±15 %.
- **Descartado:** flip vertical (caras imposibles), rotaciones grandes y recortes agresivos
  (tapan cejas/boca, que es la señal de la emoción).
- **Desbalance:** pesos por clase "balanced" en la pérdida + F1 macro como métrica.

## D-007 · CNN propia desde cero (experimento de control) — 2026-10-26
- **Arquitectura:** 4 bloques [Conv3×3 → Conv3×3 → BatchNorm → MaxPool] con filtros
  32/64/128/256, GlobalAveragePooling, Dense(128), Dropout(0,5), softmax. L2 = 1e-4.
  ~1,2 M parámetros. Adam 1e-3, hasta 60 épocas, EarlyStopping(val_loss, paciencia 8,
  restaura mejores pesos) + ReduceLROnPlateau.
- **Por qué GAP y no Flatten:** con 224×224, Flatten daría ~50.000 entradas a la densa
  (millones de parámetros para ~1.900 imágenes).
- **Experimento de permutación (NB11):** misma CNN con píxeles permutados; si su desempeño
  se desploma, prueba que usa la estructura espacial (al revés que el MLP, D-005).
- **Resultados:** pendientes de la corrida con GPU.

## D-008 · ResNet50 con transferencia en dos fases — 2026-10-26
- **Fase 1:** base de ImageNet congelada (`training=False`), cabeza GAP → Dropout(0,3) →
  softmax. Adam 1e-3, hasta 20 épocas. Entrenables: 40.980 (identidad) / 10.245 (emoción).
- **Fase 2:** se descongela el bloque conv5 (BatchNorm sigue congelada), se recompila con
  Adam 1e-5 (100× menor), hasta 20 épocas. Entrenables: ~15,0 M de 23,6 M.
- **Por qué este orden (NB15):** la cabeza aleatoria genera gradientes grandes que, con la
  base libre, destruyen lo aprendido en ImageNet.
- **Preprocesamiento:** `resnet50.preprocess_input` como capa dentro del modelo (NB13).
- **Resultados:** pendientes de la corrida con GPU.

## D-009 · Prueba de humo en Colab (CPU) — 2026-10-26
- El notebook completo corrió de punta a punta en Colab (TF 2.21, Keras 3.13,
  scikit-learn 1.6.1) sobre un subconjunto de 360 imágenes y 1 época por fase: sin errores.
  Huella del dataset en Colab = local (`512b8d7c66ec`): la carga es reproducible.
- **Bug encontrado y corregido:** scikit-learn 1.6 falla en `MLPClassifier(early_stopping=True)`
  con etiquetas de texto → los baselines usan etiquetas enteras.
- **Limitación operativa:** sin GPU, ResNet50 tarda ~15 s por lote en CPU (~13 min por
  época con el train completo). La corrida final exige GPU T4.

## D-010 · El código vive en el notebook de Colab; GitHub solo versiona — 2026-10-27
- **Qué:** el notebook `notebooks/TP2_Grupo6.ipynb` es autocontenido: todas las funciones
  están escritas en sus celdas, cada una en la sección donde se usa (definidas una sola vez).
  Se eliminó el paquete `src/tp_emociones` (queda en el historial de git).
- **Por qué:** la entrega se presenta y se defiende en Colab; un notebook que clona código
  externo no se puede leer ni defender celda por celda. GitHub se usa como control de
  versiones (volver atrás, ver qué cambió y quién).
- **Cambio respecto de D-003:** el video de origen ya no sale de un CSV versionado en el
  repo, sino que el notebook lo lee directamente (en solo lectura) de los registros de cada
  grupo en el Drive del curso.
- **Reemplaza:** la regla "la lógica vive en el paquete" del CLAUDE.md original.

## D-011 · Primera corrida con GPU: baselines, acta y una falla de la CNN corregida — 2026-10-27
Corrida en Colab con GPU T4 (TF 2.21 · Keras 3.13 · scikit-learn 1.6.1), huella `512b8d7c66ec`,
partición: train 1.869 / val 428 / test 426. Todos los números de esta entrada son de **validación**.

**Baselines (D-005, recalculados en Colab):**

| Tarea | Trivial acc / F1m | RF acc / F1m | MLP acc / F1m |
|---|---|---|---|
| Identidad | 0,147 / 0,013 | **0,334 / 0,277** | 0,280 / 0,218 |
| Emoción | 0,238 / 0,077 | 0,255 / 0,167 | **0,276 / 0,252** |

**Acta de límites:** MLP de 263 mil a 4,7 M parámetros → 19× más tiempo (1,0 s → 19,4 s) y
la misma accuracy (27,8 %). Permutación de píxeles: identidad 28,0 % → 29,2 %; emoción
27,6 % → 26,6 %. El MLP no usa la estructura espacial.

**Falla de la CNN propia (intento 1) — qué no funcionó:** identidad con 75 % en train y
**4,4 %** en validación (peor que el trivial); val_loss subiendo desde la época 1.
- Diagnóstico: BatchNormalization con momentum 0,99 → los promedios móviles que usa en
  inferencia tardan muchas épocas en estabilizarse (59 pasos por época); el early stopping
  (paciencia 8) cortó en la época 9 y restauró los pesos de la época 1.
- Se interrumpió la corrida **antes de la sección 9**: el test no se usó.
- Corrección (decidida contra validación): `BatchNormalization(momentum=0.9)` y
  `EarlyStopping(start_from_epoch=10, patience=10)` para la CNN (ResNet50 sin cambios).

**CNN propia (intento 2), identidad:** val accuracy 0,192 · F1 macro 0,175 (1,21 M
parámetros, 25 épocas). Train llega a ~89 %: sobreajuste fuerte. Con ~90 imágenes por actor
de 2 a 10 videos, la red desde cero aprende rasgos de cada video que no se repiten en
validación. **No supera al Random Forest** — resultado esperado del experimento de control
(NB14) y argumento a favor de la transferencia.

## Pendientes
- Corrida completa con GPU y completar resultados de D-007 / D-008.
- D-010 Evaluación final en test (una sola vez) + oclusión + conclusiones.
- Avisar al responsable del Drive sobre Pachu Peña (ñ) y Rodrigo de la Serna.

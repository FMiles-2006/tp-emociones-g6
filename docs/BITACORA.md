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

## D-012 · Comparación final, evaluación única en test y conclusiones — 2026-10-27
Misma corrida T4 que D-011 (huella `512b8d7c66ec`). Completa los resultados de D-007 y D-008.

**Validación (acc / F1 macro):**

| Modelo | Identidad | Emoción |
|---|---|---|
| CNN propia | 0,192 / 0,175 | 0,255 / 0,247 |
| ResNet50 fase 1 (base congelada) | 0,502 / 0,430 | **0,428 / 0,412** |
| ResNet50 fase 2 (conv5 descongelado) | **0,614 / 0,521** | 0,425 / 0,409 |

- Modelo final elegido **por F1 macro en validación** (regla 3 del CLAUDE.md). En identidad gana
  la fase 2. En emoción la fase 2 no mejora (0,412 → 0,409, diferencia dentro del ruido); el
  notebook evalúa la fase 2 para ambas tareas, por lo que el test de emoción corresponde a fase 2.
- Prueba de permutación aplicada a la CNN: identidad 0,192 → 0,150 en val, pero 77 % en train
  con píxeles permutados → la CNN memoriza.

**Test (evaluado una sola vez, sección 9):**

| Modelo | Identidad acc / F1m | Emoción acc / F1m |
|---|---|---|
| CNN propia | 0,190 / 0,171 | 0,232 / 0,221 |
| ResNet50 fase 2 | **0,589 / 0,515** | **0,383 / 0,371** |

- Emoción por clase (F1): alegría 0,63 · enojo 0,31 · neutral 0,34 · sorpresa 0,32 · tristeza 0,25.
  La validación (42,5 %) coincide con el acuerdo entre etiquetadores de TP1 (42,6 %): el techo
  lo ponen las etiquetas, no el modelo.
- Identidad, pares más confundidos: javier_milei→tini_stoessel 20 · ricardo_darin→pablo_echarri 15 ·
  dolores_fonzi→mora_bianchi 14 · tini_stoessel→lali_esposito 14 · eva_de_dominici→tini_stoessel 9.
  julio_chavez y ricardo_darin con recall 0 en test (ambos con R4: un solo video en val/test).
- Oclusión (sección 10): en identidad parte de la evidencia cae fuera de la cara (micrófono,
  mano, fondo, esquinas) → atajos de la fuente. En emoción, alegría se decide por la boca; en
  Milei la evidencia se reparte e incluye el zócalo de TV.

**Conclusiones** (volcadas en la sección 11 del notebook): transferencia es la decisión que más
pesa; fine-tuning sólo ayuda en identidad; emoción está en el techo de las etiquetas; identidad
usa atajos del video. Limitaciones: pocos videos por actor, R4 en 9 actores, faltan Pachu Peña y
Rodrigo de la Serna, recortes no homogéneos, una sola semilla, un etiquetador por imagen.

**Quién:** Francisco Miles (con Claude). **Alternativas no exploradas por tiempo:** re-recorte con
detector de rostros, EfficientNet/MobileNet, cabeza multitarea, promedio por frames vecinos.

## D-013 · Se mejora la CNN propia: la que compite es una red desde cero — 2026-10-27
**Contexto:** en la presentación cada grupo predice actor y emoción de fotos nuevas con su CNN
**desde cero** (sin pesos preentrenados); el profesor habilitó cambiar hiperparámetros y
arquitectura. ResNet50 queda como referencia, no compite.
**Qué:** CNN propia v2: bloques residuales (~1,24 M parámetros, igual presupuesto que la v1),
**una red con dos salidas** (actor y emoción), aumentación fuerte (gris y borrado aleatorio),
label smoothing 0,1, AdamW con tasa coseno y calentamiento, 120 épocas fijas, TTA con espejo.
**Por qué:** la v1 memoriza el video (train 89 %, val 19 %); cada cambio ataca el sobreajuste.
Multitarea porque la competencia pide las dos cosas de cada foto.
**Resultado (val):** v2 con rostro alineado 0,292 / 0,334; v2 con recorte original 0,357 / 0,304
(identidad / emoción). La v1 daba 0,192 / 0,255. Detalle en `docs/EXPERIMENTOS_CNN.md` (E1–E3).
**Quién:** Francisco Miles (con Claude).

## D-014 · Entrada de la CNN: recorte original del curso, no rostro alineado — 2026-10-27
**Qué:** la CNN v2 se entrena con el recorte del curso reducido a 112×112. La detección de rostro
(YuNet) se usa para encuadrar las fotos nuevas igual que el curso: cuadrado centrado en la cara,
de lado = ancho de la cara / 0,527 (mediana medida sobre los 2.723 recortes del curso).
**Por qué:** búsqueda E4–E8 en validación: alinear con la plantilla de 5 puntos ajustada (0,85)
mejora emoción (+3 puntos) pero baja identidad (−6,5); con más margen (0,65 / 0,55) empeora
todo porque los recortes del curso no tienen más imagen alrededor y el relleno de bordes mete
rayas. Con el recorte original la identidad es la mejor en todas las comparaciones.
**Alternativas descartadas:** alineación 0,85 (pierde pelo y contorno de la cabeza), 0,65 y 0,55
(bordes artificiales). Se reserva la alineación ajustada como opción para una red de emoción.
**Riesgo declarado:** parte de la ventaja del recorte original puede venir de fondos parecidos
entre videos del mismo canal, que no se repiten en fotos nuevas.
**Quién:** Francisco Miles (con Claude).

## D-015 · La CNN final usa solo técnicas vistas en la cátedra (v3) — 2026-10-27
**Qué:** la CNN que se entrega y compite (v3) usa solo herramientas del programa: bloques
residuales con `Add` (NB13), dos salidas con la API Functional (clase 10), capas de aumentación
(NB03/NB16, clase 17), Dropout + L2 + EarlyStopping (NB07), pesos por clase (NB08), Adam (NB06) y
**búsqueda de hiperparámetros con KerasTuner** (NB09). También se quita ReduceLROnPlateau de la v1 y
de ResNet50. Lo que no se vio (D-013/D-014: alineación de rostros con YuNet, mixup, label smoothing,
AdamW, tasa coseno, borrado/gris aleatorio, TTA, conjunto de redes) pasa al **Anexo A** del
notebook, plegado, con su código y sus números (E2–E13), sin borrar nada.
**Por qué:** pedido del grupo: cada decisión tiene que poder defenderse con el material del curso,
aunque una técnica externa rinda más (E12: 0,360 / 0,418 en val con mixup).
**Alternativas descartadas:** usar mixup y justificarlo en el informe (descartado por el grupo).
**Reemplaza:** D-013 y D-014 en lo que respecta a la CNN final (quedan como historia del anexo).
**Quién:** Francisco Miles (con Claude).

## D-016 · Mejoras dentro del programa: aumentación en KerasTuner, ResNet50 conv4+conv5 y exclusión de actores — 2026-10-28
**Qué:**
1. KerasTuner (NB09) busca también la **intensidad de la aumentación** (rotación ±7/14/22°, zoom
   5/10/15 %, brillo/contraste 10/20/30 %), además de lr, dropout, L2 y ancho; 12 intentos.
2. ResNet50, fase 2 (NB15): se prueban **dos variantes** desde los mismos pesos de la fase 1: descongelar
   solo conv5, y conv4 + conv5. Queda la de mejor F1 macro en validación.
3. Opción `ACTORES_EXCLUIDOS` en 1.1 (habilitada por el profesor solo para nuestro grupo) y nueva
   sección 8.3 con el **recall por actor en validación** de la CNN v3 y ResNet50. Criterio: un actor
   es candidato si ninguno de los dos modelos lo reconoce (recall < 0,05) y tiene pocos videos. Se
   decide con validación, nunca con test, y se informa con y sin exclusión. Se busca conservar la
   mayor cantidad de actores posible.
**Por qué:** pedido del grupo de mejorar sin salir del programa y vigilando el sobreajuste. El techo
de emoción (42,6 %) se tomó de nuestro control cruzado del TP1, sobre las imágenes de un solo grupo:
con el dataset unificado es una **referencia aproximada**, no un techo medido.
**Quién:** Francisco Miles (con Claude).

## D-017 · Elegir por accuracy de validación, no por val_loss — 2026-10-28
**Qué:** KerasTuner usa como objetivo `val_acc_total` = accuracy de validación de actor + de emoción
(maximizar), y el early stopping de todos los modelos vigila el accuracy de validación en lugar de
`val_loss`. Rango de lr de la búsqueda: 3e-4 a 3e-3. Se corrige además la sección 10 (la oclusión de
la v3 recibía imágenes de 224 en vez de 112).
**Por qué:** en la primera corrida completa de la v3 (E14) KerasTuner eligió el intento de menor
`val_loss` (4,98), que tenía 6,5 % de accuracy en actor, y descartó uno con `val_loss` 5,04 y 35,3 % /
35,3 %. Con pocos videos por actor la red se vuelve sobreconfiada y la pérdida deja de reflejar los
aciertos. NB09 de la cátedra usa `objective="val_accuracy"`.
**Alternativas descartadas:** F1 macro como objetivo (no es una métrica de Keras en la versión vista;
se sigue reportando aparte).
**Pendiente de decidir:** excluir a julio_chavez (D-016): un solo video para val/test (6 + 6 imágenes).
**Quién:** Francisco Miles (con Claude).

## D-018 · Búsqueda afinada y una red por tarea como variante de la v3 — 2026-10-29
**Qué:** (1) KerasTuner busca en un rango afinado alrededor del mejor intento de E15: base 48–64,
dropout 0,3–0,5, L2 0 o 1e-4, lr 8e-4–2,5e-3, aumentación suave (rotación 0,02–0,04, zoom 0,05–0,10,
color 0,1–0,2); 10 intentos. (2) Con los hiperparámetros elegidos se entrena además la v3 con **una red
por tarea** (misma arquitectura, una sola salida, pesos por clase solo de esa tarea) y, por tarea, se
usa la variante de mejor F1 macro en validación. (3) No se excluye ningún actor (supera a D-016/D-017
como pendiente).
**Por qué:** en E15 los intentos con base 16 y la aumentación fuerte quedaron entre los peores: la
búsqueda gastaba la mitad de los intentos en una zona mala. La v3 tiene recall 0 en 7 actores; con dos
salidas la emoción puede competir con la identidad por el mismo tronco, y la v1 (una red por tarea) es
el antecedente. Julio Chávez tuvo recall 0,50 con la v3 en E15: la exclusión ya no se justifica con
datos; el único actor que fallan los dos modelos (Eva De Dominici) tiene 8 videos, así que no es falta
de datos y excluirlo solo inflaría la métrica.
**Alternativas descartadas:** pesos distintos por salida en la pérdida (no lo vimos en la cátedra);
más intentos con el rango amplio (más de 1 h y no cabe antes de la entrega).
**Quién:** Francisco Miles (con Claude).

## D-019 · Modelo final y notebook de entrega simplificado — 2026-10-30
**Qué:** la CNN que compite es la v3 de E16 (residual, una red por tarea, hiperparámetros de
KerasTuner fijos en `HP_CNN`). La entrega se presenta con un notebook corto,
`notebooks/TP_Grupo6_Final.ipynb` (~570 líneas de código contra ~1.100): entorno, registro,
partición por video, baselines (trivial y RF), CNN final, ResNet50 conv4+conv5, comparación en
validación con recall por actor, test una vez, oclusión, predicción sobre fotos nuevas y conclusiones.
El notebook completo `TP2_Grupo6.ipynb` queda en el repo sin cambios, con todo el camino (v1, KerasTuner,
anexo con versiones anteriores).
**Por qué:** tres corridas más no subieron sustancialmente la CNN y la entrega es hoy; el grupo pidió un
Colab más legible. En el corto se fijan los hiperparámetros (sin repetir 1,5 h de búsqueda), se usa
`class_weight` con etiquetas enteras (equivalente a los pesos por muestra normalizados de la versión de
dos salidas) y ResNet descongela directamente conv4+conv5 (la variante que ganó en validación en E15 y E16).
**Alternativas descartadas:** seguir buscando hiperparámetros (sin tiempo); entregar el notebook
completo (correcto pero difícil de seguir en la presentación).
**Quién:** Francisco Miles (con Claude).

## Pendientes
- Avisar al responsable del Drive sobre Pachu Peña (ñ) y Rodrigo de la Serna.
- (Opcional) Repetir con 3 semillas para reportar media ± desvío.
- Elegir la configuración final de la CNN v2 (búsqueda 2) y correr el notebook de punta a punta.
- Al final: código para subir fotos y predecir (pedido del grupo).

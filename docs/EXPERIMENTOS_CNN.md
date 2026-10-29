# Experimentos con la CNN propia (insumo del informe)

> **Aviso (2026-10-27, decisión del grupo, D-015):** E2–E13 usan técnicas que **no se vieron en la
> cátedra** (detector y alineación de rostros, mixup, label smoothing, AdamW con tasa coseno, borrado
> y gris aleatorio, TTA, conjunto de redes). Quedan como registro en el **Anexo A** del notebook; la
> CNN que se entrega y compite es la **v3**, solo con técnicas del programa (E14 en adelante).

Registro de **todas** las configuraciones de la CNN desde cero que se probaron, incluidas las que
empeoraron. Todas se miden en **validación** (428 imágenes, videos que no están en train); el test
no se usa para elegir. Métricas: accuracy y F1 macro, con TTA (imagen + espejo) desde E2.

Contexto fijo salvo que se diga lo contrario: dataset huella `512b8d7c66ec` (2.723 imágenes,
20 actores, 154 videos), partición por video (train 1.869 / val 428 / test 426), semilla 42,
Colab GPU T4. Azar: 5 % identidad, 20 % emoción. Mejor baseline clásico: RF 33,4 % / 0,277
(identidad), MLP 27,6 % / 0,252 (emoción).

**Objetivo (pedido del curso):** la CNN propia compite contra las de los otros grupos prediciendo
actor y emoción de fotos nuevas. Sin pesos preentrenados.

## Tabla resumen

| ID | Configuración | Identidad acc / F1 | Emoción acc / F1 | Train acc (actor) | Conclusión |
|---|---|---|---|---|---|
| E1 | v1: 4 bloques Conv-Conv-BN-Pool, 224×224, una red por tarea | 0,192 / 0,175 | 0,255 / 0,247 | ~0,89 | Memoriza el video; peor que el RF |
| E2 | v2: residual multitarea, rostro alineado (escala 0,85), 112×112, 120 ép. | 0,292 / 0,257 | **0,334 / 0,305** | 0,986 | Mejora mucho a la v1; sigue sobreajustando |
| E3 | E2 pero sin alinear (recorte del curso reducido a 112×112) | **0,357 / 0,311** | 0,304 / 0,282 | 0,988 | Alinear ajustado ayuda emoción y perjudica identidad |
| E4 | Búsqueda (80 ép.): alineado 0,85 | 0,243 / 0,223 | 0,299 / 0,261 | 0,964 | Referencia de la búsqueda: con 80 épocas rinde menos que con 120 |
| E5 | Búsqueda (80 ép.): alineado 0,65 (más contexto) | 0,250 / 0,223 | 0,266 / 0,226 | 0,963 | Más margen no ayuda |
| E6 | Búsqueda (80 ép.): alineado 0,65 + mixup 0,2 | 0,238 / 0,221 | 0,255 / 0,210 | 0,915 | Mixup baja el sobreajuste pero no mejora validación |
| E7 | Búsqueda (80 ép.): alineado 0,55 + mixup 0,2 | 0,215 / 0,185 | 0,222 / 0,202 | 0,916 | Peor: el margen extra sale de bordes replicados |
| E8 | Búsqueda (80 ép.): sin alinear + mixup 0,2 | **0,311 / 0,286** | 0,271 / 0,231 | 0,927 | Mejor identidad de la búsqueda: confirma el recorte original |
| E9 | Búsqueda 2 (120 ép.): recorte original + mixup 0,2 | 0,334 / 0,309 | **0,343 / 0,324** | 0,957 | Identidad igual a E3 (F1), emoción +4 puntos: mejor equilibrio |
| E10 | Búsqueda 2 (120 ép.): recorte original + zoom 0,2 / traslación 0,1 | 0,334 / 0,291 | 0,320 / 0,297 | 0,984 | No mejora: más aumentación geométrica no frena la memorización |
| E11 | Búsqueda 2 (120 ép.): recorte original a 64×64 | 0,343 / 0,319 | 0,299 / 0,288 | 0,986 | Identidad igual, emoción peor (se pierde detalle de boca y ojos); 40 % más rápida |
| E12 | Búsqueda 3: recorte original + mixup 0,2, **200 épocas** | **0,360 / 0,344** | **0,418 / 0,400** | 0,969 | Mejor de todas: emoción casi en el techo humano (42,6 %) |
| E13 | Búsqueda 3: conjunto de 3 redes (mixup, 120 ép. c/u) | 0,350 / 0,311 | 0,388 / 0,368 | ~0,96 | No mejora a una sola red (0,348 / 0,407): no vale el costo ×3 |
| **E14** | **v3 solo cátedra**, KerasTuner por `val_loss` (10 intentos) → base 48, dropout 0,1, L2 1e-4, lr 1,26e-4 | 0,252 / 0,224 | 0,290 / 0,270 | — | Debajo del RF: KerasTuner eligió por pérdida un intento que casi no aprendía |
| **E15** | **v3 solo cátedra**, KerasTuner por accuracy de validación (12 intentos, aumentación en la búsqueda) → base 48, dropout 0,4, L2 0, lr 1,5e-3, rot 0,04 / zoom 0,05 / color 0,1 | 0,355 / 0,281 | 0,350 / 0,305 | — | Mejora a E14 en las dos tareas y supera al RF; 7 actores con recall 0 |

## Detalle

### E1 · CNN v1 (experimento de control)
- 4 bloques `[Conv3×3 → Conv3×3 → BN(momentum 0,9) → MaxPool]`, filtros 32→256, GAP, Dense 128,
  Dropout 0,5, L2 1e-4, Adam 1e-3, early stopping desde la época 10 (D-011). 1,21 M parámetros.
- Entrada: recorte del curso 224×224 (la cara ocupa ~53 % del ancho; el resto es fondo).
- Resultado: train ~89 %, val 19 %. Permutación de píxeles: val 15 %, pero 77 % en train
  → la red memoriza. Los mapas de oclusión (sobre ResNet) muestran evidencia en micrófono y fondo.

### E2 · CNN v2 (D-013, D-014)
Cambios respecto de E1 y por qué:
- **Detección + alineación de rostro** (YuNet + 5 puntos → plantilla 112×112, escala 0,85): sacar
  el fondo para que no se pueda memorizar, y preparar igual las fotos nuevas de la competencia.
  Rostro detectado en 2.723/2.723 imágenes.
- **Bloques residuales** (stem 32 + bloques 32, 64, 128, 256): más profundidad con el mismo
  presupuesto (1,24 M parámetros).
- **Multitarea**: una red con dos salidas (actor, emoción); la competencia pide las dos.
- **Aumentación fuerte**: flip, rotación ±11°, zoom/traslación, brillo/contraste ±20 %, gris
  aleatorio (20 %), borrado de un parche (50 %).
- **Label smoothing 0,1**, **AdamW** (wd 0,05) con **tasa coseno** y 5 épocas de calentamiento
  (pico 2e-3), 120 épocas fijas, batch 64, pesos por muestra (actor × emoción).
- **TTA**: promedio imagen + espejo.

Resultado: identidad 0,292 / 0,257 (mejor época ~70: 0,318 sin TTA); emoción 0,334 / 0,305.
Train actor 98,6 %: el sobreajuste sigue siendo el problema principal. 714 s.

### E3 · Ablación: E2 sin alinear
Misma red y receta con el recorte original del curso reducido a 112×112.
- Identidad **sube** a 0,357 / 0,311; emoción **baja** a 0,304 / 0,282.
- Lectura: la alineación con escala 0,85 recorta pelo y contorno de la cabeza, que son pistas
  de identidad legítimas; para la emoción, en cambio, centrar ojos y boca ayuda. Parte de la
  ventaja del recorte original puede venir de fondos parecidos entre videos del mismo canal,
  que no se repiten en fotos nuevas.
- Próximo paso: alineación con más margen (escala 0,65 / 0,55) y mixup contra el sobreajuste.

### E4–E8 · Búsqueda 1: margen de la alineación y mixup (2026-10-27)
Hipótesis después de E3: la alineación ajustada (0,85) pierde pelo y contorno de la cabeza.
Se probó alinear con más margen (0,65 y 0,55) y **mixup** (mezclar pares de imágenes y de
etiquetas con proporción ~Beta(0,2; 0,2)), un regularizador fuerte contra el sobreajuste.
Todas con 80 épocas (para que cada prueba tarde ~8 min) y la misma semilla.

Resultados y lectura:
- **Más margen empeora** (0,85 → 0,65 → 0,55: identidad 0,243 → 0,250 → 0,215). Causa: los
  recortes del curso **ya son recortes** (la cara ocupa ~53 % del lado). Alinear con más margen
  pide píxeles fuera de la imagen y OpenCV los rellena repitiendo el borde: rayas artificiales.
- **Mixup** baja el accuracy de train (96 % → 92 %) pero no mejora validación con 80 épocas.
- **El recorte original del curso gana en identidad** también con mixup (0,311 vs 0,238).
- 80 épocas rinden menos que 120 (E2: 0,292 vs E4: 0,243): la tasa coseno necesita el recorrido largo.
- Ruido de la medición: con 428 imágenes de validación, el error estándar de un accuracy ~0,3 es
  ±2,2 puntos; diferencias menores a ~4 puntos entre corridas únicas no son concluyentes.

Decisión: la CNN se entrena con el **recorte original del curso** (112×112). Para las fotos nuevas
de la competencia se reproduce ese encuadre: en los recortes del curso la cara (caja del detector)
mide en mediana **52,7 % del lado** (p10 37,9 %, p90 70,0 %) y está centrada (centro x 0,50,
centro y 0,50). Una foto nueva se recorta con un cuadrado centrado en la cara de lado
ancho_cara / 0,527. La alineación ajustada se reserva como opción para emoción.

### E9–E11 · Búsqueda 2: sobre el recorte original, 120 épocas (2026-10-27)
Referencia: E3 (recorte original sin mixup) = identidad 0,357 / 0,311, emoción 0,304 / 0,282.
- **E9 · + mixup 0,2:** identidad 0,334 / 0,309 (igual en F1), emoción **0,343 / 0,324** (+4 puntos
  de accuracy, +0,042 de F1). Train actor 95,7 % (vs 98,8 %): mixup sí frena la memorización
  cuando el entrenamiento es largo. 678 s.
- **E10 · + aumentación de encuadre (zoom ±20 %, traslación ±10 %):** 0,334 / 0,291 y 0,320 / 0,297.
  Train actor 98,4 %: mover el encuadre no frena la memorización; no mejora.
- **E11 · entrada 64×64:** 0,343 / 0,319 y 0,299 / 0,288. Identidad dentro del ruido de E3/E9,
  emoción peor: a 64 px se pierde el detalle de boca y cejas. 416 s (40 % más rápida).

**Decisión de la búsqueda 2:** la receta base pasa a ser **recorte original 112×112 + mixup 0,2**
(E9): es la única que mejora una tarea sin empeorar la otra (suma de F1 macro 0,633 contra
0,593 de E3, 0,588 de E10 y 0,607 de E11).

### E12–E13 · Búsqueda 3: entrenar más largo y conjunto de redes (2026-10-27)
- **E12 · E9 con 200 épocas:** identidad **0,360 / 0,344**, emoción **0,418 / 0,400**. Train actor 96,9 %.
  1.126 s. Mejor resultado de todas las búsquedas: con mixup la red sigue aprendiendo después de
  la época 120 (la mezcla de imágenes hace que memorizar cueste más), y la tasa coseno más larga
  pasa más tiempo con tasas bajas, donde se refina. La emoción queda prácticamente en el acuerdo
  entre etiquetadores medido en el TP1 (42,6 %). Identidad ya supera al Random Forest (0,334 / 0,277).
- **E13 · conjunto de 3 redes** (mixup, 120 épocas cada una, semillas 1–3): una sola red da 0,348 / 0,307
  y 0,407 / 0,384; promediar las tres da 0,350 / 0,311 y 0,388 / 0,368. **No mejora** (emoción baja):
  las tres redes se equivocan en las mismas imágenes (los mismos videos difíciles), así que promediar
  no corrige. Triplica el costo sin beneficio.

### Revisión del programa de la cátedra (2026-10-27)
Se revisó el programa (EVA, AULA.html) y los notebooks de práctica NB03–NB16 + clase 17:
- **Vistos:** capas de aumentación (flip, rotación, zoom; brillo en el TP de aumentación; clase 17:
  cada transformación declara una invarianza), Dropout, L1/L2, EarlyStopping (NB07), `class_weight`
  (NB08), KerasTuner (NB09), conexiones residuales con `layers.Add` (NB13), API Functional con varias
  salidas (clase 10), transferencia y ajuste fino (NB14–NB15), oclusión (NB12), SGD/Adam (NB06).
- **No vistos:** mixup, label smoothing, AdamW, tasas de aprendizaje variables (coseno, warmup,
  ReduceLROnPlateau), borrado/gris aleatorio, conjuntos de redes, TTA, detectores de rostro.

Decisión del grupo: la CNN final usa **solo lo visto** (v3). Ver E14 en adelante.

### E14 · v3 solo cátedra, primera corrida completa (2026-10-28, corrida del grupo)
Notebook `cnn-v2` (985c7e4), mismo dataset y partición (huella `512b8d7c66ec`, 154 videos).
- KerasTuner (`RandomSearch`, 10 intentos, 2.833 s) con objetivo **`val_loss`** eligió base 48,
  dropout 0,1, L2 1e-4, **lr 1,26e-4**. Modelo final (42 épocas, early stopping por `val_loss`):
  validación identidad **0,252 / 0,224**, emoción **0,290 / 0,270**; test 0,237 / 0,200 y 0,270 / 0,261.
- **Error de diseño encontrado:** el intento elegido tenía la menor `val_loss` (4,98) pero **6,5 %** de
  accuracy en actor; el intento 1 (lr 4e-4, dropout 0,2) tenía `val_loss` 5,04 y **35,3 % / 35,3 %**.
  Con pocos videos la red se vuelve sobreconfiada: cada error cuesta mucha pérdida, así que
  `val_loss` y accuracy dejan de ir juntos. Lo mismo afecta al early stopping.
- Corrección (D-017): KerasTuner y el early stopping eligen por **accuracy de validación** (suma de
  actor + emoción), como el `objective="val_accuracy"` de NB09; lr entre 3e-4 y 3e-3.
- Referencias de la misma corrida (val): RF 0,334 / 0,277 · v1 0,227 / 0,170 y 0,271 / 0,271 ·
  ResNet50 fase 2 0,624 / 0,530 y 0,425 / 0,409 (test 0,608 / 0,536 y 0,383 / 0,371).
- ResNet50 en test, recall 0 para julio_chavez (6 imágenes) y ricardo_darin (15, confundido con
  pablo_echarri). Pares más confundidos: tini→lali 19, darin→echarri 15, fonzi→mora_bianchi 14,
  milei→fonzi 12. Francella (293 imágenes de train) F1 0,91.

### E15 · v3 con selección por accuracy (D-017), corrida del 29/10
Notebook `cnn-v2` (9bb33da), huella `512b8d7c66ec`, 154 videos, misma partición que E14.
- KerasTuner (12 intentos, 2.973 s) por `val_acc_total`. Elegido: base 48, dropout 0,4, L2 0,
  lr 1,5e-3, aumentación suave (rotación 0,04, zoom 0,05, color 0,1). Los 3 mejores intentos tienen
  base 32–48; la aumentación más fuerte (rotación 0,06 / zoom 0,15) quedó entre los peores.
- Modelo final (66 épocas): validación identidad **0,355 / 0,281**, emoción **0,350 / 0,305**
  (E14: 0,252 / 0,224 y 0,290 / 0,270). Test: 0,326 / 0,250 y 0,286 / 0,264.
- Referencias (val): RF 0,334 / 0,277 y 0,255 / 0,167 · v1 0,287 / 0,249 y 0,264 / 0,256 ·
  ResNet50 conv4+conv5 0,654 / 0,550 y 0,477 / 0,455 (test 0,660 / 0,547 y 0,362 / 0,341).
- Recall por actor (val, 8.3): la v3 tiene recall 0 en azzaro, colapinto, eva_de_dominici,
  erica_rivas, mora_bianchi, javier_milei y scaloni; 1,0 en momo y dario_barassi. julio_chavez:
  v3 0,50, ResNet 0,33 (en test ResNet 0). Único candidato con recall < 0,05 en los dos modelos:
  eva_de_dominici (8 videos, no es falta de videos).
- La caída val → test de ResNet en emoción (0,477 → 0,362) muestra el sesgo de elegir entre muchas
  variantes con solo 428 imágenes de validación.

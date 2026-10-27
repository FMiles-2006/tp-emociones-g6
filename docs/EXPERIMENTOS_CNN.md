# Experimentos con la CNN propia (insumo del informe)

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

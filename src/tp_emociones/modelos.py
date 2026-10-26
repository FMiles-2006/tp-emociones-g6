"""Modelos: CNN propia desde cero y ResNet50 con transferencia en dos fases.

Todas las constructoras:
- fijan la semilla antes de crear pesos (reproducibilidad, regla 4 del CLAUDE.md);
- reciben imágenes en [0, 255] y llevan adentro aumentación + preprocesamiento.

Aumentación válida para rostros (BITACORA D-006, clase 17: "cada transformación declara
una invarianza"):
- Flip horizontal: SÍ. Una cara espejada sigue siendo la misma persona con el mismo gesto.
- Rotación leve (±~8°): SÍ. Las cabezas se inclinan en los videos.
- Zoom leve (±10 %) y traslación chica: SÍ. Los recortes no están perfectamente centrados.
- Brillo/contraste moderados: SÍ. Las fuentes tienen iluminación muy distinta.
- Flip vertical, rotaciones grandes, recortes agresivos: NO. Producen caras imposibles o
  tapan cejas/boca, que es justo la señal de la emoción.
"""
from __future__ import annotations

import keras
from keras import layers

from . import config


# ---------------------------------------------------------------------------
# Bloques comunes
# ---------------------------------------------------------------------------
def capas_aumentacion(seed: int = config.SEED) -> keras.Sequential:
    """Aumentación como capas: solo actúa con training=True (en val/test es identidad)."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(0.022, fill_mode="nearest", seed=seed),  # 0.022*360 ≈ 8°
            layers.RandomZoom(0.10, fill_mode="nearest", seed=seed),
            layers.RandomTranslation(0.05, 0.05, fill_mode="nearest", seed=seed),
            layers.RandomBrightness(0.15, value_range=(0, 255), seed=seed),
            layers.RandomContrast(0.15, seed=seed),
        ],
        name="aumentacion",
    )


@keras.saving.register_keras_serializable(package="tp_emociones")
class PreprocesoResNet50(layers.Layer):
    """preprocess_input de ResNet50 como capa (convierte RGB->BGR y resta la media de
    ImageNet). Cada base preentrenada exige SU preprocesamiento (NB13): usar /255 en vez
    de este corre las features y la transferencia rinde menos."""

    def call(self, x):
        return keras.applications.resnet50.preprocess_input(x)


def compilar(modelo: keras.Model, lr: float) -> keras.Model:
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


# ---------------------------------------------------------------------------
# CNN propia desde cero
# ---------------------------------------------------------------------------
def cnn_propia(
    n_clases: int,
    filtros: tuple[int, ...] = (32, 64, 128, 256),
    dropout: float = 0.5,
    densa: int = 128,
    l2: float = 1e-4,
    aumentar: bool = True,
    seed: int = config.SEED,
    nombre: str = "cnn_propia",
) -> keras.Model:
    """Anatomía del NB11 adaptada a 224x224: bloques [Conv-Conv-BN-MaxPool] con filtros que
    se duplican (menos "dónde", más "qué"), cierre con GlobalAveragePooling + Dense + Dropout.

    Por qué GlobalAveragePooling y no Flatten: con 224x224 el Flatten dejaría ~50.000
    entradas a la densa (millones de parámetros para ~1.900 imágenes = memorizar). GAP deja
    una entrada por filtro.
    Por qué BatchNormalization: estabiliza el entrenamiento desde cero con lotes chicos.
    """
    config.fijar_semillas(seed)
    reg = keras.regularizers.l2(l2) if l2 else None
    entrada = keras.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3), name="imagen")
    x = capas_aumentacion(seed)(entrada) if aumentar else entrada
    x = layers.Rescaling(1.0 / 255, name="escalado")(x)
    for i, f in enumerate(filtros, start=1):
        x = layers.Conv2D(f, 3, padding="same", activation="relu", kernel_regularizer=reg, name=f"b{i}_conv1")(x)
        x = layers.Conv2D(f, 3, padding="same", activation="relu", kernel_regularizer=reg, name=f"b{i}_conv2")(x)
        x = layers.BatchNormalization(name=f"b{i}_bn")(x)
        x = layers.MaxPooling2D(name=f"b{i}_pool")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(densa, activation="relu", kernel_regularizer=reg, name="densa")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    salida = layers.Dense(n_clases, activation="softmax", name="salida")(x)
    return keras.Model(entrada, salida, name=nombre)


# ---------------------------------------------------------------------------
# ResNet50 con transferencia
# ---------------------------------------------------------------------------
def resnet50_transferencia(
    n_clases: int,
    dropout: float = 0.3,
    aumentar: bool = True,
    seed: int = config.SEED,
    nombre: str = "resnet50_transfer",
) -> tuple[keras.Model, keras.Model]:
    """Fase 1 (NB14): base ResNet50 de ImageNet **congelada** + cabeza nueva.

    `base(x, training=False)` deja las BatchNormalization en modo inferencia aunque más
    adelante se descongele la base: con lotes chicos sus estadísticas se distorsionarían
    (bug clásico, NB14). Devuelve (modelo, base) para poder descongelar en la fase 2.
    """
    config.fijar_semillas(seed)
    base = keras.applications.ResNet50(include_top=False, weights="imagenet",
                                       input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    base.trainable = False
    entrada = keras.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3), name="imagen")
    x = capas_aumentacion(seed)(entrada) if aumentar else entrada
    x = PreprocesoResNet50(name="preproceso_resnet50")(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    salida = layers.Dense(n_clases, activation="softmax", name="salida")(x)
    return keras.Model(entrada, salida, name=nombre), base


def descongelar_desde(base: keras.Model, primera_capa: str = "conv5_block1_1_conv") -> int:
    """Fase 2 (NB15): descongela la base desde `primera_capa` en adelante (por defecto el
    bloque conv5 completo, el más específico de ImageNet). Las BatchNormalization quedan
    congeladas. Devuelve cuántas capas quedaron entrenables.

    IMPORTANTE: después de llamar a esto hay que **recompilar** (con lr ~1e-5): sin
    recompilar, el cambio de `trainable` no tiene efecto.
    """
    base.trainable = True
    nombres = [l.name for l in base.layers]
    if primera_capa not in nombres:
        raise ValueError(f"No existe la capa {primera_capa!r} en la base")
    corte = nombres.index(primera_capa)
    for i, capa in enumerate(base.layers):
        capa.trainable = i >= corte and not isinstance(capa, layers.BatchNormalization)
    return sum(c.trainable for c in base.layers)


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def entrenar(
    modelo: keras.Model,
    ds_train,
    ds_val,
    lr: float,
    epocas: int,
    paciencia: int = 5,
    pesos_clase: dict[int, float] | None = None,
    verbose: int = 2,
) -> keras.callbacks.History:
    """Compila y entrena con EarlyStopping sobre val_loss restaurando los mejores pesos.

    La decisión de cuándo parar se toma SIEMPRE contra validación (regla 3).
    """
    compilar(modelo, lr)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=paciencia, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, paciencia // 2), min_lr=1e-7),
    ]
    return modelo.fit(ds_train, validation_data=ds_val, epochs=epocas, class_weight=pesos_clase,
                      callbacks=callbacks, verbose=verbose)


def contar_parametros(modelo: keras.Model) -> dict[str, int]:
    entrenables = sum(int(w.numpy().size) for w in modelo.trainable_weights)
    total = modelo.count_params()
    return {"total": total, "entrenables": entrenables, "congelados": total - entrenables}

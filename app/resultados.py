"""Resultados del proyecto que muestra la app del equipo (sin imágenes del dataset).

Copiados de las salidas del notebook final (corrida E19, partición sin fugas, D-022) y de
docs/EXPERIMENTOS_CNN.md. Formato: (accuracy, F1 macro).
"""

VALIDACION = [
    # modelo, identidad, emoción
    ("Trivial (clase más frecuente)", (0.145, 0.013), (0.241, 0.078)),
    ("Random Forest (píxeles 64×64)", (0.260, 0.222), (0.322, 0.225)),
    ("MLP scikit-learn (píxeles 64×64)", (0.218, 0.187), (0.274, 0.171)),
    ("CNN propia · E19 (compite)", (0.391, 0.368), (0.391, 0.351)),
    ("ResNet50 fase 1 (base congelada)", (0.522, 0.452), (0.441, 0.407)),
    ("ResNet50 fase 2 (conv4 + conv5)", (0.664, 0.543), (0.524, 0.496)),
]

TEST = [
    ("CNN propia · E19 (compite)", (0.311, 0.269), (0.333, 0.303)),
    ("ResNet50 fase 2 (conv4 + conv5)", (0.658, 0.528), (0.399, 0.382)),
]

# Evolución de la CNN en validación. Hasta E17 con la partición anterior; E19 sin fugas.
EVOLUCION = [
    ("v1", "Control: 4 bloques planos, 224×224", (0.287, 0.249), (0.264, 0.256)),
    ("v2", "Residual con técnicas fuera de la cátedra (descartada, Anexo A)", (0.360, 0.344), (0.418, 0.400)),
    ("E14", "Residual, una red con dos salidas, elegida por val_loss", (0.252, 0.224), (0.290, 0.270)),
    ("E15", "Se elige por accuracy de validación (D-017)", (0.355, 0.281), (0.350, 0.305)),
    ("E16a", "Base 48, dropout 0,4, dos salidas", (0.355, 0.284), (0.362, 0.302)),
    ("E16b", "Una red por tarea (D-018)", (0.414, 0.361), (0.393, 0.349)),
    ("E17", "Misma receta, otra corrida", (0.451, 0.406), (0.339, 0.341)),
    ("E19", "Misma receta, partición sin fugas (D-022) · final", (0.391, 0.368), (0.391, 0.351)),
]

DATOS = {
    "recortes": 2723, "actores": 20, "emociones": 5, "videos": "154 → 137 al unificar",
    "train": "1.847 imágenes · 59 videos", "validación": "435 imágenes · 40 videos",
    "test": "441 imágenes · 47 videos (9 compartidos con validación, R4)",
}

PERMUTACION = {"MLP": (0.218, 0.221), "CNN propia": (0.391, 0.241)}

LIMITACIONES = [
    "Entre 2 y 20 videos por actor: las redes aprenden cómo se ve cada actor en esos videos.",
    "Con 4 fotos de la web la CNN acertó 1 y ResNet50 ninguna como primera opción.",
    "Validación de solo 435 imágenes: es un número ruidoso.",
    "En emoción ni las personas coinciden: el acuerdo entre etiquetadores fue de 42,6 %.",
    "El porcentaje es lo que la red cree, no una garantía: hay errores con 96–100 % de confianza.",
]

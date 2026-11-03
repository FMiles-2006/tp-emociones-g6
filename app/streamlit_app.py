"""App del equipo (Streamlit): predicción con las redes finales + resultados del proyecto.

Correr en la compu:   pip install -r app/requirements.txt  y  streamlit run app/streamlit_app.py
En la nube: Streamlit Community Cloud, archivo principal app/streamlit_app.py (D-024).

No usa el dataset ni muestra imágenes del curso: solo la foto que sube quien la usa y los números
del proyecto. Las redes se bajan de Hugging Face (ver inferencia.py).
"""
import pandas as pd
import streamlit as st

import resultados as R
from inferencia import PANELES, abrir_imagen, cargar_redes, frase, predecir, recortar

st.set_page_config(page_title="Quién es y qué siente · Grupo 6", page_icon="🎭", layout="wide")


@st.cache_resource(show_spinner="Descargando las redes (solo la primera vez, unos 210 MB)…")
def redes():
    return cargar_redes()


st.title("Quién es y qué siente")
st.caption("TP IAyAA II · UCA Rosario 2026 · Grupo 6 — CNN propia entrenada desde cero (compite) "
           "y ResNet50 preentrenada (referencia), sobre 20 actores argentinos y 5 emociones.")

tab_pred, tab_res, tab_como = st.tabs(["📷 Predecir", "📊 Resultados del proyecto", "⚙️ Cómo funciona"])

# ----------------------------------------------------------------------------- Predecir
with tab_pred:
    col_foto, col_resp = st.columns([1, 1.4], gap="large")
    with col_foto:
        origen = st.radio("Foto", ["Subir archivo", "Sacar con la cámara"], horizontal=True,
                          label_visibility="collapsed")
        archivo = (st.file_uploader("Subí una foto de uno de los 20 actores", type=["jpg", "jpeg", "png", "webp"])
                   if origen == "Subir archivo" else st.camera_input("Sacá una foto"))
        if archivo:
            img = abrir_imagen(archivo)
            st.markdown("**Recortá la cara.** Las redes se entrenaron con recortes de rostro: "
                        "cuanto más ocupe la cara, mejor.")
            tamano = st.slider("Tamaño del recorte", 0.2, 1.0, 1.0, 0.05)
            cx = st.slider("Horizontal", 0.0, 1.0, 0.5, 0.01)
            cy = st.slider("Vertical", 0.0, 1.0, 0.5, 0.01)
            recorte = recortar(img, tamano, cx, cy)
            c1, c2 = st.columns(2)
            c1.image(img, caption="Foto original", use_container_width=True)
            c2.image(recorte, caption="Lo que ven las redes", use_container_width=True)

    with col_resp:
        if not archivo:
            st.info("Subí una foto (o sacala con la cámara) para ver qué responden las redes.")
            st.markdown("Las redes solo conocen a **20 actores** y **5 emociones** (alegría, enojo, neutral, "
                        "sorpresa, tristeza): con una persona que no está, igual eligen a alguien de la lista.")
        else:
            todas = redes()
            preds = {(f, t): predecir(recorte, todas[f"{f}_{t}"]) for f, t in PANELES}
            for familia, titulo in (("cnn", "CNN propia · compite"), ("resnet", "ResNet50 · referencia")):
                st.subheader(titulo)
                ca, ce = st.columns(2)
                for col, tarea, rotulo in ((ca, "actor", "Actor"), (ce, "emocion", "Emoción")):
                    p = preds[(familia, tarea)]
                    col.metric(rotulo, p[0][0], f"{100 * p[0][1]:.0f} % de confianza", delta_color="off")
                    col.dataframe(
                        pd.DataFrame({"Opción": [n for n, _ in p], "Confianza": [100 * v for _, v in p]}),
                        hide_index=True, use_container_width=True,
                        column_config={"Confianza": st.column_config.ProgressColumn(
                            "Confianza", format="%.0f %%", min_value=0, max_value=100)})
                    texto = frase(p, tarea)
                    col.caption(texto[0].upper() + texto[1:])
            st.caption("El porcentaje es lo que la red cree (salida softmax), no una garantía: "
                       "con fotos de otras fuentes las redes se equivocan mucho, a veces con mucha confianza.")

# ----------------------------------------------------------------------------- Resultados
def tabla(filas):
    return pd.DataFrame([{"Modelo": m, "Identidad · acc": i[0], "Identidad · F1": i[1],
                          "Emoción · acc": e[0], "Emoción · F1": e[1]} for m, i, e in filas])


with tab_res:
    k = st.columns(4)
    k[0].metric("Recortes de rostro", f"{R.DATOS['recortes']:,}".replace(",", "."))
    k[1].metric("Actores · emociones", f"{R.DATOS['actores']} · {R.DATOS['emociones']}")
    k[2].metric("CNN propia · identidad (test)", "0,311", "6 veces el azar (5 %)", delta_color="off")
    k[3].metric("ResNet50 · identidad (test)", "0,658", "referencia, no compite", delta_color="off")

    st.subheader("Validación: baselines, CNN propia y ResNet50")
    st.dataframe(tabla(R.VALIDACION), hide_index=True, use_container_width=True)
    st.subheader("Test (se mira una sola vez, al final)")
    st.dataframe(tabla(R.TEST), hide_index=True, use_container_width=True)
    st.caption("Accuracy y F1 macro (las clases están desbalanceadas). Partición por video, sin fugas (D-022).")

    st.subheader("Evolución de la CNN (validación)")
    evo = pd.DataFrame([{"Paso": p, "Qué cambió": d, "Identidad": i[0], "Emoción": e[0]}
                        for p, d, i, e in R.EVOLUCION])
    st.bar_chart(evo.set_index("Paso")[["Identidad", "Emoción"]], stack=False, height=280)
    st.dataframe(evo, hide_index=True, use_container_width=True)
    st.caption("Hasta E17, partición anterior; E19 con la partición corregida. "
               "Los dos saltos más grandes: elegir por accuracy (E15) y una red por tarea (E16b).")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Partición por video")
        for nombre in ("videos", "train", "validación", "test"):
            st.markdown(f"- **{nombre.capitalize()}:** {R.DATOS[nombre]}")
        st.markdown("**Píxeles permutados:** el MLP rinde igual (0,218 → 0,221) y la CNN cae "
                    "(0,391 → 0,241): la CNN sí usa la estructura de la imagen.")
    with c2:
        st.subheader("Limitaciones")
        for l in R.LIMITACIONES:
            st.markdown(f"- {l}")

# ----------------------------------------------------------------------------- Cómo funciona
with tab_como:
    st.markdown("""
**Separar el modelo de los datos.** Esta app no tiene el dataset: solo las cuatro redes finales (E19),
exportadas a **ONNX** desde el notebook (celda 12.5) y publicadas en Hugging Face
(`Franmiles/tp-g6-modelos`). Antes de publicarlas se verificó que den las mismas probabilidades que en
Keras (diferencia máxima 0,000006).

**Qué le pasa a una foto:**
1. Se corrige la orientación y se recorta un cuadrado (la cara).
2. Se redimensiona a 112×112 para la CNN propia y a 224×224 para ResNet50, en RGB de 0 a 255.
3. Cada red aplica su propia escala por dentro y devuelve una probabilidad por clase (softmax).
4. Se muestran las 5 opciones más probables y la frase con las 3 primeras.

**Dos apps, el mismo modelo:** esta (Streamlit, para el equipo: predicción + resultados) y el frontend
para el cliente (Lovable), donde la red corre en el navegador con `onnxruntime-web`. Código y decisiones:
[github.com/FMiles-2006/tp-emociones-g6](https://github.com/FMiles-2006/tp-emociones-g6) (D-024).
""")

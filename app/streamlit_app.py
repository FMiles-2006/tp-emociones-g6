"""App del equipo (Streamlit): predicción con las redes finales + resultados del proyecto.

Correr en la compu:   pip install -r app/requirements.txt  y  streamlit run app/streamlit_app.py
En la nube: Streamlit Community Cloud, archivo principal app/streamlit_app.py (D-024).

No usa el dataset ni muestra imágenes del curso: solo la foto que sube quien la usa y los números
del proyecto. Las redes se bajan de Hugging Face (ver inferencia.py).

Todo se dibuja con st.markdown (HTML y tablas markdown) en lugar de st.metric, st.dataframe o
st.bar_chart: esos componentes se cargan aparte en el navegador y en Community Cloud fallaban con
"Failed to fetch dynamically imported module". Así la página depende de una sola pieza y carga más rápido.
"""
import io
import threading

import streamlit as st

import resultados as R
from PIL import ImageDraw

from inferencia import PANELES, abrir_imagen, caja_recorte, cargar_redes, frase, predecir, recortar

st.set_page_config(page_title="Quién es y qué siente · Grupo 6", page_icon="🎭", layout="wide")
if "id_foto" not in st.session_state:
    st.session_state.id_foto = 0

COLOR = {"cnn": "#2E86AB", "resnet": "#E07A2E"}
st.markdown("""
<style>
.barra {display:flex; align-items:center; gap:.5rem; margin:.15rem 0; font-size:.9rem}
.barra .nom {flex:0 0 9.5rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
.barra .pista {flex:1; height:.8rem; border-radius:.4rem; background:rgba(128,128,128,.18); overflow:hidden}
.barra .relleno {height:100%; border-radius:.4rem}
.barra .val {flex:0 0 3rem; text-align:right; font-variant-numeric:tabular-nums}
.top {font-size:1.35rem; font-weight:700; margin:.1rem 0 .4rem}
.top small {font-size:.85rem; font-weight:400; opacity:.75}
.kpi {border:1px solid rgba(128,128,128,.3); border-radius:.6rem; padding:.7rem .9rem}
.kpi b {display:block; font-size:1.5rem}
.kpi span {font-size:.85rem; opacity:.75}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def _precarga():
    """Empieza a bajar y abrir las redes apenas se abre la página, en segundo plano.

    Así, mientras la persona elige la foto, las redes ya se están descargando (la primera vez, ~210 MB)
    y la primera predicción no espera desde cero.
    """
    hilo = threading.Thread(target=cargar_redes, daemon=True)
    hilo.start()
    return hilo


def redes():
    hilo = _precarga()
    if hilo.is_alive():
        with st.spinner("Terminando de descargar las redes (solo la primera vez, unos 210 MB)…"):
            hilo.join()
    return cargar_redes()


@st.cache_data(max_entries=8, show_spinner=False)
def foto(datos: bytes):
    """Abre la foto una sola vez por archivo (orientación corregida) y arma una miniatura para mostrar."""
    img = abrir_imagen(io.BytesIO(datos))
    vista = img.copy()
    vista.thumbnail((480, 480))
    return img, vista


@st.cache_data(max_entries=64, show_spinner=False)
def predecir_todo(datos: bytes, zoom: int, mover_x: int, mover_y: int):
    """Predicciones de las 4 redes para una foto y un recorte. Queda en caché: volver a un recorte ya
    probado (o mover otro control y regresar) responde al instante, sin volver a correr las redes."""
    img, _ = foto(datos)
    recorte = recortar(img, 1 - 0.8 * zoom / 100, 0.5 + mover_x / 100, 0.5 + mover_y / 100)
    todas = redes()
    return {(f, t): predecir(recorte, todas[f"{f}_{t}"]) for f, t in PANELES}


def barras(pred, color):
    """Barras de confianza en HTML puro (no dependen de componentes que el navegador baja aparte)."""
    filas = "".join(
        f'<div class="barra"><span class="nom">{n}</span><span class="pista">'
        f'<span class="relleno" style="display:block;width:{100 * p:.1f}%;background:{color}"></span></span>'
        f'<span class="val">{100 * p:.0f} %</span></div>' for n, p in pred)
    return f"<div>{filas}</div>"


def tabla_md(filas):
    """Tabla markdown: modelo, accuracy y F1 macro en identidad y emoción."""
    f = lambda x: f"{x:.3f}".replace(".", ",")
    lineas = ["| Modelo | Identidad · acc | Identidad · F1 | Emoción · acc | Emoción · F1 |", "|---|---:|---:|---:|---:|"]
    lineas += [f"| {m} | {f(i[0])} | {f(i[1])} | {f(e[0])} | {f(e[1])} |" for m, i, e in filas]
    return "\n".join(lineas)


_precarga()   # arranca la descarga de las redes en segundo plano (definida arriba)
st.title("Quién es y qué siente")
st.caption("TP IAyAA II · UCA Rosario 2026 · Grupo 6 — CNN propia entrenada desde cero y ResNet50 con "
           "transfer learning, sobre 20 actores argentinos y 5 emociones.")

tab_pred, tab_res, tab_como = st.tabs(["📷 Predecir", "📊 Resultados del proyecto", "⚙️ Cómo funciona"])

# ----------------------------------------------------------------------------- Predecir
def otra_foto():
    """Cambiar la clave del cargador lo vacía: es la forma confiable de sacar la foto y subir otra."""
    st.session_state.id_foto += 1


def centrar(clave):
    """Vuelve el zoom y el desplazamiento a cero (el recorte central, el mismo que usa el notebook)."""
    for k in (f"zoom_{clave}", f"x_{clave}", f"y_{clave}"):
        st.session_state[k] = 0


@st.fragment
def seccion_predecir():
    """Sección aparte (fragment): mover un control vuelve a dibujar solo esta parte, no toda la página."""
    col_foto, col_resp = st.columns([1, 1.4], gap="large")
    with col_foto:
        origen = st.radio("Foto", ["Subir archivo", "Sacar con la cámara"], horizontal=True,
                          label_visibility="collapsed", on_change=otra_foto)
        clave = f"foto_{st.session_state.id_foto}"
        archivo = (st.file_uploader("Subí una foto de uno de los 20 actores", type=["jpg", "jpeg", "png", "webp"], key=clave)
                   if origen == "Subir archivo" else st.camera_input("Sacá una foto", key=clave))
        if archivo:
            datos = archivo.getvalue()
            st.button("🔄 Probar otra foto", on_click=otra_foto, use_container_width=True)
            img, vista = foto(datos)
            st.markdown("**Recortá la cara.** Las redes se entrenaron con recortes de rostro: "
                        "cuanto más ocupe la cara, mejor.")
            # Controles en palabras: zoom de 0 % (foto entera) a 100 % (máximo acercamiento) y desplazamiento
            # del recorte partiendo del centro. En la barra vertical, «más arriba» queda a la derecha y «más
            # abajo» a la izquierda (como subir un volumen): por eso sus opciones van de +50 a −50.
            zoom = st.slider("🔍 Zoom", 0, 100, 0, 5, format="%d %%", key=f"zoom_{clave}",
                             help="0 %: el cuadrado más grande de la foto. 100 %: lo más cerca posible.")
            lado_h = lambda v: "centro" if v == 0 else (f"{abs(v)} % más a la izquierda" if v < 0
                                                         else f"{v} % más a la derecha")
            lado_v = lambda v: "centro" if v == 0 else (f"{abs(v)} % más arriba" if v < 0 else f"{v} % más abajo")
            mover_x = st.select_slider("↔️ Izquierda · derecha", options=list(range(-50, 51, 2)), value=0,
                                       format_func=lado_h, key=f"x_{clave}")
            mover_y = st.select_slider("↕️ Abajo · arriba", options=list(range(50, -51, -2)), value=0,
                                       format_func=lado_v, key=f"y_{clave}")
            st.button("↺ Volver al recorte inicial", on_click=centrar, args=(clave,), use_container_width=True)
            tamano, cx, cy = 1 - 0.8 * zoom / 100, 0.5 + mover_x / 100, 0.5 + mover_y / 100
            recorte = recortar(img, tamano, cx, cy)
            # Recuadro sobre la foto: muestra qué parte ven las redes (mismas cuentas que el recorte).
            con_caja = vista.copy()
            esc = vista.width / img.width
            x0, y0, x1, y1 = (round(v * esc) for v in caja_recorte(*img.size, tamano, cx, cy))
            ImageDraw.Draw(con_caja).rectangle((x0, y0, x1 - 1, y1 - 1), outline="#FF4B4B", width=3)
            c1, c2 = st.columns(2)
            c1.image(con_caja, caption="Foto original (en rojo, el recorte)")
            c2.image(recorte.resize((224, 224)), caption="Lo que ven las redes")

    with col_resp:
        if not archivo:
            st.info("Subí una foto (o sacala con la cámara) para ver qué responden las redes.")
            st.markdown("Las redes solo conocen a **20 actores** y **5 emociones** (alegría, enojo, neutral, "
                        "sorpresa, tristeza): con una persona que no está, igual eligen a alguien de la lista.")
            return
        with st.spinner("Prediciendo…"):
            preds = predecir_todo(datos, zoom, mover_x, mover_y)
        for familia, titulo in (("cnn", "CNN propia · entrenada desde cero"), ("resnet", "ResNet50 · transfer learning")):
            st.subheader(titulo)
            ca, ce = st.columns(2)
            for col, tarea, rotulo in ((ca, "actor", "Actor"), (ce, "emocion", "Emoción")):
                p = preds[(familia, tarea)]
                texto = frase(p, tarea)
                col.markdown(f'<div class="top">{rotulo}: {p[0][0]} <small>{100 * p[0][1]:.0f} %</small></div>'
                             + barras(p, COLOR[familia]), unsafe_allow_html=True)
                col.caption(texto[0].upper() + texto[1:])
        st.caption("El porcentaje es lo que la red cree (salida softmax), no una garantía: "
                   "con fotos de otras fuentes las redes se equivocan mucho, a veces con mucha confianza.")


with tab_pred:
    seccion_predecir()

# ----------------------------------------------------------------------------- Resultados
with tab_res:
    kpis = [("2.723", "recortes de rostro · 20 actores · 5 emociones"),
            ("154 → 137", "videos de origen → después de unificar"),
            ("0,311", "CNN propia · identidad en test (6 veces el azar)"),
            ("0,658", "ResNet50 · identidad en test (transfer learning)")]
    for col, (v, t) in zip(st.columns(4), kpis):
        col.markdown(f'<div class="kpi"><b>{v}</b><span>{t}</span></div>', unsafe_allow_html=True)

    st.subheader("Validación: baselines, CNN propia y ResNet50")
    st.markdown(tabla_md(R.VALIDACION))
    st.subheader("Test (se mira una sola vez, al final)")
    st.markdown(tabla_md(R.TEST))
    st.caption("Accuracy y F1 macro (las clases están desbalanceadas). Partición por video, sin fugas (D-022).")

    st.subheader("Evolución de la CNN (accuracy en validación)")
    ci, ce = st.columns(2)
    for col, idx, titulo, color in ((ci, 2, "Identidad", COLOR["cnn"]), (ce, 3, "Emoción", "#7B6CD9")):
        col.markdown(f"**{titulo}**")
        col.markdown(barras([(e[0], e[idx][0]) for e in R.EVOLUCION], color), unsafe_allow_html=True)
    st.markdown("\n".join(["| Paso | Qué cambió | Identidad (acc / F1) | Emoción (acc / F1) |", "|---|---|---|---|"]
                          + [f"| {p} | {d} | {i[0]:.3f} / {i[1]:.3f} | {e[0]:.3f} / {e[1]:.3f} |".replace(".", ",")
                             for p, d, i, e in R.EVOLUCION]))
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

**Lo que ve el cliente es solo esta app:** sube una foto y recibe actor y emoción; los datos y el
entrenamiento quedan en el notebook. Código y decisiones:
[github.com/FMiles-2006/tp-emociones-g6](https://github.com/FMiles-2006/tp-emociones-g6) (D-024, D-025).
""")

# modules/mod1_senia_a_texto.py
# Version sin OpenCV ni MediaPipe — compatible con Streamlit Cloud
# La camara en la nube no tiene acceso directo al hardware,
# por lo que este modulo ofrece simulacion interactiva

import streamlit as st
from utils.vocabulary import VOCAB


def run():
    st.markdown("### 🤙 Seña → Texto")
    st.caption(
        "**Prototipo experimental** — vocabulario cerrado de 10 señas. "
        "No reemplaza intérpretes humanos."
    )

    st.info(
        "**Nota:** En la versión web (Streamlit Cloud) la cámara no está disponible. "
        "Usá los botones para simular la detección de señas, "
        "o corré el proyecto localmente para usar la cámara real."
    )

    if "detection_history" not in st.session_state:
        st.session_state.detection_history = []

    st.markdown("---")
    st.markdown("**Seleccioná una seña para simular su detección:**")

    cols = st.columns(2)
    vocab_items = list(VOCAB.items())

    for i, (key, info) in enumerate(vocab_items):
        with cols[i % 2]:
            if st.button(
                f"{info['emoji']}  {info['label']}",
                key=f"detect_{key}",
                use_container_width=True
            ):
                st.session_state.detection_history.append(info['label'])
                if len(st.session_state.detection_history) > 8:
                    st.session_state.detection_history.pop(0)
                st.session_state["last_detected"] = key

    st.markdown("---")

    if st.session_state.get("last_detected"):
        key = st.session_state["last_detected"]
        info = VOCAB[key]
        st.success(f"## {info['emoji']} {info['label']}")
        st.caption(info['description'])
    else:
        st.markdown("*Ninguna seña detectada aún*")

    if st.session_state.detection_history:
        st.markdown("**Secuencia detectada:**")
        st.markdown("`" + " → ".join(st.session_state.detection_history) + "`")

    if st.button("Limpiar historial"):
        st.session_state.detection_history = []
        st.session_state["last_detected"] = None
        st.rerun()

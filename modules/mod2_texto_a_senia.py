# modules/mod2_texto_a_senia.py
# Version sin OpenCV — compatible con Streamlit Cloud
 
import streamlit as st
import time
import re
import io
 
from utils.vocabulary import VOCAB, TEXT_ALIASES
from utils.pose_renderer import render_pose, render_gif, POSES
 
 
def _normalizar_texto(texto: str) -> list:
    texto = texto.lower().strip()
    texto = re.sub(r"[^a-z\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1\s]", "", texto)
    return [t for t in texto.split() if t]
 
 
def _resolver_secuencia(tokens: list) -> tuple:
    secuencia = []
    desconocidas = []
    vistos = []
    for token in tokens:
        key = TEXT_ALIASES.get(token)
        if key and key not in vistos:
            secuencia.append({"key": key, **VOCAB[key]})
            vistos.append(key)
        elif not key:
            desconocidas.append(token)
    return secuencia, desconocidas
 
 
def run():
    st.markdown("### \u2328\ufe0f Texto \u2192 Se\u00f1a")
    st.caption(
        "**Prototipo experimental** \u2014 ingres\u00e1 texto en espa\u00f1ol argentino. "
        "El sistema identifica palabras del vocabulario cerrado y las representa visualmente."
    )
 
    with st.sidebar:
        st.markdown("**Configuraci\u00f3n \u2014 M\u00f3dulo 2**")
        frame_delay = st.slider("Duraci\u00f3n por se\u00f1a (seg)", 0.8, 3.0, 1.5, 0.1)
        usar_gif = st.checkbox("Usar animaci\u00f3n GIF", value=True)
 
    # Palabras rapidas
    st.markdown("**Palabras r\u00e1pidas del vocabulario:**")
    chips_cols = st.columns(5)
    for i, (key, info) in enumerate(VOCAB.items()):
        with chips_cols[i % 5]:
            if st.button(f"{info['emoji']} {info['label']}", key=f"chip_{key}",
                         use_container_width=True):
                current = st.session_state.get("texto_input", "")
                nuevo = (current + " " + info['label'].lower()).strip()
                st.session_state["texto_input"] = nuevo
 
    st.markdown("**Texto a traducir:**")
    texto = st.text_area(
        "texto",
        value=st.session_state.get("texto_input", ""),
        height=100,
        placeholder="Ejemplo: hola necesito ayuda",
        key="texto_input",
        label_visibility="collapsed",
    )
 
    col_traducir, col_limpiar = st.columns([3, 1])
    with col_traducir:
        traducir = st.button("\u25b6 Traducir a se\u00f1a", type="primary", use_container_width=True)
    with col_limpiar:
        if st.button("Limpiar", use_container_width=True):
            st.session_state["texto_input"] = ""
            st.rerun()
 
    st.markdown("---")
    st.markdown("**Representaci\u00f3n visual**")
 
    col_avatar, col_info = st.columns([2, 3])
    avatar_placeholder = col_avatar.empty()
    word_placeholder = col_avatar.empty()
 
    with col_info:
        info_box = st.empty()
        progress_box = st.empty()
        sequence_box = st.empty()
 
    _mostrar_avatar_vacio(avatar_placeholder, 300, 380)
 
    if not traducir or not texto.strip():
        if not traducir:
            info_box.info("Ingres\u00e1 texto y presion\u00e1 **Traducir a se\u00f1a**.")
        return
 
    tokens = _normalizar_texto(texto)
    secuencia, desconocidas = _resolver_secuencia(tokens)
 
    if desconocidas:
        st.warning(
            f"\u26a0\ufe0f Sin cobertura en el vocabulario: **{', '.join(desconocidas)}**"
        )
 
    if not secuencia:
        info_box.error("Ninguna palabra pertenece al vocabulario disponible.")
        return
 
    labels = [f"{s['emoji']} {s['label']}" for s in secuencia]
    sequence_box.markdown("**Secuencia identificada:**  \n" + " \u2192 ".join(labels))
 
    total = len(secuencia)
 
    if usar_gif:
        # Mostrar GIF animado de cada seña
        for i, seña in enumerate(secuencia):
            progress_box.progress((i) / total, text=f"Mostrando se\u00f1a {i+1} de {total}")
            info_box.success(f"## {seña['emoji']} {seña['label']}")
            gif_bytes = render_gif(seña["key"], width=300, height=380, fps=3)
            if gif_bytes:
                avatar_placeholder.image(gif_bytes, width=300)
            word_placeholder.caption(f"Se\u00f1a {i+1}/{total}")
            time.sleep(frame_delay)
    else:
        # Mostrar frames estaticos alternados
        for i, seña in enumerate(secuencia):
            progress_box.progress((i) / total, text=f"Mostrando se\u00f1a {i+1} de {total}")
            info_box.success(f"## {seña['emoji']} {seña['label']}")
            for fi in range(2):
                img = render_pose(seña["key"], frame_idx=fi, width=300, height=380,
                                  label=seña["label"])
                avatar_placeholder.image(img, width=300)
                word_placeholder.caption(f"Se\u00f1a {i+1}/{total}")
                time.sleep(frame_delay / 2)
 
    progress_box.progress(1.0, text="\u2713 Secuencia completa")
    info_box.success(f"\u2713 Se representaron {total} se\u00f1a(s).")
 
    with st.expander("Ver todas las poses"):
        ref_cols = st.columns(min(total, 4))
        for i, seña in enumerate(secuencia):
            with ref_cols[i % 4]:
                if usar_gif:
                    gif_bytes = render_gif(seña["key"], width=180, height=230, fps=3)
                    if gif_bytes:
                        st.image(gif_bytes, width=180, caption=seña["label"])
                else:
                    img = render_pose(seña["key"], frame_idx=0, width=180, height=230,
                                      label=seña["label"])
                    st.image(img, width=180, caption=seña["label"])
 
 
def _mostrar_avatar_vacio(placeholder, w, h):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (w, h), (245, 243, 240))
    d = ImageDraw.Draw(img)
    d.text((w//2 - 60, h//2 - 10), "Avatar aparecera", fill=(180, 180, 180))
    d.text((w//2 - 40, h//2 + 14), "al traducir", fill=(180, 180, 180))
    placeholder.image(img, width=w)

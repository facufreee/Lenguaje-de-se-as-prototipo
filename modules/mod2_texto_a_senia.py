# modules/mod2_texto_a_seña.py
# Módulo 2: texto a representación visual de seña.
#
# Flujo:
#   1. Usuario ingresa texto libre
#   2. Sistema tokeniza y normaliza el texto
#   3. Busca equivalentes en vocabulario cerrado via TEXT_ALIASES
#   4. Para cada seña encontrada, anima el esqueleto 2D frame por frame
#
# LIMITES EXPLÍCITOS:
#   - Solo funciona con palabras del vocabulario (10 unidades)
#   - Las señas son representaciones simplificadas, no LSA verificada por intérpretes
#   - El esqueleto 2D es una aproximación visual, no un avatar certificado de LSA
#   - No hay voz ni audio

import streamlit as st
import time
import re

from utils.vocabulary import VOCAB, TEXT_ALIASES
from utils.pose_renderer import render_pose, POSES as POSE_KEYPOINTS


def _normalizar_texto(texto: str) -> list[str]:
    """Limpia y tokeniza el texto de entrada."""
    texto = texto.lower().strip()
    texto = re.sub(r"[^a-záéíóúüñ\s]", "", texto)
    return [t for t in texto.split() if t]


def _resolver_secuencia(tokens: list[str]) -> tuple[list[dict], list[str]]:
    """
    Convierte tokens en secuencia de señas reconocidas + lista de desconocidas.
    Evita repetir la misma seña consecutivamente.
    """
    secuencia = []
    desconocidas = []
    visto_keys = []

    for token in tokens:
        key = TEXT_ALIASES.get(token)
        if key and key not in visto_keys:
            secuencia.append({"key": key, **VOCAB[key]})
            visto_keys.append(key)
        elif not key:
            desconocidas.append(token)

    return secuencia, desconocidas


def run():
    st.markdown("### ⌨️ Texto → Seña")
    st.caption(
        "**Prototipo experimental** — ingresá texto en español argentino. "
        "El sistema identifica palabras del vocabulario cerrado y las representa visualmente como esqueleto 2D animado."
    )

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("**Configuración — Módulo 2**")
        frame_delay = st.slider(
            "Duración por seña (seg)", 0.8, 3.0, 1.5, 0.1,
            help="Tiempo que se muestra cada seña antes de pasar a la siguiente"
        )
        avatar_size = st.select_slider(
            "Tamaño de avatar",
            options=["Pequeño", "Mediano", "Grande"],
            value="Mediano"
        )

    size_map = {"Pequeño": (220, 280), "Mediano": (300, 380), "Grande": (380, 480)}
    av_w, av_h = size_map[avatar_size]

    # --- Palabras rápidas ---
    st.markdown("**Palabras rápidas del vocabulario:**")
    chips_cols = st.columns(5)
    for i, (key, info) in enumerate(VOCAB.items()):
        with chips_cols[i % 5]:
            if st.button(f"{info['emoji']} {info['label']}", key=f"chip_{key}",
                         use_container_width=True):
                current = st.session_state.get("texto_input", "")
                nuevo = (current + " " + info['label'].lower()).strip()
                st.session_state["texto_input"] = nuevo

    # --- Input de texto ---
    st.markdown("**Texto a traducir:**")
    texto = st.text_area(
        "Escribí en español argentino",
        value=st.session_state.get("texto_input", ""),
        height=100,
        placeholder="Ejemplo: hola necesito ayuda médico",
        key="texto_input",
        label_visibility="collapsed",
    )

    col_traducir, col_limpiar = st.columns([3, 1])
    with col_traducir:
        traducir = st.button("▶ Traducir a seña", type="primary", use_container_width=True)
    with col_limpiar:
        if st.button("Limpiar", use_container_width=True):
            st.session_state["texto_input"] = ""
            st.rerun()

    # --- Sección de resultado ---
    st.markdown("---")
    st.markdown("**Representación visual**")

    col_avatar, col_info = st.columns([2, 3])

    avatar_placeholder = col_avatar.empty()
    word_placeholder = col_avatar.empty()

    with col_info:
        info_box = st.empty()
        progress_box = st.empty()
        sequence_box = st.empty()

    # Estado inicial del avatar
    _mostrar_avatar_vacio(avatar_placeholder, av_w, av_h)

    if not traducir or not texto.strip():
        if not traducir:
            info_box.info("Ingresá texto y presioná **Traducir a seña**.")
        return

    # --- Procesamiento ---
    tokens = _normalizar_texto(texto)
    secuencia, desconocidas = _resolver_secuencia(tokens)

    # Feedback inmediato
    if desconocidas:
        st.warning(
            f"⚠️ Palabras sin cobertura en el vocabulario actual: **{', '.join(desconocidas)}**  \n"
            f"El prototipo solo reconoce las 10 unidades del vocabulario cerrado."
        )

    if not secuencia:
        info_box.error(
            "Ninguna palabra del texto pertenece al vocabulario disponible.  \n"
            "Usá las palabras rápidas o revisá la lista de señas."
        )
        return

    # Mostrar secuencia identificada
    labels = [f"{s['emoji']} {s['label']}" for s in secuencia]
    sequence_box.markdown("**Secuencia identificada:**  \n" + " → ".join(labels))

    # --- Animación ---
    total = len(secuencia)
    FRAMES_PER_SEÑA = 2

    for i, seña in enumerate(secuencia):
        progress_box.progress(
            (i) / total,
            text=f"Mostrando seña {i+1} de {total}: {seña['label']}"
        )
        info_box.success(f"## {seña['emoji']} {seña['label']}")

        # Alternar frames de la pose para dar sensación de movimiento
        for fi in range(FRAMES_PER_SEÑA):
            img = render_pose(
                seña["key"],
                frame_idx=fi,
                width=av_w,
                height=av_h,
                label=seña["label"]
            )
            avatar_placeholder.image(img, use_container_width=False, width=av_w)
            word_placeholder.caption(f"Seña {i+1}/{total}")
            time.sleep(frame_delay / FRAMES_PER_SEÑA)

    # Fin de secuencia
    progress_box.progress(1.0, text="✓ Secuencia completa")
    info_box.success(f"✓ Se representaron {total} seña(s) del vocabulario.")

    # Mostrar referencia visual de todas las señas de la secuencia
    with st.expander("Ver todas las poses de la secuencia"):
        ref_cols = st.columns(min(total, 4))
        for i, seña in enumerate(secuencia):
            with ref_cols[i % 4]:
                ref_img = render_pose(seña["key"], frame_idx=0,
                                      width=180, height=230, label=seña["label"])
                st.image(ref_img, width=180, caption=seña["label"])


def _mostrar_avatar_vacio(placeholder, w, h):
    """Muestra el área de avatar vacía con instrucción."""
    import numpy as np
    import cv2
    img = np.ones((h, w, 3), dtype=np.uint8) * 245
    cv2.putText(img, "Avatar aparecera", (w//2 - 80, h//2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.putText(img, "al traducir", (w//2 - 45, h//2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    placeholder.image(img, use_container_width=False, width=w)

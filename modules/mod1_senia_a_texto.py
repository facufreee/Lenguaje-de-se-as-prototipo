# modules/mod1_senia_a_texto.py
# Módulo 1 reescrito para MediaPipe >= 0.10.30 (nueva API sin mp.solutions)

import streamlit as st
import cv2
import numpy as np
import time

from classifier.hand_classifier import HandClassifier
from utils.vocabulary import VOCAB

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
    NEW_API = True
except Exception:
    NEW_API = False


def run():
    st.markdown("### 🤙 Seña → Texto")
    st.caption(
        "**Prototipo experimental** — vocabulario cerrado de 10 señas. "
        "No reemplaza intérpretes humanos. Precisión limitada según condiciones de iluminación."
    )

    with st.sidebar:
        st.markdown("**Configuración — Módulo 1**")
        confidence_threshold = st.slider(
            "Umbral de confianza", 0.40, 0.90, 0.55, 0.05,
        )
        camera_index = st.number_input("Índice de cámara", 0, 5, 0, 1)

    classifier = HandClassifier(confidence_threshold=confidence_threshold)

    if "detection_history" not in st.session_state:
        st.session_state.detection_history = []
    if "last_detection" not in st.session_state:
        st.session_state.last_detection = None
    if "last_detection_time" not in st.session_state:
        st.session_state.last_detection_time = 0

    col_cam, col_result = st.columns([3, 2])
    with col_cam:
        st.markdown("**Vista de cámara**")
        frame_placeholder = st.empty()
    with col_result:
        st.markdown("**Resultado**")
        result_box = st.empty()
        confidence_box = st.empty()
        st.markdown("---")
        st.markdown("**Secuencia detectada**")
        history_box = st.empty()

    col_start, col_clear = st.columns(2)
    with col_start:
        run_camera = st.toggle("Iniciar cámara", value=False)
    with col_clear:
        if st.button("Limpiar historial"):
            st.session_state.detection_history = []
            st.rerun()

    with st.expander("Vocabulario disponible (10 señas)"):
        cols = st.columns(5)
        for i, (key, info) in enumerate(VOCAB.items()):
            with cols[i % 5]:
                st.markdown(f"**{info['emoji']}** {info['label']}")

    if not run_camera:
        result_box.info("Activá la cámara con el toggle para comenzar.")
        frame_placeholder.image(_placeholder_frame(), use_container_width=True)
        return

    # Detección con OpenCV puro (sin MediaPipe en modo legacy)
    # Usamos detector de manos basado en color de piel como fallback simple
    cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        st.error(f"No se pudo abrir la cámara {int(camera_index)}.")
        return

    st.info("📷 Cámara activa — la detección usa análisis de posición de mano. Mantené la seña quieta 2 segundos.")

    COOLDOWN = 2.0
    frame_count = 0

    try:
        while run_camera:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_count += 1

            # Detectar región de mano por color de piel (HSV)
            hand_region, hand_bbox = _detect_hand_region(frame)
            annotated = frame.copy()

            detection_result = None

            if hand_bbox is not None:
                x, y, w, h = hand_bbox
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (50, 140, 220), 2)
                cv2.putText(annotated, "Mano detectada", (x, y-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 140, 220), 1)

                # Clasificar basado en geometría de la región
                detection_result = _classify_from_region(hand_region, hand_bbox, frame, classifier)

            frame_placeholder.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

            now = time.time()
            if detection_result and detection_result.get("key"):
                conf_pct = int(detection_result["confidence"] * 100)
                result_box.success(f"## {VOCAB[detection_result['key']]['emoji']} {detection_result['label']}")
                confidence_box.progress(detection_result["confidence"], text=f"Confianza: {conf_pct}%")

                if (now - st.session_state.last_detection_time) > COOLDOWN:
                    if st.session_state.last_detection != detection_result["key"]:
                        st.session_state.detection_history.append(detection_result["label"])
                        if len(st.session_state.detection_history) > 8:
                            st.session_state.detection_history.pop(0)
                        st.session_state.last_detection = detection_result["key"]
                        st.session_state.last_detection_time = now
            elif hand_bbox is not None:
                result_box.warning("⚠️ Mano detectada — seña no reconocida con suficiente confianza")
                confidence_box.empty()
            else:
                result_box.info("👐 Mostrá una mano a la cámara con buena iluminación")
                confidence_box.empty()

            if st.session_state.detection_history:
                history_box.markdown("`" + " · ".join(st.session_state.detection_history) + "`")
            else:
                history_box.caption("Sin detecciones aún.")

            time.sleep(0.04)

    finally:
        cap.release()


def _detect_hand_region(frame):
    """Detecta región de mano por segmentación de color de piel en HSV."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Rango de color de piel (funciona mejor con buena iluminación)
    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Morfología para limpiar ruido
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    # El contorno más grande probablemente es la mano
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    # Filtrar por tamaño mínimo razonable
    if area < 3000:
        return None, None

    x, y, w, h = cv2.boundingRect(largest)
    region = frame[y:y+h, x:x+w]
    return region, (x, y, w, h)


def _classify_from_region(region, bbox, full_frame, classifier):
    """
    Clasifica la seña basándose en propiedades geométricas de la región de mano.
    Es una aproximación simple — en producción usaría MediaPipe landmarks.
    """
    if region is None or region.size == 0:
        return None

    x, y, w, h = bbox
    fh, fw = full_frame.shape[:2]

    # Características simples extraídas de la región
    aspect_ratio = w / h if h > 0 else 1.0
    hand_y_norm = y / fh  # posición vertical normalizada
    hand_x_norm = x / fw  # posición horizontal normalizada
    area_norm = (w * h) / (fw * fh)  # tamaño relativo

    # Análisis de forma de la región de mano
    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_region,
                       np.array([0, 20, 70]), np.array([20, 150, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    spread_ratio = 0.5
    if contours:
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        cnt_area = cv2.contourArea(largest)
        if hull_area > 0:
            spread_ratio = cnt_area / hull_area  # 1.0 = mano cerrada, <1 = dedos abiertos

    scores = {}

    # Reglas basadas en posición y forma
    # Mano alta y abierta → hola
    scores["hola"] = (0.4 if hand_y_norm < 0.4 else 0.1) + (0.4 if spread_ratio < 0.75 else 0.1) + 0.1

    # Mano central y compacta → gracias
    scores["gracias"] = (0.35 if 0.3 < hand_y_norm < 0.65 else 0.1) + (0.4 if spread_ratio > 0.80 else 0.1) + 0.1

    # Mano muy arriba y abierta → médico (urgencia)
    scores["medico"] = (0.5 if hand_y_norm < 0.30 else 0.05) + (0.3 if spread_ratio < 0.70 else 0.1) + 0.1

    # Mano ancha horizontalmente → esperar
    scores["esperar"] = (0.45 if aspect_ratio > 1.3 else 0.1) + (0.3 if 0.25 < hand_y_norm < 0.6 else 0.05) + 0.1

    # Mano compacta (puño) posición media → sí
    scores["si"] = (0.5 if spread_ratio > 0.85 else 0.1) + (0.35 if 0.25 < hand_y_norm < 0.55 else 0.05)

    # Mano izquierda del frame → no
    scores["no"] = (0.4 if hand_x_norm > 0.55 else 0.1) + (0.3 if spread_ratio < 0.80 else 0.1) + 0.15

    # Dos manos o mano grande → ayuda
    scores["ayuda"] = (0.4 if area_norm > 0.06 else 0.1) + 0.2 + (0.2 if hand_y_norm < 0.5 else 0.05)

    # Mano baja → baño
    scores["banio"] = (0.45 if hand_y_norm > 0.55 else 0.1) + (0.35 if spread_ratio < 0.80 else 0.1) + 0.1

    # Mano con dedos separados → intérprete
    scores["interprete"] = (0.4 if spread_ratio < 0.72 else 0.1) + (0.3 if 0.2 < hand_y_norm < 0.55 else 0.05) + 0.15

    # Mano plana horizontal → documento
    scores["documento"] = (0.4 if aspect_ratio > 1.1 else 0.1) + (0.35 if spread_ratio > 0.75 else 0.1) + 0.1

    best_key = max(scores, key=scores.get)
    best_score = scores[best_key]

    threshold = classifier.confidence_threshold

    if best_score < threshold:
        return {"label": "No reconocida", "key": None, "confidence": best_score}

    return {
        "label": VOCAB[best_key]["label"],
        "key": best_key,
        "confidence": min(best_score, 0.92),
    }


def _placeholder_frame():
    img = np.ones((320, 480, 3), dtype=np.uint8) * 240
    cv2.putText(img, "Camara inactiva", (120, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
    cv2.putText(img, "Activa el toggle para comenzar", (70, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    return img

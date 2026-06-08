# utils/pose_renderer.py
# Renderiza representaciones visuales de señas como esqueleto 2D en imágenes numpy.
# No usa un avatar 3D real — decision de diseño deliberada:
# un avatar completo (p.ej. con Unity o Blender) requeriría activos de animación
# que no existen para LSA en forma abierta. El esqueleto 2D es honesto, explicable
# y suficiente para comunicar el concepto en un prototipo académico.

import numpy as np
import cv2


# Poses: coordenadas normalizadas [0-1] para un canvas 300x400
# Estructura: head_y, neck_y, shoulder_y,
#   left_elbow (x,y), right_elbow (x,y),
#   left_hand (x,y), right_hand (x,y)
# x=0 izquierda, x=1 derecha, y=0 arriba, y=1 abajo

POSE_KEYPOINTS = {
    "hola": [
        # Mano derecha levantada y abierta — saludo
        {"lh": (0.30, 0.45), "rh": (0.78, 0.20), "le": (0.28, 0.62), "re": (0.72, 0.42)},
        {"lh": (0.30, 0.47), "rh": (0.75, 0.18), "le": (0.28, 0.62), "re": (0.70, 0.40)},
    ],
    "gracias": [
        # Mano al pecho, gesto de agradecer
        {"lh": (0.42, 0.58), "rh": (0.58, 0.55), "le": (0.35, 0.65), "re": (0.65, 0.62)},
        {"lh": (0.44, 0.56), "rh": (0.56, 0.53), "le": (0.36, 0.63), "re": (0.64, 0.60)},
    ],
    "ayuda": [
        # Puño sobre palma — señal de ayuda/urgencia
        {"lh": (0.35, 0.40), "rh": (0.65, 0.40), "le": (0.30, 0.60), "re": (0.70, 0.60)},
        {"lh": (0.37, 0.42), "rh": (0.63, 0.42), "le": (0.32, 0.61), "re": (0.68, 0.61)},
    ],
    "si": [
        # Puño que asiente hacia abajo
        {"lh": (0.38, 0.65), "rh": (0.62, 0.38), "le": (0.34, 0.65), "re": (0.66, 0.55)},
        {"lh": (0.38, 0.65), "rh": (0.62, 0.42), "le": (0.34, 0.65), "re": (0.66, 0.58)},
    ],
    "no": [
        # Dedo índice que oscila horizontalmente
        {"lh": (0.38, 0.65), "rh": (0.72, 0.35), "le": (0.34, 0.65), "re": (0.68, 0.50)},
        {"lh": (0.38, 0.65), "rh": (0.60, 0.35), "le": (0.34, 0.65), "re": (0.65, 0.50)},
    ],
    "medico": [
        # Cruz médica indicada / señal de emergencia
        {"lh": (0.28, 0.38), "rh": (0.72, 0.38), "le": (0.25, 0.58), "re": (0.75, 0.58)},
        {"lh": (0.30, 0.36), "rh": (0.70, 0.36), "le": (0.27, 0.56), "re": (0.73, 0.56)},
    ],
    "interprete": [
        # Señal de llamado / mediación
        {"lh": (0.30, 0.42), "rh": (0.70, 0.42), "le": (0.27, 0.60), "re": (0.73, 0.60)},
        {"lh": (0.32, 0.40), "rh": (0.68, 0.40), "le": (0.29, 0.59), "re": (0.71, 0.59)},
    ],
    "banio": [
        # Letra B en movimiento descendente
        {"lh": (0.38, 0.65), "rh": (0.60, 0.32), "le": (0.34, 0.65), "re": (0.65, 0.50)},
        {"lh": (0.38, 0.65), "rh": (0.60, 0.40), "le": (0.34, 0.65), "re": (0.65, 0.55)},
    ],
    "esperar": [
        # Palmas hacia adelante, gesto de pausa
        {"lh": (0.28, 0.48), "rh": (0.72, 0.48), "le": (0.26, 0.65), "re": (0.74, 0.65)},
        {"lh": (0.30, 0.46), "rh": (0.70, 0.46), "le": (0.28, 0.63), "re": (0.72, 0.63)},
    ],
    "documento": [
        # Gesto de mostrar/presentar un papel
        {"lh": (0.28, 0.52), "rh": (0.72, 0.52), "le": (0.26, 0.65), "re": (0.74, 0.65)},
        {"lh": (0.30, 0.50), "rh": (0.70, 0.50), "le": (0.28, 0.63), "re": (0.72, 0.63)},
    ],
}


def render_pose(seña_key: str, frame_idx: int = 0,
                width: int = 300, height: int = 380,
                bg_color=(245, 247, 250), label: str = "") -> np.ndarray:
    """
    Renderiza una seña como esqueleto 2D sobre imagen numpy.

    Args:
        seña_key: clave del vocabulario (ej: "hola")
        frame_idx: índice de frame para animación (alterna entre keypoints)
        width, height: dimensiones del canvas
        bg_color: color de fondo RGB
        label: texto a mostrar debajo del esqueleto

    Returns:
        imagen numpy BGR lista para mostrar con cv2 o st.image
    """
    img = np.ones((height, width, 3), dtype=np.uint8)
    img[:] = bg_color[::-1]  # BGR

    if seña_key not in POSE_KEYPOINTS:
        _draw_text_center(img, "Seña no disponible", width // 2, height // 2)
        return img

    frames = POSE_KEYPOINTS[seña_key]
    kp = frames[frame_idx % len(frames)]

    # Puntos anatómicos fijos del torso
    head_cx = int(width * 0.50)
    head_cy = int(height * 0.14)
    head_r  = int(min(width, height) * 0.08)

    neck_x  = head_cx
    neck_y  = head_cy + head_r + 4

    sh_y    = int(height * 0.38)
    ls_x    = int(width * 0.28)
    rs_x    = int(width * 0.72)

    hip_y   = int(height * 0.62)
    lh_x    = int(width * 0.32)
    rh_x    = int(width * 0.68)

    # Puntos de brazo desde keypoints
    le = (int(kp["le"][0] * width), int(kp["le"][1] * height))
    re = (int(kp["re"][0] * width), int(kp["re"][1] * height))
    lhand = (int(kp["lh"][0] * width), int(kp["lh"][1] * height))
    rhand = (int(kp["rh"][0] * width), int(kp["rh"][1] * height))

    color_skeleton = (100, 110, 130)   # gris azulado
    color_joint    = (60,  70,  90)
    color_hand     = (50, 140, 220)    # azul destacado para manos
    color_hand_fill= (180, 215, 245)   # relleno de mano
    thickness      = 3

    # Torso
    cv2.line(img, (neck_x, neck_y), (neck_x, hip_y), color_skeleton, thickness)
    cv2.line(img, (ls_x, sh_y), (rs_x, sh_y), color_skeleton, thickness)
    cv2.line(img, (lh_x, hip_y), (rh_x, hip_y), color_skeleton, thickness)

    # Brazos: hombro → codo → mano
    cv2.line(img, (ls_x, sh_y), le, color_skeleton, thickness)
    cv2.line(img, le, lhand, color_skeleton, thickness)
    cv2.line(img, (rs_x, sh_y), re, color_skeleton, thickness)
    cv2.line(img, re, rhand, color_skeleton, thickness)

    # Cabeza
    cv2.circle(img, (head_cx, head_cy), head_r, color_skeleton, thickness)
    cv2.circle(img, (head_cx, head_cy), head_r, color_skeleton, thickness - 1)

    # Articulaciones del torso
    for pt in [(neck_x, neck_y), (ls_x, sh_y), (rs_x, sh_y), le, re, (lh_x, hip_y), (rh_x, hip_y)]:
        cv2.circle(img, pt, 5, color_joint, -1)

    # Manos destacadas (el elemento semánticamente relevante de la seña)
    cv2.circle(img, lhand, 12, color_hand_fill, -1)
    cv2.circle(img, lhand, 12, color_hand, thickness)
    cv2.circle(img, rhand, 12, color_hand_fill, -1)
    cv2.circle(img, rhand, 12, color_hand, thickness)

    # Label
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        text_size = cv2.getTextSize(label, font, scale, 1)[0]
        tx = (width - text_size[0]) // 2
        ty = height - 18
        cv2.putText(img, label, (tx, ty), font, scale, (60, 70, 90), 1, cv2.LINE_AA)

    return img


def _draw_text_center(img, text, cx, cy):
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = cv2.getTextSize(text, font, 0.6, 1)[0]
    cv2.putText(img, text, (cx - size[0]//2, cy), font, 0.6, (120, 120, 120), 1)

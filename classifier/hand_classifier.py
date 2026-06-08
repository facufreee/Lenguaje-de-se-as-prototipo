# classifier/hand_classifier.py
# Clasificador de gestos para el Módulo 1: señas a texto.
#
# DECISION DE DISEÑO:
# Se usa un enfoque de reglas geométricas sobre landmarks de MediaPipe
# en lugar de un clasificador ML entrenado (SVM, red neuronal).
# RAZON: no existen datasets públicos de LSA con suficiente cobertura
# para entrenar un clasificador robusto. Usar sklearn con 10 ejemplos
# por seña daría resultados engañosamente confiables.
#
# Este enfoque basado en reglas es:
# - Honesto sobre sus límites
# - Explicable académicamente
# - Extensible (agregar señas = agregar reglas)
# - No requiere datos de entrenamiento
#
# En una segunda etapa, podría reemplazarse con un clasificador KNN/SVM
# entrenado sobre un dataset real capturado con MediaPipe.

import numpy as np


class HandClassifier:
    """
    Clasifica gestos de manos usando landmarks de MediaPipe.
    Retorna la seña más probable del vocabulario cerrado.
    """

    def __init__(self, confidence_threshold: float = 0.55):
        self.confidence_threshold = confidence_threshold
        # Cada regla es una función que recibe los 21 landmarks (x, y, z)
        # y retorna un score [0.0, 1.0]
        self._rules = {
            "hola":       self._rule_hola,
            "gracias":    self._rule_gracias,
            "ayuda":      self._rule_ayuda,
            "si":         self._rule_si,
            "no":         self._rule_no,
            "medico":     self._rule_medico,
            "interprete": self._rule_interprete,
            "banio":      self._rule_banio,
            "esperar":    self._rule_esperar,
            "documento":  self._rule_documento,
        }

    def classify(self, landmarks) -> dict:
        """
        Clasifica un conjunto de landmarks de MediaPipe Hands.

        Args:
            landmarks: lista de 21 objetos con atributos x, y, z (normalizados 0-1)

        Returns:
            dict con keys: 'label', 'key', 'confidence', 'all_scores'
        """
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

        scores = {}
        for key, rule_fn in self._rules.items():
            try:
                scores[key] = float(rule_fn(pts))
            except Exception:
                scores[key] = 0.0

        best_key = max(scores, key=scores.get)
        best_score = scores[best_key]

        if best_score < self.confidence_threshold:
            return {
                "label": "No reconocida",
                "key": None,
                "confidence": best_score,
                "all_scores": scores,
            }

        from utils.vocabulary import VOCAB
        return {
            "label": VOCAB[best_key]["label"],
            "key": best_key,
            "confidence": best_score,
            "all_scores": scores,
        }

    # ------------------------------------------------------------------
    # Helpers geométricos
    # ------------------------------------------------------------------

    def _fingers_extended(self, pts) -> list:
        """
        Retorna lista de 5 bools: [pulgar, índice, medio, anular, meñique]
        True si el dedo está extendido.
        """
        # Índices MediaPipe: punta=4,8,12,16,20 / PIP=3,7,11,15,19 / MCP=2,6,10,14,18
        tips  = [4, 8, 12, 16, 20]
        pips  = [3, 7, 11, 15, 19]
        mcps  = [2, 6, 10, 14, 18]

        extended = []

        # Pulgar: comparación horizontal con MCP
        thumb_ext = pts[4][0] < pts[3][0]  # mano derecha: punta más a la izquierda que articulación
        extended.append(thumb_ext)

        # Otros dedos: punta más arriba (y menor) que PIP
        for i in range(1, 5):
            tip_y = pts[tips[i]][1]
            pip_y = pts[pips[i]][1]
            extended.append(tip_y < pip_y - 0.02)

        return extended

    def _hand_height(self, pts) -> float:
        """Altura de la mano (muñeca a dedo medio)."""
        return abs(pts[0][1] - pts[12][1])

    def _wrist_y(self, pts) -> float:
        return pts[0][1]

    def _spread(self, pts) -> float:
        """Dispersión horizontal de las puntas."""
        tips_x = [pts[i][0] for i in [8, 12, 16, 20]]
        return max(tips_x) - min(tips_x)

    # ------------------------------------------------------------------
    # Reglas por seña
    # ------------------------------------------------------------------

    def _rule_hola(self, pts) -> float:
        """Mano abierta, palma al frente, posición alta → saludo."""
        ext = self._fingers_extended(pts)
        fingers_open = sum(ext[1:])  # dedos 1-4 extendidos
        hand_high = self._wrist_y(pts) < 0.45  # mano en la mitad superior
        spread = self._spread(pts)

        score = (fingers_open / 4) * 0.5 + (0.3 if hand_high else 0) + min(spread * 2, 0.2)
        return min(score, 0.95)

    def _rule_gracias(self, pts) -> float:
        """Mano plana tocando el pecho (posición central-baja)."""
        ext = self._fingers_extended(pts)
        fingers_together = self._spread(pts) < 0.08
        hand_center = 0.35 < self._wrist_y(pts) < 0.70
        open_hand = sum(ext[1:]) >= 3

        score = (0.4 if fingers_together else 0) + (0.35 if hand_center else 0) + (0.25 if open_hand else 0)
        return score

    def _rule_ayuda(self, pts) -> float:
        """Puño cerrado con pulgar arriba — señal de necesidad."""
        ext = self._fingers_extended(pts)
        fist = sum(ext[1:]) <= 1  # la mayoría de dedos cerrados
        thumb_up = ext[0]

        score = (0.55 if fist else 0) + (0.40 if thumb_up else 0)
        return score

    def _rule_si(self, pts) -> float:
        """Puño que asiente — dedo índice semiflexionado, movimiento hacia abajo."""
        ext = self._fingers_extended(pts)
        # Índice semiflexionado, resto cerrado
        index_semi = not ext[1] and not ext[2] and not ext[3] and not ext[4]
        hand_mid = 0.30 < self._wrist_y(pts) < 0.65

        score = (0.6 if index_semi else 0) + (0.35 if hand_mid else 0)
        return score

    def _rule_no(self, pts) -> float:
        """Dedo índice extendido oscilando — negación."""
        ext = self._fingers_extended(pts)
        only_index = ext[1] and not ext[2] and not ext[3] and not ext[4]

        score = (0.75 if only_index else 0) + 0.15
        return min(score, 0.90)

    def _rule_medico(self, pts) -> float:
        """Ambas manos (solo se evalúa una), posición alta y abierta."""
        ext = self._fingers_extended(pts)
        all_open = sum(ext[1:]) == 4
        hand_high = self._wrist_y(pts) < 0.40

        score = (0.5 if all_open else 0) + (0.45 if hand_high else 0)
        return score

    def _rule_interprete(self, pts) -> float:
        """Mano abierta moviéndose — índice y medio extendidos (V)."""
        ext = self._fingers_extended(pts)
        v_sign = ext[1] and ext[2] and not ext[3] and not ext[4]
        hand_mid = 0.25 < self._wrist_y(pts) < 0.60

        score = (0.65 if v_sign else 0) + (0.30 if hand_mid else 0)
        return score

    def _rule_banio(self, pts) -> float:
        """Mano con forma de B — dedos juntos y extendidos."""
        ext = self._fingers_extended(pts)
        four_ext = sum(ext[1:]) == 4
        spread = self._spread(pts)
        fingers_together = spread < 0.10

        score = (0.45 if four_ext else 0) + (0.45 if fingers_together else 0) + 0.10
        return score

    def _rule_esperar(self, pts) -> float:
        """Palma abierta hacia adelante, posición central — pausa."""
        ext = self._fingers_extended(pts)
        all_open = sum(ext) >= 4
        hand_center = 0.30 < self._wrist_y(pts) < 0.60
        spread = self._spread(pts) > 0.12

        score = (0.45 if all_open else 0) + (0.30 if hand_center else 0) + (0.25 if spread else 0)
        return score

    def _rule_documento(self, pts) -> float:
        """Mano plana horizontal — como si sostuviera un papel."""
        ext = self._fingers_extended(pts)
        flat = sum(ext[1:]) >= 3
        spread = self._spread(pts)
        wide = spread > 0.10

        score = (0.5 if flat else 0) + (0.35 if wide else 0) + 0.15
        return score

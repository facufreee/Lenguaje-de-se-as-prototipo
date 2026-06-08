# app.py
# Punto de entrada principal del prototipo LSA Comunicar.
# Ejecutar con: streamlit run app.py
#
# DESCRIPCION GENERAL:
# Este prototipo demuestra una interfaz bidireccional de mediación comunicacional
# para personas sordas en Argentina. Está diseñado como prueba de concepto para
# una tesina de Comunicación Social — no como producto terminado ni traductor completo.
#
# Módulo 1: señas → texto (cámara + MediaPipe + clasificador por reglas)
# Módulo 2: texto → seña (animación de esqueleto 2D por vocabulario cerrado)
#
# LIMITES DEL SISTEMA:
# - Vocabulario de 10 señas aisladas (no lenguaje continuo)
# - Precisión variable según iluminación y fondo
# - Las poses no han sido validadas por intérpretes certificados de LSA
# - Este sistema no reemplaza intérpretes humanos bajo ninguna circunstancia

import streamlit as st

st.set_page_config(
    page_title="LSA Comunicar — Prototipo",
    page_icon="🤙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS mínimo de UI ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        padding: 1rem 0 0.5rem;
        border-bottom: 1px solid #e8e8e8;
        margin-bottom: 1.5rem;
    }
    .badge-experimental {
        display: inline-block;
        font-size: 11px;
        padding: 2px 10px;
        border-radius: 12px;
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffc107;
        margin-left: 8px;
        vertical-align: middle;
    }
    .footer-note {
        font-size: 11px;
        color: #999;
        text-align: center;
        padding: 1.5rem 0 0.5rem;
        border-top: 1px solid #f0f0f0;
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #1a1a2e;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h2 style="margin:0; font-weight:500;">
    🤙 LSA Comunicar
    <span class="badge-experimental">prototipo experimental</span>
  </h2>
  <p style="margin:4px 0 0; color:#666; font-size:14px;">
    Herramienta de mediación comunicacional · Lengua de Señas Argentina · Tesina de Comunicación Social
  </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar global ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## LSA Comunicar")
    st.markdown("**Prototipo experimental**  \nTesina de Comunicación Social")
    st.markdown("---")
    st.markdown("**Sobre este sistema**")
    st.info(
        "Este prototipo demuestra el concepto de traducción bidireccional LSA ↔ texto. "
        "Trabaja solo con un vocabulario cerrado de 10 señas orientadas a "
        "situaciones cotidianas de necesidad comunicacional en Argentina."
    )
    st.warning(
        "⚠️ **No es un traductor completo.** "
        "No reemplaza intérpretes certificados de LSA. "
        "Precisión limitada según condiciones del entorno."
    )
    st.markdown("---")
    st.markdown("**Vocabulario disponible**")
    from utils.vocabulary import VOCAB
    for key, info in VOCAB.items():
        st.markdown(f"- {info['emoji']} {info['label']}")

# ── Tabs principales ─────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🤙 Seña → Texto",
    "⌨️ Texto → Seña",
    "ℹ️ Sobre el prototipo",
])

with tab1:
    from modules.mod1_senia_a_texto import run as run_mod1
    run_mod1()

with tab2:
    from modules.mod2_texto_a_senia import run as run_mod2
    run_mod2()

with tab3:
    st.markdown("## Sobre este prototipo")

    st.markdown("""
### Contexto académico
Este sistema fue desarrollado como prueba de concepto para una tesina de Comunicación Social.
Su objetivo no es resolver la Lengua de Señas Argentina (LSA) en su totalidad, sino demostrar
cómo podría funcionar una herramienta de mediación comunicacional accesible entre personas
sordas y oyentes en Argentina.

### Decisiones de diseño
""")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
**¿Por qué vocabulario cerrado?**
Un clasificador entrenado sobre toda la LSA requeriría:
- Miles de ejemplos por seña
- Validación con intérpretes certificados
- Consideración de variaciones regionales de LSA

Para un prototipo académico, 10 señas bien delimitadas
son más honestas que 100 señas mal clasificadas.

**¿Por qué MediaPipe?**
MediaPipe Hands es open source, corre localmente (sin APIs de pago),
devuelve 21 puntos de landmarks por mano, y funciona en tiempo real.
Es el estándar en investigación de reconocimiento de gestos.
""")

    with col2:
        st.markdown("""
**¿Por qué esqueleto 2D en lugar de avatar 3D?**
Los avatares 3D de LSA requieren:
- Modelos de animación certificados (no disponibles públicamente para LSA)
- Integración con motores 3D (Unity, Blender)
- Datos de captura de movimiento con intérpretes reales

El esqueleto 2D es honesto sobre lo que es: una representación
simplificada con valor demostrativo, no una transcripción fiel de LSA.

**¿Por qué reglas en vez de ML?**
No existen datasets públicos suficientes de LSA.
Entrenar un modelo con 5–10 ejemplos por seña produciría
confianza falsa. Las reglas geométricas son transparentes y
explicables académicamente.
""")

    st.markdown("---")
    st.markdown("### Límites éticos y técnicos")

    st.error("""
**Límites que deben quedar claros en cualquier presentación de este sistema:**

1. Las poses del esqueleto NO fueron validadas por intérpretes certificados de LSA
2. La LSA tiene variaciones regionales que este sistema no contempla
3. La comunicación continua y el contexto pragmático no están implementados
4. El sistema solo funciona bien con buena iluminación y fondo limpio
5. Este prototipo no debe usarse en situaciones de urgencia real
6. El uso de IA en lengua de señas plantea preguntas éticas sobre quién define qué es una seña "correcta"
""")

    st.markdown("### Próximos pasos para una segunda etapa")

    steps = [
        ("Captura de dataset real", "Filmar a intérpretes certificados de LSA con MediaPipe para crear un dataset de entrenamiento verificado."),
        ("Clasificador ML entrenado", "Reemplazar las reglas geométricas por un clasificador KNN o SVM entrenado sobre landmarks reales."),
        ("Detección de dos manos", "Muchas señas de LSA requieren ambas manos. El prototipo actual solo procesa una."),
        ("Secuencias temporales", "Implementar ventanas deslizantes para detectar señas en movimiento continuo (no solo poses estáticas)."),
        ("Avatar validado", "Colaborar con diseñadores y la comunidad sorda para producir un avatar 2D con animaciones verificadas."),
        ("Evaluación comunitaria", "Involucrar a personas sordas en pruebas de usuario reales para validar utilidad y precisión."),
    ]

    for title, desc in steps:
        with st.expander(f"**{title}**"):
            st.markdown(desc)

    st.markdown("""
---
<div class="footer-note">
LSA Comunicar · Prototipo experimental · Tesina de Comunicación Social · Argentina<br>
No usar como herramienta clínica, legal o de emergencia.
</div>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-note">
Prototipo experimental · Vocabulario cerrado de 10 señas · No reemplaza intérpretes certificados de LSA
</div>
""", unsafe_allow_html=True)

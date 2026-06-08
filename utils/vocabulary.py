# utils/vocabulary.py
# Vocabulario cerrado del prototipo LSA.
# Este archivo centraliza las 10 unidades comunicacionales disponibles
# y las poses de esqueleto simplificadas para representación visual.
# LIMITE EXPLICITO: esto no cubre LSA completa. Es un recorte deliberado para prototipo.

VOCAB = {
    "hola":       {"label": "Hola",                "emoji": "👋", "description": "Saludo inicial"},
    "gracias":    {"label": "Gracias",             "emoji": "🙏", "description": "Agradecimiento"},
    "ayuda":      {"label": "Ayuda",               "emoji": "🆘", "description": "Solicitud de asistencia"},
    "si":         {"label": "Sí",                  "emoji": "✅", "description": "Afirmación"},
    "no":         {"label": "No",                  "emoji": "❌", "description": "Negación"},
    "medico":     {"label": "Necesito médico",     "emoji": "🏥", "description": "Urgencia médica"},
    "interprete": {"label": "Necesito intérprete", "emoji": "🤝", "description": "Solicitud de intérprete LSA"},
    "banio":      {"label": "Baño",                "emoji": "🚻", "description": "Necesidad de baño"},
    "esperar":    {"label": "Esperar",             "emoji": "⏳", "description": "Indicación de espera"},
    "documento":  {"label": "Documento",           "emoji": "📄", "description": "Referencia a documentación"},
}

# Alias de texto para normalización en módulo 2 (texto → seña)
TEXT_ALIASES = {
    "hola": "hola",
    "gracias": "gracias",
    "ayuda": "ayuda",
    "auxilio": "ayuda",
    "si": "si",
    "sí": "si",
    "no": "no",
    "médico": "medico",
    "medico": "medico",
    "doctor": "medico",
    "intérprete": "interprete",
    "interprete": "interprete",
    "señante": "interprete",
    "baño": "banio",
    "bano": "banio",
    "sanitario": "banio",
    "esperar": "esperar",
    "espera": "esperar",
    "momento": "esperar",
    "documento": "documento",
    "dni": "documento",
    "papel": "documento",
}

VOCAB_KEYS = list(VOCAB.keys())

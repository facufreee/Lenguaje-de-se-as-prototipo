# LSA Comunicar — Prototipo Experimental

> **Prototipo de mediación comunicacional para personas sordas en Argentina**  
> Desarrollado como prueba de concepto para tesina de Comunicación Social.  
> No es un traductor completo de LSA. No reemplaza intérpretes certificados.

---

## Qué hace este prototipo

Una interfaz web bidireccional con dos módulos:

| Módulo | Función | Tecnología |
|--------|---------|------------|
| **Seña → Texto** | Detecta señas por cámara y muestra texto equivalente | OpenCV + MediaPipe Hands + clasificador por reglas geométricas |
| **Texto → Seña** | Traduce palabras del vocabulario a animación de esqueleto 2D | NumPy + OpenCV + Streamlit |

**Vocabulario cerrado (10 señas):** Hola · Gracias · Ayuda · Sí · No · Necesito médico · Necesito intérprete · Baño · Esperar · Documento

---

## Estructura del proyecto

```
lsa_comunicar/
├── app.py                        # Punto de entrada Streamlit
├── requirements.txt              # Dependencias Python
├── .streamlit/
│   └── config.toml              # Configuración de tema
├── modules/
│   ├── __init__.py
│   ├── mod1_seña_a_texto.py     # Módulo 1: cámara → texto
│   └── mod2_texto_a_seña.py     # Módulo 2: texto → esqueleto 2D
├── classifier/
│   ├── __init__.py
│   └── hand_classifier.py       # Clasificador de gestos por reglas geométricas
└── utils/
    ├── __init__.py
    ├── vocabulary.py             # Vocabulario cerrado y aliases de texto
    └── pose_renderer.py          # Renderizador de esqueleto 2D con OpenCV
```

---

## Requisitos del sistema

- Python 3.10 o superior
- Cámara web (para Módulo 1)
- Sistema operativo: Windows, macOS o Linux

---

## Instalación paso a paso

### 1. Clonar o descomprimir el proyecto

```bash
# Si descargaste el ZIP:
unzip lsa_comunicar.zip
cd lsa_comunicar
```

### 2. Crear entorno virtual (recomendado)

```bash
# Crear entorno
python -m venv venv

# Activar — macOS/Linux:
source venv/bin/activate

# Activar — Windows:
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> La primera instalación puede tardar 2–5 minutos por el tamaño de MediaPipe (~50 MB).

### 4. Ejecutar la aplicación

```bash
streamlit run app.py
```

El navegador se abre automáticamente en `http://localhost:8501`.  
Si no abre, copiá esa URL manualmente.

---

## Uso

### Módulo 1 — Seña → Texto

1. Ir a la pestaña **🤙 Seña → Texto**
2. Activar el toggle **"Iniciar cámara"**
3. Posicionarse a 40–70 cm de la cámara con buena iluminación
4. Hacer una de las señas del vocabulario
5. El sistema muestra el texto detectado y la confianza

**Tips para mejor detección:**
- Fondo claro y uniforme (pared blanca o gris)
- Buena iluminación frontal, sin contraluz
- Mover la mano despacio y sostener la pose 1–2 segundos
- Ajustar el umbral de confianza en el sidebar si hay falsos positivos

### Módulo 2 — Texto → Seña

1. Ir a la pestaña **⌨️ Texto → Seña**
2. Escribir texto en el área de texto, o usar los botones de palabras rápidas
3. Presionar **▶ Traducir a seña**
4. El avatar esqueleto anima cada seña identificada en secuencia
5. Las palabras sin cobertura quedan marcadas en amarillo

---

## Problemas comunes

| Problema | Solución |
|----------|---------|
| `No se pudo abrir la cámara` | Verificar que ninguna otra app esté usando la cámara. Probar cambiando el índice de cámara a `1` en el sidebar. |
| `ModuleNotFoundError: mediapipe` | Verificar que el entorno virtual está activo y corriste `pip install -r requirements.txt` |
| Detección muy inestable | Mejorar iluminación. Bajar el umbral de confianza a 0.45 en el slider. |
| La app no abre en el navegador | Ir manualmente a `http://localhost:8501` |
| Error en macOS con OpenCV y cámara | Dar permisos de cámara a la Terminal en Preferencias del Sistema → Privacidad |

---

## Límites explícitos del sistema

1. **Vocabulario acotado**: solo 10 señas aisladas. No cubre LSA completa.
2. **Señas estáticas**: no detecta movimiento continuo entre señas.
3. **Una mano**: el clasificador actual solo procesa una mano a la vez.
4. **Poses no verificadas**: las representaciones 2D son aproximaciones no validadas por intérpretes certificados de LSA.
5. **Sin variaciones regionales**: la LSA tiene diferencias entre provincias que no están contempladas.
6. **Sin contexto pragmático**: el sistema no entiende oraciones, solo palabras aisladas.

---

## Próximos pasos para una segunda etapa

1. Capturar dataset real con intérpretes certificados de LSA
2. Reemplazar reglas geométricas por clasificador KNN/SVM entrenado
3. Implementar detección de dos manos simultáneas
4. Agregar ventanas temporales para lenguaje continuo
5. Colaborar con la comunidad sorda para validar el sistema
6. Evaluar con usuarios reales en contextos situados (hospital, banco, escuela)

---

## Consideraciones éticas

El uso de inteligencia artificial en lenguas de señas plantea preguntas que van más allá de lo técnico:

- ¿Quién tiene autoridad para definir qué es una seña "correcta"?
- ¿Cómo se respeta la identidad cultural de la comunidad sorda?
- ¿Puede un sistema automatizado reemplazar la riqueza de la comunicación mediada por intérpretes?

Este prototipo asume que **no puede ni debe reemplazar intérpretes humanos**, sino facilitar interacciones cotidianas básicas mientras se accede a mediación humana adecuada.

---

*Prototipo experimental · Tesina de Comunicación Social · Argentina*

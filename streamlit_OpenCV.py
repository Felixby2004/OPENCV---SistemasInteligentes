import streamlit as st
import sys
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import io
import math
import types
import time
from time import sleep
import _pickle as pickle
from scipy.special import softmax
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from av import VideoFrame

st.set_page_config(
    page_title="Detección de Rostros",
    layout="centered",
    initial_sidebar_state="collapsed"
)

opciones = [
    'Portada',
    'Capítulo 1 - Tranformaciones Geométricas a imágenes', 
    'Capítulo 2 - Detección de bordes y aplicación de filtros de imágen', 
    'Capítulo 3 - Caricatura de una imagen', 
    'Capítulo 4 - Detectar y traquear diferentes partes del cuerpo', # <-- ESTE es el capítulo con controles
    'Capítulo 5 - Extraer características de una imagen', 
    'Capítulo 6 - Eliminador de Objetos', 
    'Capítulo 7 - Detectar formas y segmentación de imagen', 
    'Capítulo 8 - Seguimiento de objetos', 
    'Capítulo 9 - Reconocimiento de objetos',
    'Capítulo 10 - Realidad Aumentada',
    'Capítulo 11 - Machine Learning por una Red Neuronal Artificial',
]


if 'page' not in st.session_state:
    st.session_state.page = opciones[0] # Inicializa en 'Portada'

# --- DIBUJAR EL SELECTBOX DE NAVEGACIÓN EN EL SIDEBAR ---
# Este es el único elemento variable del sidebar que debe estar aquí

st.sidebar.title("Índice")
st.sidebar.write("Selecciona un encabezado:")

# Usar el selectbox para actualizar el estado de la página
selected_chapter = st.sidebar.selectbox(
    " ", 
    opciones,
    key="chapter_selector",
    # Cuando el valor cambia, actualiza el estado de la sesión
    on_change=lambda: st.session_state.__setitem__('page', st.session_state.chapter_selector)
)

# --- CSS para la portada
def estilos():
    """Aplica estilos CSS con fondo de gradiente morado, sidebar sólido, y acentos morados/ámbar."""
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            /* Fondo SÓLIDO (Morado/Índigo oscuro) */
            background: linear-gradient(135deg, #a4508b, #5f0a87);
            color: #FFFFFF;
        }
                
        h1, h2, h3 {
            color: #8C9EFF; /* Título morado/índigo brillante */
        }
        
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label {
            color: #E0E0FF; /* Color claro para los textos y labels del sidebar */
        }

        .chapter-box {
            color: #ffffff;
            background: linear-gradient(135deg, #a4508b, #5f0a87);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }

        .chapter-title {
            font-size: 2em;
            font-weight: 700;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 1px solid #FFFFFF; 
        }

        [data-testid="stMetricValue"] {
            color: #FFC107;
        }
        .author-box {
            font-size: 1.5em;
            margin-bottom: 20px;
            padding-bottom: 5px;
            border-bottom: 2px solid #a4508b;
            text-align: right;
        }
        .cover-title {
            font-size: 3em;
            font-weight: 700;
            color: #FFFFFF;
            background: linear-gradient(135deg, #a4508b, #5f0a87);
            text-align: center;
            margin: 50px 0;
            padding: 20px;
            border-radius: 10px;
        }
        .centro {
            text-align: center;
            margin-top: 50px;
            font-size: 1.2em;
            color: #C5CAE9; /* Lila suave */
        }
        .img-mediana {
            width: 150px;
            margin-top: 20px;
            filter: drop-shadow(0 0 10px #8C9EFF); /* Sombra para resaltar el logo */
        }
        a {
            color: #8C9EFF !important; /* Enlaces en el color de acento */
        }    
        .main-footer {
            left: 0;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 10px 0;
            font-size: 0.9em;
            border-top: 3px solid #a4508b;
        }
                

        .streamlit-expanderHeader {
            /* Fondo del encabezado (Green Teal Oscuro, #008080 es un buen contraste) */
            background-color: #008080; 
            color: #FFFFFF !important; /* Texto blanco en el encabezado */
            padding: 10px;
            border-radius: 8px; /* Bordes redondeados */
            border: 1px solid #006969; /* Borde del mismo color que el sidebar */
            font-weight: bold;
            font-size: 1.1em;
            margin-top: 15px; /* Espacio arriba para separarlo del contenido anterior */
        }

        /* Estilo para el contenido cuando se despliega (para que use el fondo oscuro) */
        [data-testid="stVerticalBlock"] > [data-testid="stExpander"] > div:nth-child(2) {
            background-color: #1A1A40; /* Fondo oscuro del cuerpo (similar al fondo degradado) */
            border: 1px solid #008080; /* Borde coincidente */
            border-top: none; /* Quitamos el borde superior para que se pegue al encabezado */
            border-radius: 0 0 8px 8px; /* Solo redondeamos la parte inferior */
            padding: 15px;
            color: #C5CAE9; /* Texto claro */
        }
        </style>
        """, unsafe_allow_html=True)

    
def mostrarContenido(opcion):
    estilos()
    if opcion == 'Portada':
        mostrar_portada()
    else:
        if opcion == "Capítulo 1 - Tranformaciones Geométricas a imágenes":
            capitulo1()

        elif opcion == "Capítulo 2 - Detección de bordes y aplicación de filtros de imágen":
            capitulo2()

        elif opcion == "Capítulo 3 - Caricatura de una imagen":
            capitulo3()

        elif opcion == "Capítulo 4 - Detectar y traquear diferentes partes del cuerpo":
            capitulo4()

        elif opcion == "Capítulo 5 - Extraer características de una imagen":
            capitulo5()

        elif opcion == "Capítulo 6 - Eliminador de Objetos":
            capitulo6()

        elif opcion == "Capítulo 7 - Detectar formas y segmentación de imagen":
            capitulo7()

        elif opcion == "Capítulo 8 - Seguimiento de objetos":
            capitulo8()

        elif opcion == "Capítulo 9 - Reconocimiento de objetos":
            capitulo9()

        elif opcion == "Capítulo 10 - Realidad Aumentada":
            capitulo10()

        elif opcion == "Capítulo 11 - Machine Learning por una Red Neuronal Artificial":
            capitulo11()

        st.markdown("<div class='main-footer'>2025 - Desarrollado por Chávez Vidal, Felix Andreé 😎<br> Universidad Nacional de Trujillo</div>", unsafe_allow_html=True)


def mostrar_portada():
    estilos()
    # Usar columnas para centrar el contenido y limitar el ancho de la "portada"
    col_left, col_center, col_right = st.columns([1, 4, 1])
    with col_center:
        st.markdown(
            """
            <div class="author-box">6to ciclo - Sistemas Inteligentes</div>
            <div class="cover-title">Open CV 3.x with Python</div>
            11 Programas desarrollados con la biblioteca OpenCV.<br>
            Basados en el libro <a href="https://www.packtpub.com/en-us/product/opencv-3x-with-python-by-example-9781788396905" target="_blank"><i>OpenCV 3.x with Python by Example - Gabriel Garrido, Prateek Joshi</i></a>.<br><br>
            <div class="main-footer">
                Desarrollado por Chávez Vidal, Felix Andreé 😎<br>
                Universidad Nacional de Trujillo
                <img class="img-mediana" src="https://upload.wikimedia.org/wikipedia/commons/6/6e/Universidad_Nacional_de_Trujillo_-_Per%C3%BA_vector_logo.png">
            </div>
            """,
            unsafe_allow_html=True
        )


def capitulo1():
    st.markdown(
        """
        <div class="chapter-box">
            <div class="chapter-title">Capítulo 1 — Transformaciones Geométricas a Imágenes</div>
            Se aplicará distorsiones geométricas como ondas (ondas sinusoidales) o remolinos a la imagen. 
            Usted puede ajustar <b>los parámetros de amplitud</b> en la barra lateral para modificar la forma y crear 
            efectos visuales dinámicos en los resultados.
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Qué son Amplitudes"):
        st.markdown(
            """
            <div class="st-accent-card">
                <h4 style='color: #8C9EFF;'>Amplitud Vertical:</h4>
                <p>La Amplitud Vertical define la fuerza con la que los píxeles son empujados en dirección horizontal a lo largo de la imagen.</p>
                <ul>
                    <li>
                        <strong>Lo que Controla</strong>: La magnitud del desplazamiento en el eje X (horizontal).
                    </li>
                    <li>
                        <strong>Patrón Visual</strong>: Crea ondas que se propagan verticalmente (de arriba hacia abajo) a través de la imagen.
                    </li>
                    <li>
                        <strong>En el código</strong>: Al aumentar la Amplitud Vertical, los píxeles se desplazan más lateralmente (izquierda o derecha), resultando en una imagen más "ondulada" a lo largo de las columnas.
                    </li>
                </ul>
                <br>
                <h4 style='color: #8C9EFF;'>Amplitud Horizontal:</h4>
                <p>La Amplitud Horizontal define la fuerza con la que los píxeles son empujados en dirección vertical a lo largo de la imagen.</p>
                <ul>
                    <li>
                        <strong>Lo que Controla</strong>: La magnitud del desplazamiento en el eje Y (vertical).
                    </li>
                    <li>
                        <strong>Patrón Visual</strong>: Genera ondulaciones que se extienden horizontalmente (de izquierda a derecha) a lo largo de la imagen.
                    </li>
                    <li>
                        <strong>En el código</strong>: Al aumentar la Amplitud Horizontal, los píxeles se desplazan más hacia arriba o hacia abajo, dando como resultado una imagen que parece estirarse o encogerse a lo largo de las filas.
                    </li>
                </ul>
                <br>
                <h4 style='color: #8C9EFF;'>Amplitud Concava:</h4>
                <p>La Amplitud Concava es el parámetro clave que controla la magnitud de una distorsión de tipo barril o cóncava.</p>
                <ul>
                    <li>
                        <strong>Efecto</strong>: Deforma la imagen de tal manera que las líneas centrales parecen estar "hundidas" o empujadas hacia el interior (como si se viera el objeto a través de una lente cóncava o un espejo deformante).
                    </li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- 2. Carga de Imagen ---
    opcion = st.radio("Selecciona una opción:", ["📂 Subir imagen", "📄 Imagen por defecto"])

    if opcion == "📂 Subir imagen":
        archivo = st.file_uploader("", type=["png", "jpg", "jpeg"])
        if archivo is not None:
            img_pil_color = Image.open(archivo)  # Imagen original en color (PIL)
            # Conversión a escala de grises para los efectos
            img = np.array(img_pil_color.convert("L"))
            st.image(img_pil_color, caption="Imagen cargada")
        else:
            st.warning("Por favor, sube una imagen para continuar.")
            # st.stop() # No es recomendable usar st.stop() con archivos, mejor usar 'return'
            return
    
    else:
        # Cargar la imagen por defecto
        # Se asume que 'input.jpg' existe en el directorio
        img_bgr = cv2.imread("input.jpg")
        if img_bgr is None:
            st.error("No se encontró la imagen por defecto 'input.jpg'.")
            return
            
        # Conversión a escala de grises para efectos
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # Conversión a RGB y luego a PIL para mostrar con Streamlit
        img_pil_color = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        st.image(img_pil_color, caption="Imagen por defecto")

    rows, cols = img.shape

    st.subheader("⚙️ Controles de Amplitudes y Efectos")
    vert_amp = st.slider("↕️ Amplitud Onda Vertical", 0, 100, 30)
    horiz_amp = st.slider("↔️ Amplitud Onda Horizontal", 0, 100, 30)
    concave_amp = st.slider("😵‍💫 Amplitud Efecto Cóncavo", 0, 200, 128)

    st.subheader("🔍 Resultados")

    # --- 2. Procesamiento para UN SOLO Resultado ---

    img_final = np.zeros(img.shape, dtype=img.dtype)

    # Paso 2: Aplicar la combinación de Onda Vertical y Horizontal (similar a la multidireccional)
    for i in range(rows):
        for j in range(cols):
            # La onda vertical afecta la coordenada X (j)
            offset_x = int(vert_amp * math.sin(2 * math.pi * i / 180)) 
            # La onda horizontal afecta la coordenada Y (i)
            offset_y = int(horiz_amp * math.cos(2 * math.pi * j / 150))
            
            src_i = i + offset_y
            src_j = j + offset_x
            
            # Asignar el píxel de la fuente a la imagen final temporalmente
            if 0 <= src_i < rows and 0 <= src_j < cols:
                img_final[i, j] = img[src_i, src_j]
            else:
                img_final[i, j] = 0 # Borde negro

    # Paso 3: Aplicar el efecto Cóncavo sobre el resultado de la onda (img_final)
    # Creamos una copia para el efecto cóncavo, usando el resultado de la onda como fuente
    img_compuesta = np.zeros(img.shape, dtype=img.dtype)
    for i in range(rows):
        for j in range(cols):
            # El efecto cóncavo (usando 'concave_amp') modifica la coordenada X (j)
            offset_x_concave = int(concave_amp * math.sin(2 * math.pi * i / (2 * cols)))
            
            src_j_concave = j + offset_x_concave
            
            # Asignar el píxel de la fuente (img_final, el resultado de la onda)
            if 0 <= src_j_concave < cols:
                # NOTA: Usamos img_final como fuente, y la asignamos a img_compuesta
                img_compuesta[i, j] = img_final[i, src_j_concave] 
            else:
                img_compuesta[i, j] = 0 # Borde negro

    st.image(img_compuesta, caption="Imagen Modificada")


def capitulo2():
    st.markdown(
        """
        <div class="chapter-box">
            <div class="chapter-title">Capítulo 2 - Detección de bordes y aplicación de filtros de imágen</div>
            Se aplicará converción a escala de grises. Luego, aplica dos transformaciones fundamentales de Morfología Matemática: Erosión y Dilatación. Además usted puede ajustar <b>los parámetros</b> que son el tamaño del kernel (ancho y alto) y el número de iteraciones para los efecto de las transformaciones.
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Qué es la Morfología Matemática"):
        st.markdown(
            """
            La Morfología Matemática es un método de procesamiento de imágenes y análisis de estructuras geométricas que emplea operadores basados en la teoría de conjuntos y conceptos topológicos para manipular y extraer información de objetos.
            <div class="st-accent-card">
                <h4 style='color: #8C9EFF;'>Erosión:</h4>
                <p>La Amplitud Vertical define la fuerza con la que los píxeles son empujados en dirección horizontal a lo largo de la imagen.</p>
                <ul>
                    <li>
                        <strong>Efecto</strong>: Reduce o encoge las regiones de color claro (objetos) y agranda los espacios oscuros (agujeros/fondo).
                    </li>
                    <li>
                        <strong>Mecanismo</strong>: El píxel solo permanece claro si el Kernel encaja completamente en el área clara. Esto "come" los bordes de los objetos.
                    </li>
                    <li>
                        <strong>Uso</strong>: Eliminar ruido, adelgazar objetos o separar objetos conectados.
                    </li>
                </ul>
                <br>
                <h4 style='color: #8C9EFF;'>Dilatación:</h4>
                <ul>
                    <li>
                        <strong>Efecto</strong>: Expande o agranda las regiones de color claro y rellena pequeños agujeros u oclusiones.
                    </li>
                    <li>
                        <strong>Mecanismo</strong>: El píxel se vuelve claro si el Kernel toca al menos un píxel claro. Esto hace que los objetos crezcan.
                    </li>
                    <li>
                        <strong>Uso</strong>: Unir componentes separados o rellenar huecos.
                    </li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    opcion = st.radio("Selecciona una opción:", ["📂 Subir imagen", "📄 Imagen por defecto"])

    if opcion == "📂 Subir imagen":
        archivo = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])
        if archivo is not None:
            img_pil = Image.open(archivo).convert("RGB")
            img = np.array(img_pil)
            st.image(img, caption="Imagen cargada")
        else:
            st.warning("Por favor, sube una imagen para continuar.")
            st.stop()
    else:
        img_bgr = cv2.imread("tree_input.jpg")
        if img_bgr is None:
            st.error("No se encontró 'tree_input.jpg' en el directorio.")
            st.stop()
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.image(img, caption="Imagen por defecto")

    st.subheader("⚙️ Controles de Kernel e Iteraciones")
    
    col_k_x, col_k_y, col_iter = st.columns(3)

    with col_k_x:
        kernel_x = st.slider("↔️ Ancho del Kernel", 1, 30, 5)
    with col_k_y:
        kernel_y = st.slider("↕️ Alto del Kernel", 1, 30, 10)
    with col_iter:
        iteraciones = st.slider("✏️ Iteraciones", 1, 10, 1)

    # --- Procesamiento ---
    # El kernel debe tener dimensiones impares y mayores que cero, pero el slider ya lo controla.
    kernel = np.ones((kernel_y, kernel_x), np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=iteraciones)
    img_dilation = cv2.dilate(img, kernel, iterations=iteraciones)

    # --- Visualización en una sola fila de 3 columnas ---
    st.subheader("🔍 Resultados")
    
    col1, col2, col3 = st.columns(3)

    col1, col2 = st.columns(2)
    with col1:
        caption_erosion = "Imagen con Erosión"
        st.markdown("**1. Erosión**", unsafe_allow_html=True)
        st.image(img_erosion, caption=caption_erosion)

    with col2:
        caption_dilation = "Imagen con Dilatación"
        st.markdown("**2. Dilatación**", unsafe_allow_html=True)
        st.image(img_dilation, caption=caption_dilation)

    
def capitulo3():
    def cartoonize_image(img, ksize=5, sketch_mode=True):
        num_repetitions, sigma_color, sigma_space, ds_factor = 10, 5, 7, 4
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.medianBlur(img_gray, 7)
        edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=ksize)
        ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
        if sketch_mode:
            return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        img_small = cv2.resize(img, None, fx=1.0/ds_factor, fy=1.0/ds_factor, interpolation=cv2.INTER_AREA)
        for i in range(num_repetitions):
            img_small = cv2.bilateralFilter(img_small, ksize, sigma_color, sigma_space)
        img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_LINEAR)
        dst = cv2.bitwise_and(img_output, img_output, mask=mask)
        return dst
    
    st.markdown(
        """
        <div class="chapter-box">
            <div class="chapter-title">Capítulo 3 - Caricatura de una imagen</div>
            Al resultado se aplicará una escala de grises y un filtro llamado ksize que siempre debe tener valor impar sino daria un error. Además usted puede ajustar <b>los parámetros</b> del tamaño del filtro. Puede elegir entre subir una imagen, usar la imagen por defecto o encender su cámara y aplicarle el filtro en tiempo real.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    opcion = st.radio("Selecciona una opción:", ["📂 Subir imagen", "📄 Imagen por defecto", "📷 Usar cámara"])

    # Inicializar ksize fuera de los bloques condicionales
    ksize = 5 
    
    if opcion == "📂 Subir imagen":
        archivo = st.file_uploader("Sube tu imagen", type=["jpg", "jpeg", "png"])
        if archivo is not None:
            img_pil = Image.open(archivo).convert("RGB")
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # 1. Mostrar Imagen Original inmediatamente
            st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption="Imagen cargada")
            
            # 2. Mostrar Control (st.select_slider)
            ksize = st.select_slider(
                "👀 Tamaño del filtro (ksize)",
                options=[1, 3, 5, 7, 9, 11, 13, 15, 17],
                value=5
            )

            # 3. Procesamiento y Resultados
            st.subheader("🔍 Resultados")
            result = cartoonize_image(img_cv, ksize=ksize, sketch_mode=True)
            
            # SOLO se muestra el resultado
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Resultado Cartoon/Sketch")
        else:
            st.warning("Sube una imagen para continuar.")


    elif opcion == "📄 Imagen por defecto":
        img_bgr = cv2.imread("road.jpg")
        if img_bgr is not None:
            
            # 1. Mostrar Imagen Original inmediatamente
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Imagen por defecto")
            
            # 2. Mostrar Control (st.select_slider)
            ksize = st.select_slider(
                "👀 Tamaño del filtro (ksize)",
                options=[1, 3, 5, 7, 9, 11, 13, 15, 17],
                value=5
            )

            # 3. Procesamiento y Resultados
            st.subheader("🔍 Resultados")
            result = cartoonize_image(img_bgr, ksize=ksize, sketch_mode=True)
            
            # SOLO se muestra el resultado
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Resultado Cartoon/Sketch")
        else:
            st.error("No se encontró 'road.jpg'")
    
    elif opcion == "📷 Usar cámara":
        ksize = st.select_slider(
            "👀 Tamaño del filtro (ksize)",
            options=[1,3,5,7,9,11,13,15,17],
            value=5
        )
        
        FRAME_WINDOW = st.empty()
        CAMERA_SLOT = st.empty()  # Aquí pondremos el st.camera_input

        img_file = CAMERA_SLOT.camera_input("")
        
        if img_file is not None:
            # Ocultar el widget de cámara
            CAMERA_SLOT.empty()  # Borra el st.camera_input del layout
        
            # Procesar la imagen
            img = Image.open(img_file)
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            cartoon_frame = cartoonize_image(frame, ksize=ksize)
        
            combined = np.hstack([
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(cartoon_frame, cv2.COLOR_BGR2RGB)
            ])
            FRAME_WINDOW.image(combined, channels="RGB", use_column_width=True)
        else:
            st.info("📷 Apunta tu cámara y toma una foto para ver el efecto.")


global face_cascade_global, glasses_img_global
face_cascade_global, eye_cascade_global, glasses_img_global = None, None, None    

def overlay_image_alpha(lentes, frame, x, y, w, h):
    # 1. Redimensionar la imagen de los lentes al tamaño deseado (sin recorte)
    lentes_resized = cv2.resize(lentes, (w, h), interpolation=cv2.INTER_AREA)

    # 2. Definir los límites de la Región de Interés (ROI) en el FRAME
    # Usamos max/min para asegurar que la ROI esté dentro de los límites del frame
    y1 = max(0, y)
    y2 = min(frame.shape[0], y + h)
    x1 = max(0, x)
    x2 = min(frame.shape[1], x + w)
    
    # Si la ROI es inválida (fuera de la pantalla), salimos
    if x1 >= x2 or y1 >= y2:
        return frame
        
    # 3. Definir los límites de la ROI en la IMAGEN DE LENTES
    # Si la imagen se salió por la izquierda/arriba, recortamos la imagen de lentes
    lentes_y1 = y1 - y
    lentes_x1 = x1 - x
    
    # Si la imagen se salió por la derecha/abajo, recortamos la imagen de lentes
    lentes_y2 = y2 - y
    lentes_x2 = x2 - x

    # 4. Obtener las porciones de imagen correctas (asegurando dimensiones iguales)
    lentes_bgr = lentes_resized[lentes_y1:lentes_y2, lentes_x1:lentes_x2, :3]
    alpha_s = lentes_resized[lentes_y1:lentes_y2, lentes_x1:lentes_x2, 3] / 255.0
    
    # Crear la ROI del fotograma para la superposición (solo si tiene 3 canales)
    roi = frame[y1:y2, x1:x2]

    # 5. Si la imagen de lentes no tiene canal alfa, solo hacemos una superposición simple.
    if lentes_resized.shape[2] < 4:
         frame[y1:y2, x1:x2] = lentes_resized
         return frame
         
    # 6. Calcular alpha_l (debe tener las mismas dimensiones que alpha_s)
    alpha_l = 1.0 - alpha_s

    # 7. Superposición: (Lentes * Alfa) + (Fondo * (1 - Alfa))
    for c in range(0, 3):
        frame[y1:y2, x1:x2, c] = (alpha_s * lentes_bgr[:, :, c] + alpha_l * roi[:, :, c])
    return frame

@st.cache_resource
def load_resources():
    # --- Rutas de Archivos Locales (Usando tus rutas) ---
    FACE_CASCADE_PATH = 'haarcascade_frontalface_alt.xml' 
    GLASSES_PATH = 'sunglasses.png' 
    EYE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_eye.xml'

    try:
        # Carga de recursos
        face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
        glasses_img = cv2.imread(GLASSES_PATH, cv2.IMREAD_UNCHANGED)
        
        # ... (Toda tu lógica de verificación de errores va aquí) ...

        if glasses_img is None or face_cascade.empty():
            st.error("FATAL: Error al cargar recursos. Ver los detalles en los logs.")
            return None, None, None

        return face_cascade, eye_cascade, glasses_img
        
    except Exception as e:
        st.error(f"Error al cargar recursos: {e}")
        return None, None, None

# =========================================================
# --- CLASE DE PROCESAMIENTO DE VIDEO EN VIVO (WEB RTC) ---
# =========================================================

class ARFaceOverlayTransformer(VideoTransformerBase):
    def __init__(self, face_cascade, glasses_img, overlay_func):
        self.face_cascade = face_cascade
        self.glasses_img = glasses_img
        self.overlay_func = overlay_func
        self.frame_count = 0
        self.face_rects = []
        self.DETECTION_INTERVAL = 5
        if self.face_cascade is None or self.glasses_img is None:
            raise RuntimeError("Recursos de RA no disponibles.")

    def recv(self, frame):
        # Convertir frame a ndarray BGR
        frame_bgr = frame.to_ndarray(format="bgr24")
        RESIZE_SCALE = 0.5
        
        if self.frame_count % self.DETECTION_INTERVAL == 0:
            small_frame = cv2.resize(frame_bgr, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            face_rects_small = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            self.face_rects = (face_rects_small / RESIZE_SCALE).astype(int)
            self.frame_count = 0

        self.frame_count += 1

        # Superponer lentes
        for (x, y, w, h) in self.face_rects:
            glasses_w = int(w * 1.2)
            glasses_h = int(h * 0.4)
            glasses_x = x - int(w * 0.10)
            glasses_y = y + int(h * 0.25)
            frame_bgr = self.overlay_func(
                self.glasses_img, frame_bgr, glasses_x, glasses_y, glasses_w, glasses_h
            )

        # Retornar frame convertido a VideoFrame
        from av import VideoFrame
        return VideoFrame.from_ndarray(frame_bgr, format="bgr24")

class FaceDetector(VideoTransformerBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        result = detect_faces(img)
        
        from av import VideoFrame
        return VideoFrame.from_ndarray(result, format="bgr24")


def capitulo4():        
    face_cascade, eye_cascade, glasses_img = load_resources()
    if face_cascade is None or glasses_img is None:
        # El mensaje de error ya fue impreso en load_resources()
        return
    st.markdown(
        """
        <div class="chapter-box">
            <div class="chapter-title">Capítulo 4 - Detectar y traquear diferentes partes del cuerpo</div>
            Podemos aplicar detectar y traquear (seguir) el movimiento de ciertas partes del cuerpo, en este caso es de la casa y los ojos con ayuda de documentos .xml y funciones. Lo que hará es identificar los ojos de la persona y supersoner unos lentes de sol.
        </div>
        """,
        unsafe_allow_html=True
    )

    img_file = st.camera_input("")
    if img_file is not None:
        # Convertir imagen a OpenCV BGR
        img = Image.open(img_file)
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Detectar caras
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            glasses_w = int(w * 1.2)
            glasses_h = int(h * 0.4)
            glasses_x = x - int(w * 0.10)
            glasses_y = y + int(h * 0.25)

            frame = overlay_image_alpha(glasses_img, frame, glasses_x, glasses_y, glasses_w, glasses_h)

        # Mostrar resultado en Streamlit
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)


def capitulo5():
    st.markdown(
        """
        <div class="chapter-box">
            <div class="chapter-title">Capítulo 5 - Extraer características de una imagen</div>
            <p>Apartado útil para el entendimiento de la máquina en descubrir la forma de un objeto solo viendo la imagen. Lo hace a travez de métodos Con NMS y Sin NMS.<br>
            Pondrá puntos verdes para identificar el objeto.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    with st.expander("Qué es Sin NMS"):
        st.markdown(
            """
            <h4 style='color: #8C9EFF;'>Sin Supresión No Máxima (Sin NMS):</h4>
            
            <p>Cuando se quiere detecctar una imagen, genera cientos de cajas delimitadoras (Bounding Boxes) superpuestas alrededor del mismo objeto. Esto sucede porque muchas de esas predicciones alcanzan el umbral mínimo de confianza del modelo. El resultado es una salida ruidosa y ambigua, donde es imposible distinguir visualmente la ubicación única y precisa de cada objeto.</p>
            """,
            unsafe_allow_html=True
        )

    with st.expander("Qué es Con NMS"):
        st.markdown(
            """
            <h4 style='color: #8C9EFF;'>Con Supresión No Máxima (Con  NMS):</h4>
            
            <p>El NMS actúa como un filtro de post-procesamiento. Su objetivo es garantizar que solo se muestre una sola caja para cada objeto. Funciona suprimiendo todas las demás cajas vecinas que se solapan significativamente con la mejor caja. El resultado final es un conjunto limpio de detecciones, donde cada objeto identificado tiene una caja única y bien definida, lo que lo hace útil para la interpretación por la computadora y el usuario.</p>
            """,
            unsafe_allow_html=True
        )


    # --- Selección de modo ---
    opcion = st.radio("Selecciona una opción:", ["📂 Subir imagen", "📄 Imagen por defecto"])

    if opcion == "📂 Subir imagen":
        archivo = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])
        if archivo is not None:
            img_pil = Image.open(archivo).convert("RGB")
            input_image_bgr = np.array(img_pil)
            st.image(input_image_bgr, caption="Imagen cargada")
        else:
            st.warning("Por favor, sube una imagen para continuar.")
            st.stop()
    else:
        img_bgr = cv2.imread("tool.png")
        if img_bgr is None:
            st.error("No se encontró 'tool.png' en el directorio.")
            st.stop()
        input_image_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.image(input_image_bgr, caption="Imagen por defecto")

    st.subheader("🔍 Resultados")

    try:
        # Convertir a escala de grises para la detección
        gray_image = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2GRAY)

        # 4. Inicializar el detector FAST
        fast = cv2.FastFeatureDetector_create() 
        
        # PARTE 1: DETECCIÓN CON SUPRESIÓN NO MÁXIMA (NMS)
        fast.setNonmaxSuppression(True) 
        keypoints_nms = fast.detect(gray_image, None)
        
        img_keypoints_with_nonmax_bgr = input_image_bgr.copy()
        cv2.drawKeypoints(
            input_image_bgr, 
            keypoints_nms, 
            img_keypoints_with_nonmax_bgr, 
            color=(0, 255, 0), # Verde en BGR
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        img_keypoints_with_nonmax_rgb = cv2.cvtColor(img_keypoints_with_nonmax_bgr, cv2.COLOR_BGR2RGB)
        
        
        # PARTE 2: DETECCIÓN SIN SUPRESIÓN NO MÁXIMA
        fast.setNonmaxSuppression(False) 
        keypoints_nonms = fast.detect(gray_image, None) 
        
        img_keypoints_without_nonmax_bgr = input_image_bgr.copy()
        cv2.drawKeypoints(
            input_image_bgr, 
            keypoints_nonms, 
            img_keypoints_without_nonmax_bgr, 
            color=(0, 255, 0), # Verde en BGR
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        img_keypoints_without_nonmax_rgb = cv2.cvtColor(img_keypoints_without_nonmax_bgr, cv2.COLOR_BGR2RGB)


        # --- VISUALIZACIÓN EN STREAMLIT ---
        st.subheader("Con Supresión No Máxima (NMS)")
        st.metric("Puntos Clave Detectados", len(keypoints_nms))
        st.image(img_keypoints_with_nonmax_rgb, caption="FAST con NMS")

        st.subheader("Sin Supresión No Máxima (NMS)")
        st.metric("Puntos Clave Detectados", len(keypoints_nonms))
        st.image(img_keypoints_without_nonmax_rgb, caption="FAST sin NMS")

        st.info("La Supresión No Máxima (NMS) reduce los puntos clave agrupados a un solo punto más representativo, resultando en menos detecciones.")

    except Exception as e:
        st.error(f"Ocurrió un error durante el procesamiento de la imagen: {e}")


def capitulo6():
    def compute_energy_matrix(img): 
        """Calcula la matriz de energía (gradiente de Sobel) de la imagen."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) 
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) 
        abs_sobel_x = cv2.convertScaleAbs(sobel_x) 
        abs_sobel_y = cv2.convertScaleAbs(sobel_y) 
        return cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0) 
    
    def compute_energy_matrix_modified(img, rect_roi): 
        """Modifica la matriz de energía: establece el ROI a 0 para que los 'seams' pasen por ahí."""
        energy_matrix = compute_energy_matrix(img)
        x,y,w,h = rect_roi 
        # Asegurarse de que las coordenadas sean válidas y el ROI no exceda los límites
        if y < energy_matrix.shape[0] and x < energy_matrix.shape[1] and y+h <= energy_matrix.shape[0] and x+w <= energy_matrix.shape[1]:
            # Establecer energía a 0 en el ROI para forzar la eliminación
            energy_matrix[y:y+h, x:x+w] = 0 
        return energy_matrix 

    def find_vertical_seam(img, energy): 
        """Encuentra el 'seam' vertical (camino de menor energía) a eliminar."""
        rows, cols = img.shape[:2] 
        seam = np.zeros(rows) 
        dist_to = np.zeros(img.shape[:2]) + float('inf')
        dist_to[0,:] = np.zeros(cols) 
        edge_to = np.zeros(img.shape[:2]) 
        
        # Programación dinámica para calcular el camino
        for row in range(rows - 1): 
            for col in range(cols): 
                # Movimiento a la izquierda, centro, derecha (col-1, col, col+1)
                for dc in [-1, 0, 1]:
                    new_col = col + dc
                    if 0 <= new_col < cols:
                        cost = energy[row + 1, new_col]
                        if dist_to[row + 1, new_col] > dist_to[row, col] + cost:
                            dist_to[row + 1, new_col] = dist_to[row, col] + cost
                            edge_to[row + 1, new_col] = -dc # Almacenar la dirección

        # Rastrear el camino desde la última fila
        seam[rows-1] = np.argmin(dist_to[rows-1, :]) 
        for i in range(rows - 1, 0, -1):
            seam[i-1] = seam[i] + edge_to[i, int(seam[i])] 
        
        return seam.astype(int)
    
    def remove_vertical_seam(img, seam): 
        """Remueve un 'seam' vertical de la imagen."""
        rows, cols = img.shape[:2] 
        # Crear una nueva imagen un píxel más estrecha
        img_removed = np.zeros((rows, cols - 1, 3), dtype=img.dtype)
        for row in range(rows): 
            # Copiar píxeles a la izquierda del seam
            img_removed[row, :seam[row]] = img[row, :seam[row]]
            # Copiar píxeles a la derecha del seam (shift left)
            img_removed[row, seam[row]:] = img[row, seam[row]+1:]
        
        return img_removed
    
    def add_vertical_seam(img, seam, num_iter): 
        """Añade un 'seam' vertical a la imagen (extender el fondo)."""
        seam = seam + num_iter 
        rows, cols = img.shape[:2] 
        # Crear una nueva imagen un píxel más ancha
        img_extended = np.zeros((rows, cols + 1, 3), dtype=img.dtype)
        
        for row in range(rows): 
            # Copiar píxeles a la izquierda del seam
            img_extended[row, :seam[row]] = img[row, :seam[row]]
            # Insertar el nuevo píxel (promedio de vecinos)
            v1 = img[row, int(seam[row]) - 1]
            v2 = img[row, int(seam[row]) % cols]
            img_extended[row, int(seam[row])] = (v1 + v2) / 2
            # Copiar píxeles a la derecha del seam (shift right)
            img_extended[row, int(seam[row]) + 1:] = img[row, int(seam[row]):]
            
        return img_extended
    
    def remove_object(img_orig, rect_roi): 
        """Función principal que ejecuta el Seam Carving para remover el objeto."""
        img = img_orig.copy()
        x, y, w, h = rect_roi
        
        if w <= 0 or h <= 0:
            return img_orig, "Rectángulo inválido o demasiado pequeño."
            
        # Número de seams a remover es el ancho del ROI más un margen
        num_seams = min(30, w)
        
        # 1. REMOVER SEAMS (Encogimiento de la imagen)
        for i in range(num_seams): 
            energy = compute_energy_matrix_modified(img, (x, y, w - i, h)) 
            seam = find_vertical_seam(img, energy) 
            img = remove_vertical_seam(img, seam) 
        
        img_output = np.copy(img) 

        # 2. AÑADIR SEAMS (Extender el fondo)
        img_for_adding = np.copy(img)
        
        for i in range(num_seams): 
            energy = compute_energy_matrix(img_for_adding) 
            seam = find_vertical_seam(img_for_adding, energy) 
            img_output = add_vertical_seam(img_output, seam, i) 
            img_for_adding = remove_vertical_seam(img_for_adding, seam) # Se remueve el seam más fácil para la próxima iteración
        
        return img_output, None


    st.markdown(
        """
        <div class="chapter-box">
            <div class="chapter-title">Capítulo 6 - Eliminador de Objetos</div>
            Usando una técnica o metodo conocida como Seam Carving. Asi como existen telas para la ropa, una imagen puede comportarse como tela y eliminar ciertos pixeles que no queremos.<br>
            Evidentemente este proceso toma un poco de tiempo pero cumple su objetivo. Aqui puede subir una imagen o usar la que está por defecto, le saldrá un rectangulo y usted debe marcar que parte de la imagen quiere eliminar.
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Qué es la Eliminador de Objetos"):
        st.markdown(
            """
            La eliminación de objetos, también conocida como Inpainting o Retoque de Contenido (Content-Aware Fill), tiene como objetivo principal borrar una región específica de una imagen y rellenar ese hueco con texturas y colores que se mezclen perfectamente con el resto del fondo.
            <br>
            <br>
            <h4 style='color: #8C9EFF;'>Detección vs. Eliminación</h4>
            <ul>
                <li>
                    <strong>Segmentación (ROI)</strong>: Primero, se debe identificar y seleccionar la Región de Interés (ROI), que es el área que ocupa el objeto a eliminar. En tu código, esto lo definen los cuatro sliders.
                </li>
                <li>
                    <strong>Eliminación (Inpainting)</strong>: Luego, el algoritmo debe generar píxeles nuevos y creíbles para rellenar ese hueco.
                </li>
            </ul>
            <br>
            <h4 style='color: #8C9EFF;'>Seam Carving (Tallado de Costuras)</h4>
            El Seam Carving, inventado por Shai Avidan y Ariel Shamir, es la técnica que has utilizado para eliminar el objeto, y funciona de manera diferente a las simples herramientas de borrado.
            <ul>
                <li>
                    <strong>Realmente hace</strong>: No borra el objeto, sino que reestructura la imagen quitando y luego añadiendo caminos de píxeles que atraviesan las áreas de menor importancia visual (los seams o costuras).
                </li>
                <li>
                    <strong>El resultado</strong>: una imagen donde el objeto ha desaparecido y el fondo circundante se ha cosido de forma fluida.
                </li>
            </ul>
            """,
            unsafe_allow_html=True
        )
    
    # --- Opción de fuente de imagen y Carga ---
    opcion = st.radio("Selecciona una opción:", ["📂 Subir imagen", "📄 Imagen por defecto"])

    img = None
    if opcion == "📂 Subir imagen":
        uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            try:
                pil_img = Image.open(uploaded_file).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                st.error(f"Error al leer la imagen: {e}")
                return
    else:
        try:
            img = cv2.imread("beach.jpg") 
            if img is None: 
                raise FileNotFoundError 
        except FileNotFoundError:
            st.error("No se encontró la imagen por defecto 'beach.jpg'.")
            return

    if img is None:
        st.warning("Por favor, sube o selecciona una imagen válida para continuar.")
        return

    img_orig = img.copy()
    rows, cols, _ = img.shape

    st.markdown("---")
    st.markdown("### 🎯 Definición de la Región de Interés (ROI)")
    
    max_x = cols - 1
    max_y = rows - 1
    
    # Valores por defecto para el ROI
    default_x_min = int(cols * 0.3)
    default_y_min = int(rows * 0.3)
    default_x_max = int(cols * 0.7)
    default_y_max = int(rows * 0.7)
    
    col_x_sliders, col_y_sliders = st.columns(2)
    
    with col_x_sliders:
        x_min = st.slider("🟥 Lado Izquierdo (Punto Rojo)", 0, max_x, default_x_min)
        y_min = st.slider("🟩 Lado Superior (Punto Verde)", 0, max_y, default_y_min)

    with col_y_sliders:
        x_max = st.slider("🟦 Lado Derecho (Punto Azul)", 0, max_x, default_x_max)
        y_max = st.slider("🟨 Lado Inferior (Punto Amarillo)", 0, max_y, default_y_max)
        

    tl_pt = (min(x_min, x_max), min(y_min, y_max))
    br_pt = (max(x_min, x_max), max(y_min, y_max))

    img_with_points = img_orig.copy() 
    
    radius = 8
    thickness = -1 

    cv2.circle(img_with_points, (x_min, y_min), radius, (0, 0, 255), thickness)
    cv2.circle(img_with_points, (x_max, y_min), radius, (255, 0, 0), thickness)
    cv2.circle(img_with_points, (x_min, y_max), radius, (0, 255, 0), thickness)
    cv2.circle(img_with_points, (x_max, y_max), radius, (0, 255, 255), thickness)
    
    # Dibujar el rectángulo ROI (borde blanco)
    cv2.rectangle(img_with_points, tl_pt, br_pt, (255, 255, 255), 2)
    
    st.image(cv2.cvtColor(img_with_points, cv2.COLOR_BGR2RGB), caption="Imagen con ROI (Región de Interés)")
    st.markdown("---")

    if st.button("🚀 Ejecutar Eliminación"):
        
        # 1. Normalizar coordenadas y calcular W y H
        x = min(x_min, x_max)
        y = min(y_min, y_max)
        w = abs(x_max - x_min)
        h = abs(y_max - y_min)
        
        rect_roi = (x, y, w, h) # Formato (x, y, w, h) necesario para remove_object

        with st.spinner('Procesando...'):
            scale_percent = 50  # Reducir al 50%
            new_width = int(img.shape[1] * scale_percent / 100)
            new_height = int(img.shape[0] * scale_percent / 100)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
            img_output, error = remove_object(img_orig.copy(), rect_roi)

        if error:
            st.error(f"Error al procesar: {error}")
            return

        # --- Mostrar resultado ---
        st.markdown("---")
        st.markdown("### 🔍 Resultados")
        
        st.image(cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB), caption="Objeto Eliminado y Fondo Recompuesto")
    

def capitulo7():
    def run_grabcut(img_orig, rect_coords):
        x_min, y_min, x_max, y_max = rect_coords
        
        # 1. Calcular el formato de rectángulo para GrabCut: (x, y, w, h)
        x = min(x_min, x_max)
        y = min(y_min, y_max)
        w = abs(x_max - x_min)
        h = abs(y_max - y_min)
        
        if w <= 1 or h <= 1:
            return img_orig.copy(), False # Rectángulo inválido

        rect_final = (x, y, w, h)
        
        # 2. Inicializar la máscara para GrabCut
        mask = np.zeros(img_orig.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        # 3. Ejecutar GrabCut
        cv2.grabCut(img_orig, mask, rect_final, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        
        # 4. Extraer la máscara final: El objeto es GC_FGD (1) o GC_PR_FGD (3)
        # Convertimos los píxeles de fondo (0 y 2) a negro (0) y el resto a blanco (1)
        mask2 = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')
        
        # 5. Aplicar la máscara (solo el foreground queda con color)
        img_output = img_orig * mask2[:, :, np.newaxis]
        return img_output, True
    
    st.markdown(
        """
        <div class="chapter-box">
            <div class="chapter-title">Capítulo 7 - Detectar formas y segmentación de imagen</div>
            <p>
                Sube una imagen o usa la una imagen por defecto.<br>
                Ajusta los 4 puntos de referencia (coordenadas) en la imagen con los sliders para definir un ROI. Luego ejecuta y sale el resultado. En este caso, usaremos la técnica GrabCut.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Qué es Segmentación"):
        st.markdown(
            """
            <h4 style='color: #8C9EFF;'>Segmentación de Imagen</h4>
            La segmentación es el proceso de dividir una imagen digital en múltiples segmentos (conjuntos de píxeles), haciendo que la representación de esa imagen sea más simple y significativa para el análisis por computadora.
            <ul>
                <li>
                    <strong>Propósito</strong>: Simplificar la imagen para que solo se quede con el objeto de interés, separándolo del fondo y de otros elementos.
                </li>
                <li>
                    <strong>Segmentación Semántica</strong>: Clasifica cada píxel en la imagen en una clase predefinida (ej. todo lo que es "cielo" es azul, todo lo que es "carretera" es gris).
                </li>
                <li>
                    <strong>Segmentación por Instancia</strong>: Identifica cada objeto individualmente. Si hay tres personas, las etiqueta como "Persona 1", "Persona 2" y "Persona 3".
                </li>
                <li>
                    <strong>Segmentación con GrabCut (USADA AQUÍ)</strong>: Es una técnica de segmentación interactiva. Se le da al algoritmo una pista (el ROI o rectángulo inicial) y él usa modelos de color y textura (GMMs) para refinar automáticamente los límites del objeto con gran precisión.
                </li>
            </ul>
            """,
            unsafe_allow_html=True
        )
    
    # --- Opción de fuente de imagen ---
    opcion = st.radio("Selecciona una opción:", ["📂 Subir imagen", "📄 Imagen por defecto"])

    img = None
    if opcion == "📂 Subir imagen":
        uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            try:
                # Leer imagen usando PIL y convertir a BGR (formato de OpenCV)
                pil_img = Image.open(uploaded_file).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                st.error(f"Error al leer la imagen: {e}")
                st.stop()
    else:
        # Intenta cargar la imagen por defecto
        img = cv2.imread("hand_pen.jpg")
        if img is None:
            st.error("No se encontró la imagen por defecto 'hand_pen.jpg'")
            st.stop()

    if img is None:
        st.warning("Por favor, sube o selecciona una imagen válida para continuar.")
        return

    img_orig = img.copy()
    rows, cols, _ = img.shape

    st.markdown("---")
    st.markdown("### 🎯 Definición de la Región de Interés (ROI)")
    
    max_x = cols - 1
    max_y = rows - 1
    
    # Valores por defecto para definir un ROI central
    default_x_min = int(cols * 0.25)
    default_y_min = int(rows * 0.25)
    default_x_max = int(cols * 0.75)
    default_y_max = int(rows * 0.75)
    
    # Sliders en 2 columnas para un diseño compacto
    col_x_sliders, col_y_sliders = st.columns(2)
    
    with col_x_sliders:
        x_min = st.slider("🟥 Lado Izquierdo (Punto Rojo)", 0, max_x, default_x_min)
        y_min = st.slider("🟩 Lado Superior (Punto Verde)", 0, max_y, default_y_min)

    with col_y_sliders:
        x_max = st.slider("🟦 Lado Derecho (Punto Azul)", 0, max_x, default_x_max)
        y_max = st.slider("🟨 Lado Inferior (Punto Amarillo)", 0, max_y, default_y_max)

    img_with_points = img.copy() 

    radius = 8
    thickness = -1 # Relleno

    # P1 (Rojo)
    cv2.circle(img_with_points, (x_min, y_min), radius, (0, 0, 255), thickness) 
    
    # P2 (Azul)
    cv2.circle(img_with_points, (x_max, y_min), radius, (255, 0, 0), thickness) 

    # P3 (Verde)
    cv2.circle(img_with_points, (x_min, y_max), radius, (0, 255, 0), thickness) 
    
    # P4 (Amarillo)
    cv2.circle(img_with_points, (x_max, y_max), radius, (0, 255, 255), thickness)
    
    # Dibujar el rectángulo ROI definido por los 4 puntos
    cv2.rectangle(img_with_points, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
    
    # Mostrar la imagen con los puntos antes del botón
    st.image(cv2.cvtColor(img_with_points, cv2.COLOR_BGR2RGB), caption="Imagen con Puntos de Referencia y ROI")

    if st.button("🚀 Ejecutar Segmentación"):
        # Coordenadas rect_coords para GrabCut (x_min, y_min, x_max, y_max)
        rect_coords = (x_min, y_min, x_max, y_max)

        with st.spinner('Procesando...'):
            img_segmented, success = run_grabcut(img_orig.copy(), rect_coords)

        if not success:
            st.error("El rectángulo de selección es demasiado pequeño. Por favor, amplía el ROI.")
            return
        
        st.subheader("🔍 Resultados")
        st.image(cv2.cvtColor(img_segmented, cv2.COLOR_BGR2RGB), caption="Imagen Segmentada")
        

def capitulo8():
    if 'run_camera_8' not in st.session_state:
        st.session_state.run_camera_8 = True

    def frame_diff(prev_frame, cur_frame, next_frame): 
        diff_frames1 = cv2.absdiff(next_frame, cur_frame) 
        diff_frames2 = cv2.absdiff(cur_frame, prev_frame) 
        return cv2.bitwise_and(diff_frames1, diff_frames2) 

    def get_frame(cap, scaling_factor):
        ret, frame = cap.read() 
        if not ret:
            return None, False
            
        frame = cv2.resize(frame, None, fx=scaling_factor, 
                            fy=scaling_factor, interpolation=cv2.INTER_AREA) 
        
        return frame, True


    st.markdown(
        """
        <div class="chapter-box">
            <div class="chapter-title">Capítulo 8 - Seguimiento de objetos</div>
            <p>
                También llamado (Object Tracking), es un proceso crucial en la visión por computadora que se centra en localizar la posición de un objeto de interés en una secuencia de video a lo largo del tiempo, manteniendo su identidad a medida que se mueve o cambia.<br>
                Aquí estamos aplicando la detección de movimiento con un filtro de grises.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Objetivo Principal"):
        st.markdown(
            """
            <p>Mantener la identidad del objeto a través de múltiples fotogramas (frames) de video.</p>
            <ul>
                <li>
                    <strong>Propósito</strong>: Simplificar la imagen para que solo se quede con el objeto de interés, separándolo del fondo y de otros elementos.
                </li>
            </ul>
            """,
            unsafe_allow_html=True
        )
    
    with st.expander("Diferencia clave con Detección"):
        st.markdown(
            """
            <ul>
                <li>
                    <strong>Detección</strong>: "¿Dónde está este objeto ahora?" (Procesa cada fotograma de forma independiente).
                </li>
                <li>
                    <strong>Seguimiento</strong>: Responde a la pregunta: "¿Qué objeto en el fotograma anterior corresponde a este objeto en este fotograma?" (Conecta la posición a lo largo del tiempo).
                </li>
            </ul>
            """,
            unsafe_allow_html=True
        )
    
    # --- Opción de fuente de imagen ---
    opcion = st.radio("Selecciona una opción:", ["📷 Activar Cámara", "📹 Subir Video"])

    uploaded_file = None
    
    if opcion == "📷 Activar Cámara":
        scaling_factor = st.slider("📏 Factor de Escala de Imagen", 0.2, 1.0, 0.5, 0.1)
    
    else:
        uploaded_file = st.file_uploader("Sube un archivo de video (mp4, avi, mov, mkv)", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
             col_scale, _ = st.columns([1, 1])
             with col_scale:
                 scaling_factor = st.slider("📏 Factor de Escala de Imagen", 0.2, 1.0, 0.5, 0.1)
        else:
            return

    st.markdown("---")
    
    is_ready = (opcion == "📷 Activar Cámara") or (opcion == "📹 Subir Video" and uploaded_file is not None)

    if is_ready:
        temp_file_path = None
        cap = None
        FRAME_WINDOW_CUR = None
        FRAME_WINDOW_DIFF = None

        source = 0
        if opcion == "📹 Subir Video" and uploaded_file is not None:
            try:
                # Guardar archivo temporalmente
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                tfile.close()
                temp_file_path = tfile.name
                source = temp_file_path
            except Exception as e:
                st.error(f"Error al guardar archivo temporal: {e}")
                return
        
        # 2. Inicializar la captura y procesamiento
        try:
            img_file = st.camera_input("Toma una foto")
            if img_file is not None:
                cap = Image.open(img_file)
                st.image(cap, caption="Imagen capturada", use_container_width=True)
            
            if not cap.isOpened():
                st.error(f"No se pudo acceder a la fuente de video ({opcion}). Verifica permisos o el archivo.")
                return
            
            st.markdown("### Fuente Actual (Grises)")
            FRAME_WINDOW_CUR = st.empty()
            st.markdown("---")
            st.markdown("### Resultado de Detección de Movimiento")
            FRAME_WINDOW_DIFF = st.empty()
                
            prev_frame, cur_frame, next_frame = None, None, None
            
            # Bucle de procesamiento de fotogramas
            while cap.isOpened():
                frame_bgr, ret = get_frame(cap, scaling_factor)
                
                if not ret:
                    break
                    
                prev_frame = cur_frame 
                cur_frame = next_frame
                next_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None and cur_frame is not None:
                    diff_img = frame_diff(prev_frame, cur_frame, next_frame)
                    _, diff_img_thresh = cv2.threshold(diff_img, 25, 255, cv2.THRESH_BINARY)
                    
                    cur_frame_rgb = cv2.cvtColor(cur_frame, cv2.COLOR_GRAY2RGB)
                    diff_img_rgb = cv2.cvtColor(diff_img_thresh, cv2.COLOR_GRAY2RGB)

                    FRAME_WINDOW_CUR.image(cur_frame_rgb, channels="RGB")
                    FRAME_WINDOW_DIFF.image(diff_img_rgb, channels="RGB")

        except Exception as e:
            st.error(f"Ocurrió un error durante el procesamiento de video: {e}")
            
        finally:
            if cap is not None:
                cap.release()
            
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    else:
        st.info("Selecciona la fuente y, si es necesario, sube el archivo para iniciar la detección de movimiento.")


def capitulo9():
    class DenseDetector(): 
        def __init__(self, step_size=20, feature_scale=20, img_bound=20): 
            # Detector de característica densa (manual)
            self.initXyStep = step_size
            self.initFeatureScale = feature_scale
            self.initImgBound = img_bound

        def detect(self, img):
            keypoints = []
            rows, cols = img.shape[:2]
            # Corrección: OpenCV espera (col, row, size) o (y, x, size).
            # Aquí x -> fila (rows), y -> columna (cols).
            for x in range(self.initImgBound, rows, self.initFeatureScale): # x es la fila
                for y in range(self.initImgBound, cols, self.initFeatureScale): # y es la columna
                    # cv2.KeyPoint usa (coordenada_x/columna, coordenada_y/fila, tamaño)
                    keypoints.append(cv2.KeyPoint(float(y), float(x), self.initXyStep))
            return keypoints 

    class SIFTDetector:
        def __init__(self):
            # Usa SIFT, no SURF
            self.detector = cv2.SIFT_create()
        
        def detect(self, image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints = self.detector.detect(gray, None)
            return keypoints

    st.markdown(
        """
        <div class="chapter-box">
            <div class="chapter-title">Capítulo 9 - Reconocimiento de objetos</div>
            <p>
                El reconocimiento de objetos se basa en encontrar puntos clave (keypoints) y descriptores en una imagen para que el sistema pueda identificar el objeto sin importar su tamaño, rotación o posición.<br>
                En este caso usamos el detector Dense (Denso) y SIFT  (Scale-Invariant Feature Transform).
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Detector Denso"):
        st.markdown(
            """
            <ul>
                <li>
                    <strong>Estrategia de Muestreo:</strong> Coloca puntos clave en una rejilla regular a intervalos fijos. No depende de la estructura, color o gradientes de la imagen.
                </li>
                <li>
                    <strong>Robustez:</strong> Baja. No maneja bien la rotación o el cambio de escala del objeto, ya que los puntos clave se mantienen fijos.
                </li>
                <li>
                    <strong>Velocidad:</strong> Muy rápida de computar. Útil para obtener descriptores en áreas grandes o como base para *clustering*.
                </li>
            </ul>
            """,
            unsafe_allow_html=True
        )

    with st.expander("Detector SIFT"):
        st.markdown(
            """
            <ul>
                <li>
                    <strong>Estrategia de Detección:</strong> Busca activamente puntos de interés únicos y estables (esquinas, picos) usando la Diferencia de Gaussianas (DoG).
                </li>
                <li>
                    <strong>Robustez:</strong> Alta. Es invariante a la escala y la rotación, lo que permite reconocer el objeto incluso si se transforma.
                </li>
                <li>
                    <strong>Reconocimiento:</strong> Proporciona descriptores fiables de 128 dimensiones por cada punto, esenciales para tareas de correspondencia y reconocimiento de objetos.
                </li>
            </ul>
            """,
            unsafe_allow_html=True
        )

    
    # --- Opción de fuente de imagen ---
    opcion = st.radio("Selecciona una opción:", ["📂 Subir imagen", "📄 Imagen por defecto"])

    input_image = None

    if opcion == "📂 Subir imagen":
        uploaded_file = st.file_uploader("Sube una imagen (JPEG, PNG) para detectar características", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            try:
                pil_img = Image.open(uploaded_file).convert("RGB")
                input_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                st.error(f"Error al leer la imagen: {e}")
                st.stop()
    else:
        input_image = cv2.imread("barco.jpg")
        if input_image is None:
            st.error("No se encontró la imagen por defecto 'barco.jpg'.")
            st.stop()

    if input_image is None:
        st.warning("Por favor, sube o selecciona una imagen válida para continuar.")
        return

    st.markdown("---")
    
    st.subheader("Ajustes del Detector Denso")
    col_step, col_scale, col_bound = st.columns(3)
    
    with col_step:
        step_size = st.slider("Step Size (Tamaño de Kp)", 10, 50, 20)
    with col_scale:
        feature_scale = st.slider("Feature Scale (Espaciado)", 10, 50, 20)
    with col_bound:
        img_bound = st.slider("Image Bound (Margen)", 0, 50, 5)

    # Preparamos copias para dibujar
    input_image_dense = np.copy(input_image)
    input_image_sift = np.copy(input_image)

    st.subheader("1. Detector Denso (Uniforme)")
    
    keypoints_dense = DenseDetector(step_size, feature_scale, img_bound).detect(input_image)
    
    input_image_dense = cv2.drawKeypoints(
        input_image_dense, 
        keypoints_dense, 
        None, 
        color=(0, 255, 0), # Verde
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    ) 
    
    st.image(cv2.cvtColor(input_image_dense, cv2.COLOR_BGR2RGB), caption='Imagen Uniforme')

    st.subheader("2. Detector SIFT (Robusto)")
    try:
        keypoints_sift = SIFTDetector().detect(input_image)
        
        input_image_sift = cv2.drawKeypoints(
            input_image_sift, 
            keypoints_sift, 
            None, 
            color=(255, 0, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        ) 
        
        st.image(cv2.cvtColor(input_image_sift, cv2.COLOR_BGR2RGB), caption='Imagen Robusta')

    except cv2.error:
        st.error("**Error de OpenCV (SIFT):**")
        st.markdown("El algoritmo SIFT falló. Esto usualmente requiere instalar **`opencv-contrib-python`**.")


def capitulo10():
    # --- ARCHIVO OBJ DE REFERENCIA ---
    OBJ_FILE = 'modelo3D/pioche.obj'

    # --- IMPORTACIÓN DE CLASES DE TRACKING (ASUMIDO) ---
    try:
        from pose_estimation import PoseEstimator
    except ImportError:
        class PoseEstimator:
            def __init__(self): pass
            def add_target(self, frame, rect): pass
            def track_target(self, frame): return []
            def clear_targets(self): pass
        st.error("Falta el archivo 'pose_estimation.py'. El tracking no funcionará.")

    # =========================================================
    # --- FUNCIONES DE SOPORTE (Lectura y Proyección 3D) ---
    # (Se omiten por ser idénticas a la versión anterior)
    # =========================================================

    def read_obj_geometry(filename):
        """Lector de OBJ con corrección de orientación Y/Z e inversión Y."""
        vertices = []; edges = []
        try:
            with open(filename, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if not parts: continue
                    if parts[0] == 'v': vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif parts[0] == 'f':
                        indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                        for i in range(len(indices)):
                            j = (i + 1) % len(indices)
                            edge = tuple(sorted((indices[i], indices[j])))
                            if edge not in edges: edges.append(edge)
        except FileNotFoundError:
            st.error(f"Error: El archivo {filename} no fue encontrado.")
            return np.array([[0,0,0]]), [], 0.3
        
        if not vertices: return np.array([[0,0,0]]), [], 0.3
        verts_np = np.array(vertices, dtype=np.float32)
        
        # Normalización y Corrección
        min_coords = verts_np.min(axis=0); max_coords = verts_np.max(axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1.0
        normalized_verts = (verts_np - min_coords) / range_coords
        
        swapped_verts = normalized_verts.copy()
        swapped_verts[:, 1] = normalized_verts[:, 2] # Nuevo Y (Altura) = Viejo Z
        swapped_verts[:, 2] = normalized_verts[:, 1] # Nuevo Z (Profundidad) = Viejo Y
        swapped_verts[:, 1] = 1.0 - swapped_verts[:, 1] # Invertir Y (Altura)
        
        scale_factor = 0.8
        offset_factor = (1.0 - scale_factor) / 2
        swapped_verts[:, 0] = swapped_verts[:, 0] * scale_factor + offset_factor
        swapped_verts[:, 1] = swapped_verts[:, 1] * scale_factor + offset_factor

        return swapped_verts, edges, 0.3 

    # =========================================================
    # --- CLASE TRACKER ADAPTADA A STREAMLIT ---
    # (Se mantiene idéntica a la versión anterior)
    # =========================================================

    class StreamlitTrackerOBJ(object): 
        """Maneja el estado y la lógica de tracking."""
        
        def __init__(self, scaling_factor, obj_model_file):
            self.scaling_factor = scaling_factor
            self.tracker = PoseEstimator() 
            self.overlay_vertices, self.overlay_edges, self.z_factor = read_obj_geometry(obj_model_file)
            self.color_lines = (255, 0, 0)
            self.color_base = (0, 255, 255)
            self.rect = None
            
        def add_target(self, frame, rect): 
            self.rect = rect
            self.tracker.add_target(frame, rect) 

        def overlay_obj_graphics(self, img, tracked):
            x_start, y_start, x_end, y_end = tracked.target.rect 
            quad_3d = np.float32([[x_start, y_start, 0], [x_end, y_start, 0], 
                                [x_end, y_end, 0], [x_start, y_end, 0]]) 
            h, w = img.shape[:2] 
            K = np.float64([[w * 0.8, 0, 0.5*(w-1)], [0, w * 0.8, 0.5*(h-1)], [0, 0, 1.0]]) 
            dist_coef = np.zeros(4) 
            
            ret, rvec, tvec = cv2.solvePnP(objectPoints=quad_3d, imagePoints=tracked.quad, cameraMatrix=K, distCoeffs=dist_coef)
            if not ret: return
            
            scale_x = x_end - x_start
            scale_y = y_end - y_start
            scale_z = -(x_end - x_start) * self.z_factor 
            
            verts = self.overlay_vertices * [scale_x, scale_y, scale_z]
            verts += (x_start, y_start, 0) 
            
            verts_2d = cv2.projectPoints(verts, rvec, tvec, cameraMatrix=K, distCoeffs=dist_coef)[0].reshape(-1, 2)
            verts_int = np.int32(verts_2d)
            
            for i, j in self.overlay_edges: 
                if i < len(verts_int) and j < len(verts_int):
                    (x_start_line, y_start_line), (x_end_line, y_end_line) = verts_int[i], verts_int[j]
                    cv2.line(img, (x_start_line, y_start_line), (x_end_line, y_end_line), self.color_lines, 2) 
                
            for (x, y) in verts_int:
                cv2.circle(img, (x, y), 3, self.color_base, -1)


    st.markdown(
        """
        <div class="chapter-box">
            <div class="chapter-title">Capítulo 10 - Realidad Aumentada</div>
            <p>
                En esta sección puede proyectar una imagen en RA (Realidad Aumentada) usando su cámara o una foto que desee subir. Para ambos casos, de preferencia ponga el modelo 3D sobre <b>una superficie plana </b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Definición de Realidad Aumentada"):
        st.markdown(
            """
            <p>La RA fusiona el mundo físico con el virtual, utilizando dispositivos (teléfonos, tabletas, gafas) como una "ventana" para ver el entorno real aumentado.</p>
            <ul>
                <li>
                    <strong>Diferencia clave con RV</strong>: La RA añade elementos al mundo real; la RV sustituye completamente el mundo real por uno simulado.
                </li>
            </ul>
            """,
            unsafe_allow_html=True
        )

    with st.expander("Componentes Esenciales de la RA"):
        st.markdown(
            """
            <p>Para lograr la superposición precisa del objeto 3D, el sistema debe comprender su entorno:</p>
            <ul>
                <li>
                    <strong>1. Detección y Tracking (Seguimiento)</strong>: Es la parte más crítica. El dispositivo debe identificar dónde colocar el objeto virtual.
                    <ul>
                        <li>Ubicación (Pose): Determinar la posición, rotación y escala de la cámara y del objeto de destino.</li>
                        <li>Métodos: Se utilizan **Marcadores** (imágenes específicas), <b>Detección de Superficies</b> o algoritmos avanzados como <b>SLAM</b>.</li>
                    </ul>
                </li>
                <li>
                    <strong>2. Procesamiento y Rendering</strong>: El sistema calcula la perspectiva y la iluminación para que el objeto virtual parezca real y luego lo dibuja (renderiza) junto con el video de la cámara.
                </li>
                <li>
                    <strong>3. Proyección (Display)</strong>: El resultado final se muestra en la pantalla del dispositivo (tu teléfono o gafas), donde el mundo real y el objeto virtual se ven unidos.
                </li>
            </ul>
            """,
            unsafe_allow_html=True
        )
    
    with st.expander("Aplicaciones de la RA"):
        st.markdown(
            """
            <p>La tecnología se usa en múltiples industrias, mejorando la interacción y la información:</p>
            <ul>
                <li>
                    <strong>Comercio Electrónico</strong>: Prueba virtual de productos (gafas, muebles).
                </li>
                <li>
                    <strong>Guías y Mantenimiento</strong>: Instrucciones paso a paso superpuestas sobre maquinaria compleja.
                </li>
                <li>
                    <strong>Entretenimiento</strong>: Juegos (Pokémon GO), filtros faciales.
                </li>
            </ul>
            """,
            unsafe_allow_html=True
        )


    if 'initialized' not in st.session_state or not st.session_state.initialized:
        st.session_state.tracker = StreamlitTrackerOBJ(0.8, OBJ_FILE)
        st.session_state.cap = None
        st.session_state.live_frame = None 
        st.session_state.first_frame = None
        st.session_state.state = "INIT"
        st.session_state.source_type = None
        st.session_state.initialized = True
    
    tracker = st.session_state.tracker

    # --- 1. SELECCIÓN DE FUENTE ---
    opcion = st.radio("Selecciona la fuente:", ["📷 Cámara en Vivo", "📂 Subir Archivo"])
    
    # --- BOTÓN DE REINICIO TOTAL (DEBAJO DE LAS FUENTES) ---
    if st.button("🔄 Reiniciar Todo", key='full_reset', type='secondary'):
        if st.session_state.cap: st.session_state.cap.release()
        st.session_state.initialized = False
        st.rerun()

    # Lógica de cambio de fuente
    current_source_type = "CAMERA" if opcion == "📷 Cámara en Vivo" else "IMAGE"
    if st.session_state.source_type != current_source_type:
        if st.session_state.cap: st.session_state.cap.release()
        st.session_state.tracker = StreamlitTrackerOBJ(0.8, OBJ_FILE)
        st.session_state.first_frame = None
        st.session_state.state = "INIT"
    st.session_state.source_type = current_source_type
    
    st.markdown("---")
    
    # --- POSICIONAMIENTO DE ELEMENTOS GLOBALES ---
    FRAME_WINDOW = st.empty()
    status_text = st.empty()
    
    # =========================================================
    # --- FLUJO PRINCIPAL BASADO EN EL ESTADO ---
    # =========================================================
    
    # === A. ESTADO: INICIO (INIT) / CÁMARA ACTIVA ===
    if st.session_state.state == "INIT" or st.session_state.state == "CAMERA_ACTIVE":
        
        if st.session_state.source_type == "CAMERA":
            # --- LÓGICA DE CÁMARA EN VIVO (Activación Automática) ---
            if st.session_state.state == "INIT":
                st.session_state.cap = cv2.VideoCapture(0)
                if st.session_state.cap and st.session_state.cap.isOpened():
                    st.session_state.state = "CAMERA_ACTIVE"
                else:
                    status_text.error("No se pudo acceder a la Webcam (capId=0).")
            
            if st.session_state.state == "CAMERA_ACTIVE":
                if st.button("📸 1. Tomar Foto (Congelar)", type="primary"):
                    if st.session_state.live_frame is not None:
                        # 1. Spinner de espera al tomar la foto
                        with st.spinner("Procesando... Capturando fotograma"):
                            st.session_state.first_frame = st.session_state.live_frame.copy()
                            st.session_state.state = "ROI_SELECTION"
                            if st.session_state.cap: st.session_state.cap.release()
                            st.session_state.cap = None 
                        st.rerun()
                    else:
                        status_text.warning("Espera a que el stream de la cámara se inicialice.")

                status_text.info("Cámara activa automáticamente. Presiona 'Tomar Foto' sobre el objeto plano.")
                
                cap = st.session_state.cap
                
                while st.session_state.state == "CAMERA_ACTIVE" and cap and cap.isOpened():
                    img_file = st.camera_input("Toma una foto")
                    
                    if img_file is not None:
                        cap = Image.open(img_file)
                        st.image(cap, caption="Imagen capturada", use_container_width=True)

                    ret, frame = cap.read()

                    if not ret: 
                        status_text.error("Error de cámara. Intente Reiniciar Todo.")
                        st.session_state.state = "INIT"
                        break
                    
                    frame = cv2.resize(frame, None, fx=tracker.scaling_factor, fy=tracker.scaling_factor, interpolation=cv2.INTER_AREA)
                    st.session_state.live_frame = frame 
                    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Cámara en Vivo")
                    sleep(0.01)
                
                if st.session_state.state == "CAMERA_ACTIVE":
                    st.session_state.state = "INIT"
                    st.session_state.cap = None
                    st.rerun()


        elif st.session_state.source_type == "IMAGE":
            # --- LÓGICA DE IMAGEN: SUBIDA ---
            uploaded_file = st.file_uploader("📂 Sube una imagen con un objeto plano.", type=["png", "jpg", "jpeg"])
                
            if uploaded_file is not None:
                # 2. Spinner de espera al subir el archivo
                with st.spinner("Procesando... Subiendo y escalando imagen"):
                    pil_img = Image.open(uploaded_file).convert("RGB")
                    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    
                    frame = cv2.resize(frame, None, fx=tracker.scaling_factor, fy=tracker.scaling_factor, interpolation=cv2.INTER_AREA)
                    st.session_state.first_frame = frame
                    st.session_state.state = "ROI_SELECTION"
                st.rerun()
            else:
                FRAME_WINDOW.markdown("")


    # === B. ESTADO: SELECCIÓN DE ROI (ROI_SELECTION) ===
    elif st.session_state.state == "ROI_SELECTION":
        frame = st.session_state.first_frame.copy()
        h, w = frame.shape[:2]

        status_text.info("Paso 2: Ajusta el ROI (rectángulo) sobre el objeto plano.")
        
        if st.button("⏪ Reiniciar Foto/Cambiar Fuente", key='reset_foto'):
            st.session_state.first_frame = None
            st.session_state.state = "INIT"
            st.rerun()
        
        # --- TÍTULO DE DEFINICIÓN DEL ROI ARRIBA (en la columna de ROI) ---
        st.subheader("🛠️ Definición del ROI (Manual)")
        
        # --- Sliders en 2 columnas para el diseño compacto ---
        col_x_sliders, col_y_sliders = st.columns(2)
        
        # --- LÍMITES y VALORES PREDETERMINADOS ---
        max_coord_x = w - 1
        max_coord_y = h - 1
        
        default_x_min = int(w * 0.3)
        default_y_min = int(h * 0.2)
        default_x_max = int(w * 0.7)
        default_y_max = int(h * 0.8)

        with col_x_sliders:
            x_min = st.slider("🟥 X Mínimo (Lado Izquierdo)", 0, max_coord_x, default_x_min, key='roi_x_min')
            y_min = st.slider("🟩 Y Mínimo (Lado Superior)", 0, max_coord_y, default_y_min, key='roi_y_min')

        with col_y_sliders:
            x_max = st.slider("🟦 X Máximo (Lado Derecho)", x_min + 1, max_coord_x, default_x_max, key='roi_x_max')
            y_max = st.slider("🟨 Y Máximo (Lado Inferior)", y_min + 1, max_coord_y, default_y_max, key='roi_y_max')
        
        # --- CONVERSIÓN DE (min, max) A (x, y, w, h) ---
        x = x_min
        y = y_min
        rect_w = x_max - x_min
        rect_h = y_max - y_min
        
        current_rect = (x, y, rect_w, rect_h)
        
        if st.button("🎯 3. CONFIRMAR ROI e INICIAR PROYECCIÓN", type="primary"):
            if rect_w > 10 and rect_h > 10:
                # 3. Spinner de espera al iniciar la proyección
                with st.spinner("Procesando... Calculando pose 3D"):
                    tracker.add_target(frame, current_rect)
                    st.session_state.state = "TRACKING_ACTIVE"
                st.rerun()
            else:
                status_text.error("El área del ROI debe ser mayor a 10x10 píxeles.")


        # Dibuja el rectángulo de selección (Visualización)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
        cv2.circle(frame, (x_min, y_min), 5, (0, 0, 255), -1)   # Rojo
        cv2.circle(frame, (x_min, y_max), 5, (0, 255, 255), -1) # Amarillo
        cv2.circle(frame, (x_max, y_max), 5, (255, 0, 0), -1)   # Azul
        cv2.circle(frame, (x_max, y_min), 5, (0, 255, 0), -1)   # Verde
        
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Fotograma Congelado: Ajusta el ROI")


    # === C. ESTADO: PROYECCIÓN ACTIVA (TRACKING_ACTIVE) ===
    elif st.session_state.state == "TRACKING_ACTIVE":
        
        frame = st.session_state.first_frame.copy() 
        img_display = frame.copy()
        
        tracked_list = tracker.tracker.track_target(frame) 
        
        if tracked_list:
            status_text.success("⛏️ Proyección 3D Activa. Resultado final listo.")
            for item in tracked_list: 
                cv2.polylines(img_display, [np.int32(item.quad)], True, (0, 255, 0), 2) 
                tracker.overlay_obj_graphics(img_display, item)
        else:
            status_text.error("⚠️ Error de Proyección. El ROI seleccionado puede ser inválido o muy pequeño.")
            
        FRAME_WINDOW.image(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB), caption="Realidad Aumentada Proyectada")


def capitulo11():
    try:
        import create_features as cf
    except ImportError:
        st.error("Error: No se encontró el módulo 'create_features.py'. Asegúrate de que está en la misma carpeta.")
        sys.exit() # Salir si el módulo de base no se puede importar

    # =========================================================
    # --- CONFIGURACIÓN DE ARCHIVOS ENTRENADOS ---
    # ! Ajusta estos nombres si son diferentes en tu proyecto.
    # =========================================================
    ANN_FILE = 'modelo_ann.xml'    # Archivo con la Red Neuronal (ANN)
    LE_FILE = 'label_encoder.pkl'      # Archivo con el LabelEncoder
    CODEBOOK_FILE = 'codebook.pkl' # Archivo con el Codebook (KMeans y Centroides)

    # =========================================================
    # --- CLASE CLASIFICADORA ADAPTADA A STREAMLIT ---
    # =========================================================

    class StreamlitImageClassifier(object): 
        """Carga los modelos entrenados y clasifica una imagen."""
        
        def __init__(self):
            # 1. Cargar la Red Neuronal Artificial (ANN)
            if not os.path.exists(ANN_FILE):
                st.error(f"Falta el archivo del modelo ANN: {ANN_FILE}. Ejecuta 'training.py' primero.")
                raise FileNotFoundError(ANN_FILE)
            self.ann = cv2.ml.ANN_MLP_load(ANN_FILE)

            # 2. Cargar el LabelEncoder
            if not os.path.exists(LE_FILE):
                st.error(f"Falta el archivo del Label Encoder: {LE_FILE}. Ejecuta 'training.py' primero.")
                raise FileNotFoundError(LE_FILE)
            with open(LE_FILE, 'rb') as f:
                self.le = pickle.load(f)

            # 3. Cargar el Codebook (KMeans y Centroides)
            if not os.path.exists(CODEBOOK_FILE):
                st.error(f"Falta el archivo del Codebook: {CODEBOOK_FILE}. Ejecuta 'create_features.py' primero.")
                raise FileNotFoundError(CODEBOOK_FILE)
            with open(CODEBOOK_FILE, 'rb') as f: 
                self.kmeans, self.centroids = pickle.load(f)

        def classify_tag(self, encoded_word):
            """Decodifica la salida numérica de la ANN a la etiqueta de texto."""
            # La función predict devuelve un array que necesitamos aplanar a un vector 1D
            prediction = np.asarray(encoded_word).ravel()
            
            # El LabelEncoder original usará inverse_transform
            # NOTA: La implementación de inverse_transform en el script original (classify_data.py)
            # sugiere que el LabelEncoder es una clase personalizada. Asumiremos el uso original.
            
            # Si la salida de predict es un vector one-hot:
            if prediction.ndim > 1:
                prediction = prediction.argmax() # Tomar el índice de la mayor probabilidad
            
            # Usamos el inverse_transform del LabelEncoder
            # Si 'self.le' es una clase LabelEncoder de scikit-learn o similar:
            # return self.le.inverse_transform([prediction])[0]
            
            # Basándonos en el código original que usa una función 'classify'
            # que a su vez usa inverse_transform, la lógica del LabelEncoder 
            # debe estar lista para manejar la salida de la ANN.
            
            # Usaremos la lógica de tu código original (asumiendo que self.le tiene un método inverse_transform):
            models = self.le.inverse_transform(prediction.reshape(1, -1)) # Usamos reshape para asegurar el formato (1, num_clases)
            return models[0]

        def get_image_tag(self, img): 
            # 1. Redimensionar la imagen
            img = cf.resize_to_size(img) 
            
            # 2. Extraer el vector de características BoVW
            feature_vector = cf.FeatureExtractor().get_feature_vector(img, self.kmeans, self.centroids) 
            
            # 3. Clasificar con la ANN
            _, image_tag_raw = self.ann.predict(feature_vector.astype(np.float32))
            
            # --- CÁLCULO DE PROBABILIDADES (SOFTMAX) ---
            # 1. Aplanar el array (ej: de [[-0.9, 0.8, -0.7]] a [-0.9, 0.8, -0.7])
            raw_scores_flattened = image_tag_raw.flatten()
            
            # 2. Aplicar Softmax para convertir las puntuaciones en probabilidades que suman 1
            probabilities = softmax(raw_scores_flattened)
            
            # 3. Decodificar la etiqueta final
            tag_final = self.classify_tag(image_tag_raw)

            # Retorna la etiqueta predicha y el array de probabilidades (porcentajes)
            return tag_final, probabilities


    st.markdown(
        """
        <div class="chapter-box">
            <div class="chapter-title">Capítulo 11 - Machine Learning por una Red Neuronal Artificial</div>
            <p>
                Es un subconjunto de la Inteligencia Artificial que se inspira en la estructura y funcionamiento del cerebro humano. Su objetivo es enseñar a una computadora a aprender patrones y tomar decisiones basándose en grandes cantidades de datos, en lugar de ser programada explícitamente para cada tarea.
                <br>
            </p>
            <div class="chapter-title"></div>
            <b>ACLARACIÓN: Esta red neuronal artificial fue entrenada con imágenes de animales como perros, gatos y loros. Cualquier otro tipo de imagen justificará un mal análisis.</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Definición"):
        st.markdown(
            """
            <p>El <strong>Machine Learning por una RNA</strong> es un modelo computacional inspirado en el cerebro que aprende patrones y toma decisiones basándose en grandes cantidades de datos.</p>
            <br>
            <ol>
                <li>
                    <strong>¿Qué es una Red Neuronal Artificial?</strong>
                    <p>Una RNA está compuesta por capas de neuronas artificiales interconectadas, cada una realizando un cálculo y aplicando una función de activación.</p>
                    <ul>
                        <li>
                            <strong>Capas de Entrada:</strong> Reciben los datos iniciales (características o <em>features</em>). Ej: Los Histogramas BoVW de una imagen.
                        </li>
                        <li>
                            <strong>Capas Ocultas:</strong> Realizan el procesamiento clave, aplicando una <strong>función de activación</strong> (como la Sigmoide Simétrica que usaste) para ponderar la información.
                        </li>
                        <li>
                            <strong>Capas de Salida:</strong> Proporcionan el resultado final, que en tu caso son las puntuaciones o <strong>probabilidades</strong> de que la imagen pertenezca a la clase "gato", "loro" o "perro".
                        </li>
                    </ul>
                </li>
                <br>
                <li>
                    <strong>El Proceso de Aprendizaje (Entrenamiento)</strong>
                    <p>La red aprende pasando los datos de entrenamiento repetidamente a través de tres fases:</p>
                    <ul>
                        <li>
                            <strong>Alimentación (Forward Propagation):</strong> Los datos pasan de la entrada a la salida para obtener una predicción.
                        </li>
                        <li>
                            <strong>Cálculo de Error:</strong> Se compara la predicción con la etiqueta real (verdadera) de la imagen.
                        </li>
                        <li>
                            <strong>Retropropagación (Backpropagation):</strong> Se ajustan los <strong>pesos</strong> de la red para minimizar ese error, permitiendo a la red "aprender" a clasificar mejor.
                        </li>
                    </ul>
                </li>
            </ol>
            """,
            unsafe_allow_html=True
        )

    try:
        classifier = StreamlitImageClassifier()
    except FileNotFoundError as e:
        return

    # --- 2. Interfaz de Subida de Archivo ---
    uploaded_file = st.file_uploader("📂 Sube tu imagen", type=["png","jpg","jpeg"])
    if uploaded_file is not None:
        try:
            # Intentamos abrir la imagen
            pil_img = Image.open(uploaded_file).convert("RGB")
            st.image(pil_img, caption="Imagen de Entrada")
            st.markdown("---")
            
            if st.button("✨ Clasificar Imagen"):
                with st.spinner("Procesando..."):
                    try:
                        # Convertir a NumPy BGR para OpenCV
                        input_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                        # Compatibilidad SIFT / ORB
                        try:
                            sift_test = cv2.SIFT_create
                        except AttributeError:
                            try:
                                sift_test = cv2.xfeatures2d.SIFT_create
                            except AttributeError:
                                def sift_test():
                                    return cv2.ORB_create()
                                st.warning("SIFT no disponible — se usará ORB en su lugar.")
                
                        # Inyectar compatibilidad
                        cv2.SIFT_create = sift_test
                        if not hasattr(cv2, "xfeatures2d"):
                            cv2.xfeatures2d = types.SimpleNamespace(SIFT_create=sift_test)
                        cv2.SURF_create = sift_test
                
                        # Clasificación
                        tag, probabilities = classifier.get_image_tag(input_img)
                        clases = classifier.le.classes_
                        prob_percent = (probabilities * 100).round(2)
                        puntuaciones = dict(zip(clases, prob_percent))
                
                        # Mostrar resultados
                        st.success("✅ **Clasificación Terminada**")
                        st.subheader(f"Clase Predicha: **{tag}**")
                        st.markdown("### Probabilidades de Clase")
                        st.dataframe({'Clase': clases, 'Probabilidad (%)': prob_percent}, hide_index=True)
                        st.bar_chart(puntuaciones)
                
                    except Exception as e:
                        st.error(f"Error durante el procesamiento o clasificación: {e}")
                        st.info("Verifica que los archivos de modelo sean compatibles.")

        except Exception:
            st.warning(
                "**No se pudo abrir la imagen.**\n\n"
                "Asegúrate de que sea un archivo de imagen válido (JPEG, PNG, BMP) y que no esté dañado.\n\n"
                "Intenta subir otra imagen o verificar que el archivo no esté corrupto."
            )


# --- Lógica Principal ---
if st.session_state.page in opciones:
    mostrarContenido(st.session_state.page)
    










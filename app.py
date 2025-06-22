
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Cargar el modelo previamente entrenado
model = load_model("traffic_sign_model.h5")

# Diccionario de nombres de clase (modificar según tu dataset real)
class_names = {
    0: "Stop",
    1: "Yield",
    2: "Speed Limit 50"
}

# Título de la app
st.title("🚦 Reconocimiento de Señales de Tránsito")

# Subida de archivo
uploaded_file = st.file_uploader("Sube una imagen de señal de tránsito", type=["jpg", "png"])

# Si se sube una imagen
if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((64, 64))
    st.image(image, caption="Imagen cargada", use_column_width=False)

    # Preprocesamiento: normalización y redimensionamiento
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 64, 64, 3)

    # Predicción
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    # Mostrar resultado
    st.markdown(f"### 🧠 Predicción: **{class_names.get(class_index, 'Desconocido')}**")
    st.markdown(f"Confianza: **{confidence * 100:.2f}%**")

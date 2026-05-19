import streamlit as st
import requests

st.set_page_config(page_title="Predicciones Airbnb", page_icon="🏡", layout="centered")

st.title("🏡 Validador y Predictor de Tarifas Airbnb")
st.write("Introduce los parámetros de tu propiedad de forma manual para estimar su precio óptimo por noche.")

st.markdown("---")

# Crear columnas visuales para organizar el formulario
col1, col2 = st.columns(2)

with col1:
    accommodates = st.slider("Capacidad (Personas)", 1, 16, 2)
    bedrooms = st.slider("Habitaciones", 1, 8, 1)
    bathrooms = st.slider("Baños", 0.5, 6.0, 1.0, step=0.5)
    beds = st.slider("Camas", 1, 16, 1)
    minimum_nights = st.number_input("Noches Mínimas", min_value=1, value=1)

with col2:
    property_type = st.selectbox("Tipo de Propiedad", ["apartment", "house", "condominium", "loft"])
    room_type = st.selectbox("Tipo de Habitación", ["entire home/apt", "private room", "shared room"])
    neighbourhood = st.text_input("Vecindario / Barrio", value="copacabana")
    review_rating = st.slider("Puntuación de Reseñas", 0, 100, 95)
    number_of_reviews = st.number_input("Número de Reseñas", min_value=0, value=10)

# Inputs binarios elegantes
st.subheader("Configuraciones adicionales")
c1, c2 = st.columns(2)
with c1:
    superhost = st.checkbox("¿El anfitrión es Superhost?")
with c2:
    instant_book = st.checkbox("¿Permite Reserva Inmediata?")

# Botón para ejecutar la acción
if st.button("🔥 Calcular Precio Estimado", use_container_width=True):
    # Estructurar el payload para tu API
    payload = {
        "accommodates": accommodates,
        "bathrooms": bathrooms,
        "bedrooms": bedrooms,
        "beds": beds,
        "guests_included": 1,
        "host_is_superhost": 1 if superhost else 0,
        "instant_bookable": 1 if instant_book else 0,
        "minimum_nights": minimum_nights,
        "neighbourhood_cleaned": neighbourhood,
        "number_of_reviews": number_of_reviews,
        "property_type": property_type,
        "review_scores_rating": review_rating,
        "room_type": room_type
    }
    
    # Consumir tu propia API local en segundo plano
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        if response.status_code == 200:
            resultado = response.json()
            # Asumiendo que tu API devuelve algo como {"prediction": X}
            st.success(f"### 💰 El precio estimado para este alojamiento es: **${resultado.get('prediction', 0.0):.2f} USD** / noche")
        else:
            st.error("Error en la comunicación con el servidor de predicción.")
    except Exception as e:
        st.error(f"Asegúrate de que la API de FastAPI esté corriendo en la terminal. Error: {e}")

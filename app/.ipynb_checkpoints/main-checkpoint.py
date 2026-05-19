import sys
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

# 1. Configuración estricta de rutas de Python
APP_PATH = os.path.abspath(os.path.dirname(__file__))
ROOT_PATH = os.path.abspath(os.path.join(APP_PATH, '..'))

if APP_PATH not in sys.path:
    sys.path.append(APP_PATH)
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

# 2. Importación segura de tus esquemas renovados
from schemas import PredictionInput, PredictionOutput

# 3. Inicialización de FastAPI con diseño mejorado (Markdown)
app = FastAPI(
    title="🏢 Airbnb Price Predictor API",
    description="""
    ## API de producción para estimar precios de alojamientos usando LightGBM.
    
    * **Uso:** Envía las características de un inmueble y obtén el precio estimado por noche.
    * **Modelo:** LightGBM Regressor optimizado mediante PyCaret.
    """,
    version="1.0.0"
)

# 4. Carga nativa del modelo usando PyCaret
MODEL_PATH = os.path.join(APP_PATH, "pipeline_ganador")
model = None

try:
    from pycaret.regression import load_model
    # PyCaret buscará automáticamente el archivo 'pipeline_ganador.pkl'
    model = load_model(MODEL_PATH)
    print("✅ ¡Modelo entrenado de PyCaret cargado con éxito!")
except Exception as e:
    print(f"❌ Error crítico al cargar el modelo con PyCaret: {e}")


@app.get("/", tags=["General"])
def home():
    return {"message": "API activa. Ve a /docs para interactuar."}


@app.post("/predict", response_model=PredictionOutput, tags=["Machine Learning"], summary="Predecir precio por noche")
def make_prediction(payload: PredictionInput):
    """
    Calcula la estimación del precio de un alojamiento basándose en sus características físicas y de reputación.
    
    - **Aplica re-escalado matemático (expm1)** de forma interna para revertir la escala logarítmica del entrenamiento.
    """
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="El modelo predictivo no está disponible en el servidor. Revisa los logs de la consola."
        )
    
    try:
        # Convertimos los datos que envía el cliente a DataFrame de Pandas
        input_data = pd.DataFrame([payload.model_dump()])
        
        # 1. El modelo realiza la predicción (PyCaret devuelve un DataFrame/Series con los resultados)
        prediction_result = model.predict(input_data)
        
        # 2. Extraemos el valor numérico logarítmico de forma segura según el formato de salida
        if hasattr(prediction_result, 'iloc'):
            prediction_log = float(prediction_result.iloc[0])
        elif isinstance(prediction_result, (list, np.ndarray)):
            prediction_log = float(prediction_result[0])
        else:
            prediction_log = float(prediction_result)
            
        # 3. 🔥 REVERTIMOS EL LOGARITMO EN CALIENTE EN LA API
        real_price = np.expm1(prediction_log)
        
        # Validación de seguridad matemática
        if np.isnan(real_price) or np.isinf(real_price):
            raise ValueError("La predicción matemática generó un valor inválido (NaN o Inf).")
            
        return PredictionOutput(estimated_price=round(real_price, 2))
        
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Error en el procesamiento de la predicción: {str(e)}"
        )

# src/processing/__init__.py
# src/processing/__init__.py
# src/processing/__init__.py
import pandas as pd
import numpy as np

def clean_raw_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Limpieza limpia y segura del target sin expresiones regulares.
    """
    df_cleaned = df.copy()
    
    if target_col not in df_cleaned.columns:
        raise KeyError(f"La columna target '{target_col}' no se encuentra en el DataFrame.")
    
    # Si la columna es de tipo texto/objeto, limpiamos de forma tradicional
    if df_cleaned[target_col].dtype == 'object':
        df_cleaned[target_col] = (
            df_cleaned[target_col]
            .astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.strip()
        )
    
    # Convertir a numérico flotante
    df_cleaned[target_col] = pd.to_numeric(df_cleaned[target_col], errors='coerce')
    
    # Eliminar filas vacías en el precio
    df_cleaned = df_cleaned.dropna(subset=[target_col])
    
    # Filtrar solo precios reales mayores a cero
    df_cleaned = df_cleaned[df_cleaned[target_col] > 0]
    
    return df_cleaned
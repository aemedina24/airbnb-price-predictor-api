# src/processing/

import pandas as pd
import numpy as np



def clean_raw_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:

    """

    Pipeline de limpieza inicial con Pandas.

    Elimina caracteres monetarios, nulos y filtra los 7 registros de $0.00.

    """

    df_cleaned = df.copy()

    

    if target_col not in df_cleaned.columns:

        raise KeyError(f"La columna target '{target_col}' no existe en el DataFrame.")

        

    # 1. Limpieza de texto (Quitar $, comas y espacios)

    if df_cleaned[target_col].dtype == 'object':

        df_cleaned[target_col] = (

            df_cleaned[target_col]

            .astype(str)

            .str.replace('$', '', regex=False)

            .str.replace(',', '', regex=False)

            .str.strip()

        )

    

    # 2. Conversión a número flotante

    df_cleaned[target_col] = pd.to_numeric(df_cleaned[target_col], errors='coerce')

    

    # 3. Dropna para filas sin precio

    df_cleaned = df_cleaned.dropna(subset=[target_col])

    

    # 4. FILTRO DE ANOMALÍAS: Elimina los $0.00 (que ahora son 0.0)

    df_cleaned = df_cleaned[df_cleaned[target_col] > 0]

    

    return df_cleaned
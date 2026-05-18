import pandas as pd
import re
import unicodedata
import pandas as pd

def clean_raw_data(df: pd.DataFrame, columna_target: str) -> pd.DataFrame:
    """
    1. Convierte la columna target a texto, elimina caracteres de moneda ($ y comas).
    2. Transforma la columna a tipo flotante de manera segura.
    3. Elimina automáticamente cualquier fila con precio menor o igual a cero (0 o negativos).
    """
    df_limpio = df.copy()
    
    if columna_target not in df_limpio.columns:
        raise KeyError(f"La columna '{columna_target}' no existe en el DataFrame.")
        
    # PASO 1: Limpieza de caracteres de texto de forma segura (quita $, comas y espacios vacíos)
    df_limpio[columna_target] = (
        df_limpio[columna_target]
        .astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    
    # PASO 2: Transformar a tipo numérico flotante. 
    # 'errors=coerce' convertirá cualquier texto corrupto ilegible en NaN en lugar de romper el código.
    df_limpio[columna_target] = pd.to_numeric(df_limpio[columna_target], errors='coerce')
    
    # PASO 3: Eliminar filas con NaNs en el target (si los hubiera tras la conversión)
    df_limpio = df_limpio.dropna(subset=[columna_target])
    
    # PASO 4: 🎯 EL FILTRO ABSOLUTO
    # Conservamos únicamente los registros con precios reales estrictamente mayores a cero (> 0)
    df_limpio = df_limpio[df_limpio[columna_target] > 0]
    
    return df_limpio


def transformar_variables_binarias(df: pd.DataFrame, columnas_binarias: list) -> pd.DataFrame:
    """
    Toma un DataFrame y una lista de columnas binarias, y convierte
    sus valores ('t'/'f', 'True'/'False', etc.) a enteros (1/0) directamente.
    """
    # 1. Hacer una copia para no modificar el DataFrame original por accidente
    df_transformado = df.copy()
    
    # DICCIONARIO DE MAPEO: Contempla todos los formatos posibles en los que venga el texto
    mapping = {
        't': 1, 'f': 0,
        'True': 1, 'False': 0,
        True: 1, False: 0,
        'true': 1, 'false': 0,
        '1': 1, '0': 0,
        1: 1, 0: 0
    }
    
    # 2. Filtrar solo las columnas que realmente existan en el DataFrame
    cols_a_procesar = [col for col in columnas_binarias if col in df_transformado.columns]
    
    # 3. Aplicar el mapeo columna por columna
    for col in cols_a_procesar:
        # Convertimos a texto y limpiamos espacios antes de mapear
        df_transformado[col] = (
            df_transformado[col]
            .astype(str)
            .str.strip()
            .map(mapping)
            .fillna(0)          # Si hay algún nulo o valor raro, lo pone en 0
            .astype(int)
        )
        
    return df_transformado


def purificar_e_imputar_numericas(df: pd.DataFrame, columnas_numericas: list, umbral_bandera: float = 0.20) -> tuple:
    """
    1. Fuerza la conversión de las columnas a tipo flotante.
    2. Convierte textos basura o caracteres inválidos en NaN.
    3. Si la columna supera el 'umbral_bandera' de nulos (ej: 20%), crea una columna indicadora binaria.
    4. Calcula la mediana de cada columna y reemplaza los nulos automáticamente.
    Retorna: (df_transformado, columnas_numericas_actualizadas)
    """
    # Hacer una copia profunda para evitar modificar el DataFrame original en memoria
    df_transformado = df.copy()
    
    # Crear una copia de la lista para no alterar la lista original por fuera
    nuevas_cols_numericas = columnas_numericas.copy()
    
    # Filtrar solo las columnas que realmente existan en el DataFrame
    cols_a_procesar = [col for col in columnas_numericas if col in df_transformado.columns]
    
    # Procesar e imputar columna por columna
    for col in cols_a_procesar:
        # Paso A: Forzar a número NaN
        df_transformado[col] = pd.to_numeric(df_transformado[col], errors='coerce').astype(float)
        
        # Calcular el porcentaje de nulos reales que tiene la columna antes de imputar
        porcentaje_nulos = df_transformado[col].isna().mean()
        
        # Paso B: Si supera el umbral (como tu 45%), creamos la variable indicadora de negocio
        if porcentaje_nulos > umbral_bandera:
            nombre_bandera = f"{col}_es_nulo"
            
            # 1 si era originalmente un nulo/basura, 0 si tenía calificación
            df_transformado[nombre_bandera] = df_transformado[col].isna().astype(int)
            
            # ¡Crucial! Añadimos esta nueva columna binaria a nuestra lista de features numéricas
            if nombre_bandera not in nuevas_cols_numericas:
                nuevas_cols_numericas.append(nombre_bandera)
        
        # Paso C: Calcular la mediana de la columna (ignora los NaN automáticamente)
        mediana_columna = df_transformado[col].median()
        
        # Paso D: Imputar (rellenar) los NaN con esa mediana
        df_transformado[col] = df_transformado[col].fillna(mediana_columna)
        
    return df_transformado, nuevas_cols_numericas


'''def purificar_e_imputar_numericas(df: pd.DataFrame, columnas_numericas: list) -> pd.DataFrame:
    """
    1. Fuerza la conversión de las columnas a tipo flotante.
    2. Convierte textos basura o caracteres inválidos en NaN.
    3. Calcula la mediana de cada columna y reemplaza los nulos automáticamente.
    """
    # 1. Hacer una copia profunda para evitar modificar el DataFrame original en memoria
    df_transformado = df.copy()
    
    # 2. Filtrar solo las columnas que realmente existan en el DataFrame
    cols_a_procesar = [col for col in columnas_numericas if col in df_transformado.columns]
    
    # 3. Procesar e imputar columna por columna
    for col in cols_a_procesar:
        # Paso A: Forzar a número (la basura se vuelve NaN)
        df_transformado[col] = pd.to_numeric(df_transformado[col], errors='coerce').astype(float)
        
        # Paso B: Calcular la mediana de la columna (ignora los NaN automáticamente)
        mediana_columna = df_transformado[col].median()
        
        # Paso C: Imputar (rellenar) los NaN con esa mediana
        df_transformado[col] = df_transformado[col].fillna(mediana_columna)
        
    return df_transformado'''


def limpiar_texto_base(texto):
    """Función interna de soporte para estandarizar las cadenas de texto."""
    if pd.isna(texto):
        return texto
    # 1. Minúsculas y quitar espacios en extremos
    texto = str(texto).lower().strip()
    # 2. Remover tildes y acentos
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # 3. Solo permitir letras, números y espacios estándar
    texto = re.sub(r'[^a-z0-9\s]', '', texto)
    # 4. Colapsar espacios múltiples en uno solo
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def limpiar_y_agrupar_texto(df: pd.DataFrame, columnas_texto: list, threshold: float = 0.006, other_label: str = 'otras') -> pd.DataFrame:
    """
    1. Estandariza y remueve caracteres especiales de las variables categóricas.
    2. Calcula la frecuencia relativa de cada categoría.
    3. Agrupa las categorías minoritarias (debajo del umbral) bajo la etiqueta 'otras'.
    """
    df_transformado = df.copy()
    
    # Filtrar solo columnas existentes
    cols_a_procesar = [col for col in columnas_texto if col in df_transformado.columns]
    
    for col in cols_a_procesar:
        # Paso 1: Limpieza y normalización de texto base
        df_transformado[col] = df_transformado[col].apply(limpiar_texto_base)
        
        # Paso 2: Calcular frecuencias relativas de lo que quedó limpio
        frecuencias = df_transformado[col].value_counts(normalize=True)
        
        # Paso 3: Identificar cuáles cumplen con el umbral mínimo (ej: > 0.6%)
        categorias_frecuentes = frecuencias[frecuencias >= threshold].index.tolist()
        
        # Paso 4: Mapear. Si no es frecuente (y no es nulo), se va a 'otras'
        df_transformado[col] = df_transformado[col].apply(
            lambda x: x if (pd.isna(x) or x in categorias_frecuentes) else other_label
        )
        
    return df_transformado
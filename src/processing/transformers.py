# =========================
# Standard library imports
# =========================
import re
import unicodedata

# =========================
# Third-party imports
# =========================
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MissingIndicatorTargeted(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.20):
        self.threshold = threshold
        self.columns_to_indicator_ = []
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        # Si PyCaret pasa un DataFrame, guardamos los nombres reales de las columnas
        if hasattr(X, 'columns'):
            self.feature_names_in_ = list(X.columns)
            X_df = pd.DataFrame(X)
        else:
            X_df = pd.DataFrame(X)
            if self.feature_names_in_ is None:
                self.feature_names_in_ = list(X_df.columns)
        
        # 1. Calculamos el porcentaje de nulos por columna
        missing_pct = X_df.isnull().mean()
        
        # 2. Guardamos solo las que superan el umbral (ej: 20%)
        self.columns_to_indicator_ = missing_pct[missing_pct > self.threshold].index.tolist()
        
        return self

    def transform(self, X):
        # Reconstruimos el DataFrame usando los nombres guardados en el fit
        if hasattr(X, 'columns'):
            X_df = pd.DataFrame(X).copy()
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        
        # 3. Creamos la variable binaria automáticamente
        for col in self.columns_to_indicator_:
            new_col_name = f"{col}_was_missing"
            X_df[new_col_name] = X_df[col].isnull().astype(int)
            
        return X_df
    

class CleanBinaryFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, binary_columns: list):
        self.binary_columns = binary_columns
        self.cols_a_procesar_ = []

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        # Identificamos y guardamos las columnas que realmente existen en el set de entrenamiento
        self.cols_a_procesar_ = [col for col in self.binary_columns if col in X_df.columns]
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        
        # Tu diccionario de mapeo optimizado para la conversión a string
        mapping = {
            't': 1, 'f': 0,
            'True': 1, 'False': 0,
            'true': 1, 'false': 0,
            '1': 1, '0': 0,
            '1.0': 1, '0.0': 0,
            'nan': 0  # Ataja los nulos que se convirtieron en la cadena "nan"
        }
        
        # Aplicamos tu lógica columna por columna
        for col in self.cols_a_procesar_:
            X_df[col] = (
                X_df[col]
                .astype(str)
                .str.strip()
                .map(mapping)
                .fillna(0)  # Por si cae un valor imprevisto que no esté en el mapeo
                .astype(int)
            )
            
        return X_df
    

class CleanAndGroupTextFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, text_columns: list, threshold: float = 0.006, other_label: str = 'otras'):
        self.text_columns = text_columns
        self.threshold = threshold
        self.other_label = other_label
        # Diccionario para memorizar las categorías válidas de cada columna
        self.valid_categories_ = {}
        self.cols_a_procesar_ = []

    @staticmethod
    def _limpiar_texto_base(texto):
        """Método estático interno para estandarizar las cadenas de texto."""
        if pd.isna(texto) or texto is None:
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

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.cols_a_procesar_ = [col for col in self.text_columns if col in X_df.columns]
        
        for col in self.cols_a_procesar_:
            # Paso 1: Limpieza temporal para calcular frecuencias reales
            texto_limpio = X_df[col].apply(self._limpiar_texto_base)
            
            # Paso 2: Calcular frecuencias relativas
            frecuencias = texto_limpio.value_counts(normalize=True)
            
            # Paso 3: Memorizar cuáles cumplen con el umbral mínimo (ej: > 0.6%)
            categorias_frecuentes = frecuencias[frecuencias >= self.threshold].index.tolist()
            self.valid_categories_[col] = categorias_frecuentes
            
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        
        for col in self.cols_a_procesar_:
            # Paso 1: Estandarizar el texto de la columna
            X_df[col] = X_df[col].apply(self._limpiar_texto_base)
            
            # Paso 4: Mapear usando las categorías memorizadas en el fit
            categorias_frecuentes = self.valid_categories_[col]
            X_df[col] = X_df[col].apply(
                lambda x: x if (pd.isna(x) or x in categorias_frecuentes) else self.other_label
            )
            
        return X_df

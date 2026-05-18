import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings


def generar_diccionario(Data):
    """Resumen de la estructura del conjunto de datos con porcentaje de nulos y moda."""
    print("\n-------TABLA CON TIPO DE VARIABLE, VALORES UNICOS, % DE NULOS Y MODA-------\n")
    dicc = []
    total_filas = len(Data)
    
    for col in Data.columns:
        # 1. Calculamos el porcentaje de nulos
        nulos = Data[col].isna().sum()
        porcentaje_nulos = (nulos / total_filas) * 100
        
        # 2. Calculamos la moda (el valor más frecuente)
        # .mode() devuelve una Serie (por si hay empates), tomamos el primer elemento [0]
        serie_moda = Data[col].mode()
        moda = serie_moda[0] if not serie_moda.empty else "N/A"
        
        dicc.append({
            "Variable": col,
            "Tipo pandas": Data[col].dtype,
            "Cantidad de valores únicos": Data[col].nunique(),
            "% Valores faltantes": round(porcentaje_nulos, 2),
            "Valor más frecuente (Moda)": moda # <--- Nueva columna
        })
        
    return pd.DataFrame(dicc)



def graficar_comparacion_target(df_limpio: pd.DataFrame, y_transformado: pd.Series, columna_target: str) -> None:
    """
    Genera dos histogramas lado a lado para comparar visualmente la distribución 
    de la variable objetivo en su escala original vs. su escala logarítmica, 
    calculando e imprimiendo el nivel de sesgo (skewness) en cada una.
    """
    # 1. Configurar el espacio para dos gráficas en una sola fila
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # --- Gráfica 1: Variable Target en su Escala Original ---
    sns.histplot(df_limpio[columna_target], bins=50, kde=True, ax=axes[0], color='crimson')
    axes[0].set_title(f"Distribución de '{columna_target}' (Sin Transformar)")
    axes[0].set_xlabel("Precio (Dólares)")
    axes[0].set_ylabel("Cantidad de Alojamientos")

    # Calculamos el sesgo (skewness) original
    sesgo_original = df_limpio[columna_target].skew()
    axes[0].text(0.6, 0.8, f"Sesgo: {sesgo_original:.2f}", transform=axes[0].transAxes, 
                 bbox=dict(facecolor='white', alpha=0.7))

    # --- Gráfica 2: Variable Target con Transformación Logarítmica ---
    sns.histplot(y_transformado, bins=50, kde=True, ax=axes[1], color='teal')
    axes[1].set_title("Distribución con Transformación Logarítmica (log1p)")
    axes[1].set_xlabel("Precio (Escala Logarítmica)")
    axes[1].set_ylabel("Cantidad de Alojamientos")

    # Calculamos el sesgo transformado
    sesgo_transformado = y_transformado.skew()
    axes[1].text(0.6, 0.8, f"Sesgo: {sesgo_transformado:.2f}", transform=axes[1].transAxes, 
                 bbox=dict(facecolor='white', alpha=0.7))

    # 2. Ajustar márgenes y mostrar las gráficas
    plt.tight_layout()
    plt.show()



def graficar_top10_numericas(df: pd.DataFrame, columns: list) -> None:
    """
    Genera una cuadrícula automatizada de gráficos de barras mostrando el Top 10 de valores
    más frecuentes para una lista de columnas numéricas, incluyendo conteos y porcentajes.
    """
    # 1. Configuración de la cuadrícula
    n_cols = 3
    n_rows = math.ceil(len(columns) / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))
    
    # Asegurar que axes sea un array plano incluso si es una sola fila o un solo gráfico
    if hasattr(axes, 'flatten'):
        axes = axes.flatten()
    else:
        axes = [axes]
        
    for i, col in enumerate(columns):
        # Filtrado: Quitamos NaNs y ceros
        data_series = df[col].dropna()
        data_series = data_series[data_series > 0]
        
        total_validos = len(data_series) # Para calcular el % real
        
        if total_validos > 0:
            # Obtenemos el Top 10
            top_10 = data_series.value_counts().head(10)
            
            # Graficamos
            sns.barplot(x=top_10.index.astype(str), 
                        y=top_10.values, 
                        ax=axes[i], 
                        hue=top_10.index.astype(str),
                        legend=False,
                        palette='viridis',
                        order=top_10.index.astype(str))
            
            axes[i].set_title(f'Top 10: {col}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Valor')
            axes[i].set_ylabel('Frecuencia')
            
            # Rotar los ejes x si los valores numéricos son muy largos
            axes[i].tick_params(axis='x', rotation=15)
            
            # 2. El truco del Porcentaje:
            for p in axes[i].patches:
                height = p.get_height()
                if height > 0: # Evitar anotar barras vacías
                    percentage = (height / total_validos) * 100
                    # Mostramos: "Conteo (Porcentaje%)"
                    axes[i].annotate(f'{int(height)}\n({percentage:.1f}%)', 
                                     (p.get_x() + p.get_width() / 2., height), 
                                     ha='center', va='center', 
                                     xytext=(0, 15), # Subimos un poco más el texto para que quepan dos líneas
                                     textcoords='offset points',
                                     fontsize=9,
                                     fontweight='bold')
        else:
            axes[i].set_title(f'{col} (Sin datos > 0)', fontsize=12, fontweight='bold')
            axes[i].axis('off') # Si no hay datos, apagamos el eje

    # 3. Limpieza de ejes sobrantes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def visualizar_binarias(df: pd.DataFrame, columnas_binarias: list) -> None:
    """
    Grafica la distribución de variables t/f sin transformarlas de forma automática.
    Muestra los conteos exactos y los porcentajes reales sobre cada barra.
    """
    # Filtrar solo las columnas que existen en el df para evitar errores
    cols_existentes = [c for c in columnas_binarias if c in df.columns]
    
    if not cols_existentes:
        print("No se encontraron las columnas especificadas en el DataFrame.")
        return

    # 1. Configuración de la cuadrícula usando math.ceil
    n_cols = 2
    n_rows = math.ceil(len(cols_existentes) / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 5))
    
    # Asegurar que axes sea iterable e indestructible si es 1 sola fila o gráfico
    if hasattr(axes, 'flatten'):
        axes = axes.flatten()
    else:
        axes = [axes]

    sns.set_style("whitegrid")
    
    # Paleta fija para consistencia visual en el informe (Verde para True, Rojo para False)
    colores_dict = {'t': '#2ecc71', 'f': '#e74c3c'}

    for i, col in enumerate(cols_existentes):
        # Limpiamos nulos para el cálculo
        datos_col = df[col].dropna().astype(str).str.strip()
        total_validos = len(datos_col)
        
        if total_validos > 0:
            # Obtenemos las frecuencias forzando el orden exacto ['t', 'f']
            conteos = datos_col.value_counts().reindex(['t', 'f'], fill_value=0)
            
            # Graficamos usando barplot para consistencia absoluta con la función de numéricas
            sns.barplot(x=conteos.index, 
                        y=conteos.values, 
                        ax=axes[i], 
                        hue=conteos.index,
                        legend=False,
                        palette=colores_dict,
                        order=['t', 'f'])
            
            # 2. El truco del Porcentaje y Conteo Centrado
            for p in axes[i].patches:
                height = p.get_height()
                if height > 0: # Solo anotar si la barra tiene registros
                    percentage = (height / total_validos) * 100
                    axes[i].annotate(f'{int(height)}\n({percentage:.1f}%)', 
                                     (p.get_x() + p.get_width() / 2., height), 
                                     ha='center', va='center', 
                                     xytext=(0, 15), # Espaciado superior para las dos líneas
                                     textcoords='offset points',
                                     fontsize=10,
                                     fontweight='bold')
        else:
            axes[i].text(0.5, 0.5, 'Sin datos t/f válidos', ha='center', va='center')
            
        # Estética de la gráfica
        nulos = df[col].isna().sum()
        axes[i].set_title(f'Variable: {col}\n(Nulos: {nulos})', fontsize=13, fontweight='bold')
        axes[i].set_xlabel('Valor Original', fontsize=11)
        axes[i].set_ylabel('Frecuencia', fontsize=11)
        axes[i].set_ylim(0, max(conteos.values) * 1.15) # Espacio extra arriba para que no se corten los textos

    # 3. Limpieza de ejes vacíos sobrantes si la lista es impar
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def visualizar_texto(df: pd.DataFrame, columnas_texto: list, top_n: int = 15) -> None:
    """
    Grafica la distribución de variables de texto usando barras horizontales.
    Muestra las 'top_n' categorías más frecuentes con conteos y porcentajes.
    """
    cols_existentes = [c for c in columnas_texto if c in df.columns]
    
    if not cols_existentes:
        print("No se encontraron las columnas de texto especificadas en el DataFrame.")
        return
        
    sns.set_style("whitegrid")
    
    for col in cols_existentes:
        # 1. Obtener la frecuencia y limitar al Top N
        conteo = df[col].dropna().astype(str).str.strip().value_counts().head(top_n)
        total = len(df[col].dropna())
        
        if len(conteo) == 0:
            print(f"La columna {col} no tiene datos válidos para graficar.")
            continue
            
        # Ajustar dinámicamente el tamaño de la figura según la cantidad de categorías reales
        altura_figura = max(4, len(conteo) * 0.4)
        plt.figure(figsize=(10, altura_figura))
        
        # 2. Crear gráfico horizontal con Seaborn
        ax = sns.barplot(
            y=conteo.index, 
            x=conteo.values, 
            palette='viridis',
            hue=conteo.index,
            legend=False,
            order=conteo.index
        )
        
        # 3. Añadir porcentajes al final de cada barra (Alineación corregida)
        for i, v in enumerate(conteo.values):
            percentage = f' {int(v)} ({100 * v / total:.1f}%)'
            # va='center' asegura que el texto quede centrado a la altura de la barra
            ax.text(v, i, percentage, va='center', ha='left', fontsize=10, fontweight='bold')
            
        # Estética e Información del Gráfico
        nulos = df[col].isna().sum()
        distintos = df[col].nunique()
        
        plt.title(f'Distribución de: {col}\n({distintos} categorías únicas, Nulos: {nulos})', 
                  fontsize=13, fontweight='bold', pad=15)
        plt.xlabel('Cantidad de Registros', fontsize=11)
        plt.ylabel('Categoría', fontsize=11)
        
        # CORRECCIÓN CRÍTICA: Multiplicamos por 1.25 para dejar un 25% de espacio libre 
        # a la derecha. Así los textos "Conteo (Porcentaje%)" nunca se cortarán.
        plt.xlim(0, conteo.max() * 1.25) 
        
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()
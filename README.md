#  Airbnb Price Predictor API

¡Bienvenido! Este proyecto consiste en el despliegue en producción de un modelo de Machine Learning capaz de estimar el precio por noche de alojamientos en Airbnb basándose en características físicas, ubicación y reputación del inmueble.

La solución abarca desde el análisis de datos y entrenamiento del modelo, hasta la creación de un microservicio robusto e interactivo listo para ser consumido por aplicaciones web o móviles.

---

## 🛠️ Tecnologías Utilizadas

* **Python 3.11**
* **PyCaret & LightGBM:** Para el pipeline automatizado de Machine Learning, selección de características, imputación de nulos y entrenamiento del regresor.
* **FastAPI & Uvicorn:** Para la creación del backend de la API REST de alto rendimiento.
* **Pydantic:** Para la validación estricta de los datos de entrada y salida (*schemas*).
* **Poetry:** Como gestor de dependencias y entornos virtuales del proyecto.

---

##  Arquitectura y Características de la API

* **Pipeline Integrado:** La API carga nativamente el pipeline entrenado (`.pkl`) garantizando que las transformaciones de datos (como la imputación de nulos o el encoding de variables categóricas) se apliquen de forma idéntica a los datos en producción.
* **Re-escalado Matemático Integrado:** El modelo fue entrenado utilizando la escala logarítmica de la variable objetivo para estabilizar la varianza. La API maneja de forma interna y en caliente la transformación inversa (`np.expm1`) para devolver el precio final en valores monetarios reales y limpios.
* **Documentación Interactiva:** Implementación nativa de Swagger UI para probar predicciones en tiempo real con datos de prueba estructurados.

---

## 💻 Cómo Ejecutar el Proyecto Localmente

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/aemedina24/airbnb-price-predictor-api](https://github.com/aemedina24/airbnb-price-predictor-api.git)
   cd Proyecto_3

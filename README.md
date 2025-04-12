# Modelo Supervisado - Predicción de Tiempos de Viaje SITP
 **Modelo Supervisado - Predicción de Tiempos de Viaje SITP**

 Aplicacion de técnicas de **aprendizaje supervisado** para predecir el **tiempo real de viaje** en el sistema de transporte masivo de Bogotá, el **SITP**, usando datos simulados de rutas, horarios y demanda.


**Fuentes de Datos**

Para entrenar el modelo, se utilizó un conjunto de datos sintético con la siguiente estructura:

```csv
origen,destino,ruta_directa,transbordos,distancia_km,hora_dia,dia_semana,tiempo_real_min,demanda
Portal Suba,Av Jimenez,0,1,8.5,8:00,3,42,alta
Museo Nacional,Calle 72,1,0,3.2,18:30,5,15,media
Portal Americas,Portal 80,0,2,12.1,7:45,1,38,alta
Calle 100,Av Jimenez,1,0,5.7,12:00,4,22,baja
```

**Variables del Dataset**

**Variables de entrada (features):**
- `origen` y `destino`: estaciones de inicio y fin del trayecto.
- `ruta_directa`: indica si hay una ruta directa disponible (1) o no (0).
- `transbordos`: número de transbordos necesarios.
- `distancia_km`: distancia aproximada del trayecto.
- `hora_dia`: hora estimada de inicio del viaje.
- `dia_semana`: número del día de la semana (1: lunes, 7: domingo).
- `demanda`: nivel de demanda del trayecto (`alta`, `media`, `baja`).

**Variable objetivo (target):**
- `tiempo_real_min`: duración real del viaje en minutos.


**Desarrollo del modelo python**

Dependencias

| Librería      | Uso en el proyecto                                         |
|---------------|------------------------------------------------------------|
| `pandas`      | Manipulación de datos: lectura, limpieza y transformación |
| `scikit-learn`| Entrenamiento y evaluación con `RandomForestRegressor`    |
| `matplotlib`  | Visualización de resultados (predicción vs. realidad)     |


**Modelo: Random Forest Regressor**

- **Tipo**: Aprendizaje supervisado (regresión)
- **Fundamento**: Aprende de ejemplos históricos con etiquetas (`tiempo_real_min`)
- **Relación con el problema**: Permite predecir tiempos reales en rutas nuevas
- **Ventajas**:
  - Robusto ante datos categóricos y numéricos
  - Buen rendimiento inicial incluso con pocos datos

**Características**

- Se aplicó **preprocesamiento supervisado**:
  - Conversión de variables categóricas (`origen`, `destino`, `demanda`) mediante **One-Hot Encoding**
  - Transformación de la hora a formato numérico si es necesario

- Se analizaron las **importancias de las características** (`feature_importances_`) del modelo:
  - Variables más influyentes: `distancia_km` y `hora_dia` (especialmente en horas pico)

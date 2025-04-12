import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# 1. Cargar datos
def cargar_datos():
    try:
        datos = pd.read_csv('data/rutas_sitp_ml.csv')
        print("Datos cargados exitosamente. Muestra:")
        print(datos.head())
        return datos
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return None

# 2. Preprocesamiento
def preprocesar(datos):
    # Convertir variables categóricas
    datos = pd.get_dummies(datos, columns=['ruta_directa', 'demanda', 'dia_semana'])
    
    # Convertir hora a minutos del día
    datos['hora_minutos'] = datos['hora_dia'].apply(
        lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]))
    
    # Seleccionar características y objetivo
    X = datos.drop(['tiempo_real_min', 'origen', 'destino', 'hora_dia'], axis=1)
    y = datos['tiempo_real_min']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entrenamiento y evaluación
def entrenar_modelo(X_train, X_test, y_train, y_test):
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    
    predicciones = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, predicciones)
    
    print(f"\nError Absoluto Medio: {mae:.2f} minutos")
    
    # Visualización
    plt.scatter(y_test, predicciones)
    plt.xlabel('Tiempo Real (min)')
    plt.ylabel('Tiempo Predicho (min)')
    plt.title('Predicción vs Real')
    plt.savefig('resultados/predicciones.png')
    plt.close()
    
    return modelo

# Ejecución principal
if __name__ == "__main__":
    print("=== Sistema de Predicción de Tiempos de Viaje SITP ===")
    
    # Crear carpetas necesarias
    import os
    os.makedirs('data', exist_ok=True)
    os.makedirs('resultados', exist_ok=True)
    
    datos = cargar_datos()
    if datos is not None:
        X_train, X_test, y_train, y_test = preprocesar(datos)
        modelo = entrenar_modelo(X_train, X_test, y_train, y_test)
        
        # Ejemplo de predicción
        print("\nEjemplo de predicción:")
        ejemplo = X_test.iloc[0:1]
        print(f"Datos de entrada:\n{ejemplo}")
        print(f"Tiempo predicho: {modelo.predict(ejemplo)[0]:.1f} minutos")
        print(f"Tiempo real: {y_test.iloc[0]} minutos")
        

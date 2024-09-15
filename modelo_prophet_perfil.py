"""
Universitat Carlemany - Bachelor en Data Science

Proyecto Final de Bachelor: 
"Optimización Económica del Almacenamiento de Energía en Sistemas Fotovoltaicos Mediante Algoritmos de Machine Learning"

Autor: Pablo Felipe Cano Galán
Descripción: Este código forma parte del trabajo de fin de grado (TFB), centrado en el desarrollo de modelos de machine learning para la predicción de momentos óptimos de carga y descarga de baterías en sistemas fotovoltaicos, con el fin de maximizar la eficiencia energética y la rentabilidad económica de dichos sistemas. El proyecto incluye la implementación de herramientas de simulación y una aplicación web para facilitar la gestión de sistemas fotovoltaicos.

Fecha: 07/09/2024
"""

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle

# Cargar el dataset
data = pd.read_csv("typical_year_profile.csv")
data['datetime'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(data['day'] - 1, unit='D') + pd.to_timedelta(data['hour'], unit='h')

data = data.set_index('datetime')

# Convertir el índice a UTC para evitar problemas con las zonas horarias
data.index = pd.to_datetime(data.index, utc=True)

# Verificar si hay NaNs en el conjunto de datos
print("Verificación de NaN en el conjunto de datos:")
print(data.isna().sum())

# Preprocesamiento para Prophet
df = data[['COEF. PERFIL A']].reset_index().rename(columns={'datetime': 'ds', 'COEF. PERFIL A': 'y'})

# Eliminar la zona horaria de la columna 'ds' (fechas)
df['ds'] = df['ds'].dt.tz_localize(None)

# División del conjunto de datos en entrenamiento y prueba
train_data = df[:int(0.8 * len(df))]
test_data = df[int(0.8 * len(df)):]

# Crear y entrenar el modelo Prophet
model = Prophet()
model.fit(train_data)

# Realizar predicciones sobre el conjunto de prueba
future = model.make_future_dataframe(periods=len(test_data), freq='H')  # Ajusta la frecuencia según tu conjunto de datos
forecast = model.predict(future)

# Extraer solo las predicciones relevantes (correspondientes al conjunto de prueba)
predictions = forecast[['ds', 'yhat']].set_index('ds').loc[test_data['ds']]

# Reemplazar predicciones negativas con cero
predictions['yhat'] = predictions['yhat'].apply(lambda x: max(0, x))

# Evaluación de las predicciones
test_data.set_index('ds', inplace=True)
test_data['Predictions'] = predictions['yhat']
rmse_value = np.sqrt(mean_squared_error(test_data['y'], test_data['Predictions']))
print(f'RMSE: {rmse_value}')

# Guardar el modelo en un archivo
with open('prophet_model_perfil.pkl', 'wb') as pkl_file:
    pickle.dump(model, pkl_file)

# Guardar las predicciones en un archivo CSV
test_data[['y', 'Predictions']].to_csv("predicciones_perfil_prophet.csv")

# Visualización de los resultados
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data['y'], label="Datos Reales")
plt.plot(test_data.index, test_data['Predictions'], label="Predicciones", color='red')
plt.xlabel('Date')
plt.ylabel('COEF. PERFIL A')
plt.title('Predicciones vs Datos Reales')
plt.legend()
plt.show()

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
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle

# Cargar el dataset
data = pd.read_csv("little_typical_year_spain_corrected.csv", parse_dates=['datetime'])
data = data.set_index('datetime')

# Verificar si hay NaNs en el conjunto de datos
print("Verificación de NaN en el conjunto de datos:")
print(data.isna().sum())

# División del conjunto de datos en entrenamiento y prueba
train_data = data[:int(0.8 * len(data))]
test_data = data[int(0.8 * len(data)):]

# Configuración del modelo SARIMA
arima_model = SARIMAX(train_data["G(i)"], order=(2,1,0), seasonal_order=(1,1,0,12)) 
arima_result = arima_model.fit()

# Guardar el modelo en un archivo
with open('arima_model.pkl', 'wb') as pkl_file:
    pickle.dump(arima_result, pkl_file)

# Realizar predicciones sobre el conjunto de prueba
start_index = test_data.index[0]
end_index = test_data.index[-1]
predictions = arima_result.predict(start=start_index, end=end_index, typ="levels").rename("Predictions")

# Evaluación de las predicciones
test_data['Predictions'] = predictions
rmse_value = np.sqrt(mean_squared_error(test_data["G(i)"], test_data["Predictions"]))
print(f'RMSE: {rmse_value}')

# Guardar las predicciones en un archivo CSV
test_data[['G(i)', 'Predictions']].to_csv("predicciones_arima.csv")

# Visualización de los resultados
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data["G(i)"], label="Datos Reales")
plt.plot(test_data.index, test_data["Predictions"], label="Predicciones", color='red')
plt.xlabel('Date')
plt.ylabel('G(i)')
plt.title('Predicciones vs Datos Reales')
plt.legend()
plt.show()

# Diagnósticos del modelo
arima_result.plot_diagnostics(figsize=(16, 8))
plt.show()

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
import joblib

# Cargar los datos
data = pd.read_csv('little_typical_year_spain_corrected.csv', parse_dates=['datetime'], index_col='datetime')

# Dividir en entrenamiento y prueba
n_periods = 868  # Número de periodos para el conjunto de prueba
train = data.iloc[:-n_periods]
test = data.iloc[-n_periods:]

# Definir el modelo ARIMA (sin variables exógenas)
model = SARIMAX(train['G(i)'], order=(2, 1, 0), seasonal_order=(1, 1, 0, 12))

# Ajustar el modelo
model_fit = model.fit(disp=False)

# Hacer predicciones
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1)

# Aplicar clipping para evitar valores negativos
predictions = np.maximum(predictions, 0)

# Guardar las predicciones en un DataFrame
predicciones_df = pd.DataFrame({'datetime': test.index, 'G(i)': test['G(i)'], 'Predictions': predictions})
predicciones_df.set_index('datetime', inplace=True)

# Guardar las predicciones en un archivo CSV
predicciones_df.to_csv('predicciones_arima.csv')

# Calcular el RMSE
rmse = np.sqrt(mean_squared_error(test['G(i)'], predictions))
print(f'RMSE: {rmse}')

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(test.index, test['G(i)'], label='Datos Reales')
plt.plot(test.index, predictions, label='Predicciones', color='red')
plt.title('Predicciones vs Datos Reales')
plt.xlabel('Fecha')
plt.ylabel('Irradiación G(i)')
plt.legend()
plt.show()

# Guardar el modelo ARIMA
joblib.dump(model_fit, 'sarimax_model_sin_exo.pkl')

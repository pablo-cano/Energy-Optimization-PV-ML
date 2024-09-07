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
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import pickle

# Función para evaluar predicciones con las métricas solicitadas
def evaluate_metrics(actual, predicted, model_name):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    print(f'Evaluación del modelo {model_name}:')
    print(f'MAE (Error Absoluto Medio): {mae}')
    print(f'MSE (Error Cuadrático Medio): {mse}')
    print(f'RMSE (Raíz del Error Cuadrático Medio): {rmse}')
    print(f'MAPE (Error Absoluto Porcentual Medio): {mape}%\n')

# Función para graficar los residuos
def plot_residuals(actual, predicted, model_name):
    residuals = actual - predicted
    plt.figure(figsize=(10, 6))
    plt.plot(residuals, label=f'Residuos {model_name}')
    plt.axhline(0, color='red', linestyle='--', label='Cero')
    plt.title(f'Gráfico de Residuos - {model_name}')
    plt.xlabel('Fecha')
    plt.ylabel('Residuos')
    plt.legend()
    plt.show()

# Cargar los datos
data = pd.read_csv('little_typical_year_spain_corrected.csv', parse_dates=['datetime'], index_col='datetime')

# Dividir en entrenamiento y prueba
n_periods = 868  # Número de periodos para el conjunto de prueba
train = data.iloc[:-n_periods]
test = data.iloc[-n_periods:]

# Cargar y evaluar el modelo ARIMAX
sarimax_model = joblib.load('sarimax_model_sin_exo.pkl')
test_exog = test[['T2m']]
sarimax_predictions = sarimax_model.predict(start=len(train), end=len(train)+len(test)-1, exog=test_exog)

# Aplicar clipping para evitar valores negativos
sarimax_predictions = np.maximum(sarimax_predictions, 0)

# Evaluar y mostrar resultados del modelo ARIMAX
evaluate_metrics(test['G(i)'], sarimax_predictions, 'ARIMAX')

# Graficar residuos del modelo ARIMAX
plot_residuals(test['G(i)'], sarimax_predictions, 'ARIMAX')

# Cargar y evaluar el modelo ARIMA
with open('arima_model.pkl', 'rb') as pkl_file:
    arima_model = pickle.load(pkl_file)
arima_predictions = arima_model.predict(start=test.index[0], end=test.index[-1])

# Aplicar clipping para evitar valores negativos
arima_predictions = np.maximum(arima_predictions, 0)

# Evaluar y mostrar resultados del modelo ARIMA
evaluate_metrics(test['G(i)'], arima_predictions, 'ARIMA')

# Graficar residuos del modelo ARIMA
plot_residuals(test['G(i)'], arima_predictions, 'ARIMA')

# Cargar y evaluar el modelo Prophet
with open('prophet_model.pkl', 'rb') as pkl_file:
    prophet_model = pickle.load(pkl_file)

# Generar el DataFrame futuro para las predicciones de Prophet
future = pd.DataFrame(test.index)
future.columns = ['ds']

# Realizar las predicciones
prophet_forecast = prophet_model.predict(future)

# Alinear las fechas de las predicciones de Prophet con las fechas de test
prophet_forecast.set_index('ds', inplace=True)
prophet_predictions = prophet_forecast.reindex(test.index).interpolate(method='time')['yhat']

# Aplicar clipping para evitar valores negativos
prophet_predictions = np.maximum(prophet_predictions, 0)

# Evaluar y mostrar resultados del modelo Prophet
evaluate_metrics(test['G(i)'], prophet_predictions, 'Prophet')

# Graficar residuos del modelo Prophet
plot_residuals(test['G(i)'], prophet_predictions, 'Prophet')

# Comparación de todos los modelos
plt.figure(figsize=(12, 8))
plt.plot(test.index, test['G(i)'], label='Datos Reales', color ='black')
plt.plot(test.index, sarimax_predictions, label='ARIMAX Predicciones', color='red')
plt.plot(test.index, arima_predictions, label='ARIMA Predicciones', color='green')
plt.plot(test.index, prophet_predictions, label='Prophet Predicciones', color='blue')
plt.title('Comparación de Predicciones - ARIMA, ARIMAX, Prophet')
plt.xlabel('Fecha')
plt.ylabel('Irradiación G(i)')
plt.legend()
plt.show()

"""
Universitat Carlemany - Bachelor en Data Science

Proyecto Final de Bachelor: 
"Optimización Económica del Almacenamiento de Energía en Sistemas Fotovoltaicos Mediante Algoritmos de Machine Learning"

Autor: Pablo Felipe Cano Galán
Descripción: Este código forma parte del trabajo de fin de grado (TFB), centrado en el desarrollo de modelos de machine learning para la predicción de momentos óptimos de carga y descarga de baterías en sistemas fotovoltaicos, con el fin de maximizar la eficiencia energética y la rentabilidad económica de dichos sistemas. El proyecto incluye la implementación de herramientas de simulación y una aplicación web para facilitar la gestión de sistemas fotovoltaicos.

Fecha: 07/09/2024
"""

import pandas as pd

# Cargar el dataset completo
df = pd.read_csv("pvpc_prices.csv", parse_dates=['time'])

# Extraer día del año, mes y hora para agrupación
df['day'] = df['time'].dt.dayofyear
df['hour'] = df['time'].dt.hour

# Agrupar por día del año, mes, hora, capital, slope y azimuth para calcular promedios
grouped = df.groupby(['capital', 'slope', 'azimuth', 'day', 'hour']).mean().reset_index()

# Seleccionar solo las columnas relevantes para el "año típico"
typical_year = grouped[['day', 'hour', 'capital', 'slope', 'azimuth', 'G(i)', 'T2m']]

# Guardar el "año típico" en un archivo CSV
typical_year.to_csv("typical_year_pvpc.csv", index=False)

# Cargar el dataset completo
df = pd.read_csv("spot_prices.csv", parse_dates=['time'])

# Extraer día del año, mes y hora para agrupación
df['day'] = df['time'].dt.dayofyear
df['hour'] = df['time'].dt.hour

# Agrupar por día del año, mes, hora, capital, slope y azimuth para calcular promedios
grouped = df.groupby(['capital', 'slope', 'azimuth', 'day', 'hour']).mean().reset_index()

# Seleccionar solo las columnas relevantes para el "año típico"
typical_year = grouped[['day', 'hour', 'capital', 'slope', 'azimuth', 'G(i)', 'T2m']]

# Guardar el "año típico" en un archivo CSV
typical_year.to_csv("typical_year_spot.csv", index=False)

print("Año típico generado y guardado exitosamente.")

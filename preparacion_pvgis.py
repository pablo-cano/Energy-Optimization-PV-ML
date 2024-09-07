"""
Universitat Carlemany - Bachelor en Data Science

Proyecto Final de Bachelor: 
"Optimización Económica del Almacenamiento de Energía en Sistemas Fotovoltaicos Mediante Algoritmos de Machine Learning"

Autor: Pablo Felipe Cano Galán
Descripción: Este código forma parte del trabajo de fin de grado (TFB), centrado en el desarrollo de modelos de machine learning para la predicción de momentos óptimos de carga y descarga de baterías en sistemas fotovoltaicos, con el fin de maximizar la eficiencia energética y la rentabilidad económica de dichos sistemas. El proyecto incluye la implementación de herramientas de simulación y una aplicación web para facilitar la gestión de sistemas fotovoltaicos.

Fecha: 07/09/2024
"""

import pandas as pd

data = pd.read_csv('typical_year_spain_corrected.csv')

capital_data = data
capital_data = data[(data['capital'] == 'Albacete')]

capital_data['datetime'] = pd.to_datetime((capital_data['day'] - 1) * 24 + capital_data['hour'], unit='h', origin='2023-01-01')
capital_data.set_index('datetime', inplace=True)
#capital_data.drop(columns=['day', 'hour','capital','slope','azimuth'], inplace=True)
capital_data.drop(columns=['day', 'hour'], inplace=True)
print(capital_data.head)
capital_data.to_csv("little_typical_year_spain_corrected.csv")

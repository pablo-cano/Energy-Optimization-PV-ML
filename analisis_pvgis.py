"""
Universitat Carlemany - Bachelor en Data Science

Proyecto Final de Bachelor: 
"Optimización Económica del Almacenamiento de Energía en Sistemas Fotovoltaicos Mediante Algoritmos de Machine Learning"

Autor: Pablo Felipe Cano Galán
Descripción: Este código forma parte del trabajo de fin de grado (TFB), centrado en el desarrollo de modelos de machine learning para la predicción de momentos óptimos de carga y descarga de baterías en sistemas fotovoltaicos, con el fin de maximizar la eficiencia energética y la rentabilidad económica de dichos sistemas. El proyecto incluye la implementación de herramientas de simulación y una aplicación web para facilitar la gestión de sistemas fotovoltaicos.

Fecha: 07/09/2024
"""

import pandas as pd
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
import plotly.express as px

def Prueba_Dickey_Fuller(series , column_name):
    print (f'Resultados de la prueba de Dickey-Fuller para columna: {column_name}')
    dftest = adfuller(series, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','No Lags Used','Número de observaciones utilizadas'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    if dftest[1] <= 0.05:
        print("Conclusion:====>")
        print("Rechazar la hipótesis nula")
        print("Los datos son estacionarios")
    else:
        print("Conclusion:====>")
        print("No se puede rechazar la hipótesis nula")
        print("Los datos no son estacionarios")

df = pd.read_csv("little_typical_year_spain_corrected.csv")
df.info()

fig = px.line(df, x=df["datetime"], y="G(i)",
              title="Irradiación")
fig.show()

fig = px.line(df, x=df["datetime"], y="T2m",
              title="Temperatura")
fig.show()

Prueba_Dickey_Fuller(df["G(i)"],"G(i)")

df1=df.copy()
df1['G(i)'] = df['G(i)'].diff()
df1.dropna(inplace=True)
df1.head()
Prueba_Dickey_Fuller(df1["G(i)"],"G(i)")

df1=df.copy()
df1['T2m'] = df['T2m'].diff()
df1.dropna(inplace=True)
df1.head()
Prueba_Dickey_Fuller(df1["T2m"],"T2m")

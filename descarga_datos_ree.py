"""
Universitat Carlemany - Grado en Data Science

Proyecto Final de Bachelor: 
"Optimización Económica del Almacenamiento de Energía en Sistemas Fotovoltaicos Mediante Algoritmos de Machine Learning"

Autor: Pablo Felipe Cano Galán
Descripción: Este código forma parte del trabajo de fin de grado (TFG), centrado en el desarrollo de modelos de machine learning para la predicción de momentos óptimos de carga y descarga de baterías en sistemas fotovoltaicos, con el fin de maximizar la eficiencia energética y la rentabilidad económica de dichos sistemas. El proyecto incluye la implementación de herramientas de simulación y una aplicación web para facilitar la gestión de sistemas fotovoltaicos.

Fecha: 07/09/2024
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Archivo de salida
pvpc_file = "pvpc_prices.csv"
spot_file = "spot_prices.csv"

# Bloqueo para evitar conflictos al escribir en el archivo
lock = threading.Lock()

def download_pvpc_and_spot_prices_for_day(date):
    url = f"https://apidatos.ree.es/es/datos/mercados/precios-mercados-tiempo-real?start_date={date}T00:00&end_date={date}T23:59&time_trunc=hour"
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json; application/vnd.esios-api-v1+json'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        pvpc_values = data['included'][0]['attributes']['values']
        spot_values = data['included'][1]['attributes']['values']
        
        pvpc_df = pd.DataFrame(pvpc_values)
        spot_df = pd.DataFrame(spot_values)
        
        # Convertir la columna 'datetime' a formato datetime
        pvpc_df['datetime'] = pd.to_datetime(pvpc_df['datetime'])
        spot_df['datetime'] = pd.to_datetime(spot_df['datetime'])
        
        with lock:
            pvpc_df.to_csv(pvpc_file, mode='a', header=not pd.io.common.file_exists(pvpc_file), index=False)
            spot_df.to_csv(spot_file, mode='a', header=not pd.io.common.file_exists(spot_file), index=False)
        
        print(f"Datos para {date} descargados y guardados correctamente.")
    else:
        print(f"Error al obtener datos de {url}: {response.status_code}")

# Fechas de inicio y fin
start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 12, 31)
date_generated = [start_date + timedelta(days=x) for x in range(0, (end_date-start_date).days + 1)]

# Descargar los datos con hilos
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = []
    for date in date_generated:
        futures.append(executor.submit(download_pvpc_and_spot_prices_for_day, date.strftime('%Y-%m-%d')))
    
    # Esperar a que todos los hilos terminen
    for future in as_completed(futures):
        try:
            future.result()
        except Exception as exc:
            print(f"Generó una excepción: {exc}")

print("Todos los datos han sido descargados y guardados correctamente.")

import requests
import pandas as pd
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Lista de capitales de provincia de España con sus coordenadas (latitud y longitud)
capitales = {
    "Albacete": (38.9943, -1.8585),
    "Alicante": (38.3452, -0.4810),
    "Almería": (36.8340, -2.4637),
    "Ávila": (40.6565, -4.6818),
    "Badajoz": (38.8786, -6.9700),
    "Barcelona": (41.3851, 2.1734),
    "Bilbao": (43.2630, -2.9350),
    "Burgos": (42.3439, -3.6969),
    "Cáceres": (39.4763, -6.3722),
    "Cádiz": (36.5271, -6.2886),
    "Castellón": (39.9864, -0.0513),
    "Ciudad Real": (38.9848, -3.9276),
    "Córdoba": (37.8882, -4.7794),
    "Cuenca": (40.0707, -2.1374),
    "Girona": (41.9794, 2.8214),
    "Granada": (37.1773, -3.5986),
    "Guadalajara": (40.6333, -3.1669),
    "Huelva": (37.2614, -6.9447),
    "Huesca": (42.1401, -0.4089),
    "Jaén": (37.7796, -3.7849),
    "León": (42.5987, -5.5671),
    "Lleida": (41.6176, 0.6200),
    "Logroño": (42.4627, -2.4445),
    "Lugo": (43.0125, -7.5550),
    "Madrid": (40.4168, -3.7038),
    "Málaga": (36.7213, -4.4214),
    "Murcia": (37.9922, -1.1307),
    "Palencia": (42.0095, -4.5270),
    "Las Palmas": (28.1235, -15.4363),
    "Pontevedra": (42.4310, -8.6444),
    "Salamanca": (40.9701, -5.6635),
    "Santa Cruz de Tenerife": (28.4636, -16.2518),
    "Santander": (43.4623, -3.8098),
    "Segovia": (40.9429, -4.1080),
    "Sevilla": (37.3891, -5.9845),
    "Soria": (41.7661, -2.4797),
    "Tarragona": (41.1189, 1.2445),
    "Teruel": (40.3440, -1.1069),
    "Toledo": (39.8628, -4.0273),
    "Valencia": (39.4699, -0.3763),
    "Valladolid": (41.6523, -4.7245),
    "Vitoria": (42.8469, -2.6725),
    "Zamora": (41.5033, -5.7447),
    "Zaragoza": (41.6490, -0.8891),
}

# Ángulos de inclinación y orientaciones
inclinaciones = [0, 10, 20, 30]
orientaciones = ["180", "90"]  # Sur, Este

# Calcular el número total de combinaciones
total_combinations = len(capitales) * len(inclinaciones) * len(orientaciones)
completed_combinations = 0

# Archivo de salida
output_file = "combined_data_spain.csv"

# Bloqueo para evitar conflictos al escribir en el archivo
lock = threading.Lock()

def download_and_save_data(capital, lat, lon, slope, azimuth):
    global completed_combinations
    
    url = f"https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat={lat}&lon={lon}&startyear=2005&endyear=2020&slope={slope}&aspect={azimuth}&outputformat=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data and 'outputs' in data and 'hourly' in data['outputs']:
            df = pd.DataFrame(data['outputs']['hourly'])
            df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M', errors='coerce')
            df['time'] = df['time'].dt.floor('h')  # Eliminar los minutos
            df['capital'] = capital
            df['slope'] = slope
            df['azimuth'] = azimuth
            df = df[['time', 'G(i)', 'T2m', 'capital', 'slope', 'azimuth']]
            
            # Bloquear y agregar los datos al archivo CSV
            with lock:
                df.to_csv(output_file, mode='a', header=not pd.io.common.file_exists(output_file), index=False)
                completed_combinations += 1
                print(f"Progreso: {completed_combinations}/{total_combinations} combinaciones completadas.")
    else:
        print(f"Error al obtener datos de {url}: {response.status_code}")

# Crear un pool de hilos con un máximo de 10 hilos simultáneos
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = []
    for capital, (lat, lon) in capitales.items():
        for slope in inclinaciones:
            for azimuth in orientaciones:
                futures.append(executor.submit(download_and_save_data, capital, lat, lon, slope, azimuth))
    
    # Esperar a que todos los hilos terminen y mostrar el progreso
    for future in as_completed(futures):
        try:
            future.result()
        except Exception as exc:
            print(f"Generó una excepción: {exc}")

print("Datos descargados y unificados exitosamente.")

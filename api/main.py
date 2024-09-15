"""
Universitat Carlemany - Bachelor en Data Science

Proyecto Final de Bachelor: 
"Optimización Económica del Almacenamiento de Energía en Sistemas Fotovoltaicos Mediante Algoritmos de Machine Learning"

Autor: Pablo Felipe Cano Galán
Descripción: Este código forma parte del trabajo de fin de grado (TFB), centrado en el desarrollo de modelos de machine learning para la predicción de momentos óptimos de carga y descarga de baterías en sistemas fotovoltaicos, con el fin de maximizar la eficiencia energética y la rentabilidad económica de dichos sistemas. El proyecto incluye la implementación de herramientas de simulación y una aplicación web para facilitar la gestión de sistemas fotovoltaicos.

Fecha: 07/09/2024
"""

from fastapi import FastAPI
from pydantic import BaseModel
import datetime
import os
import joblib
import pandas as pd
import numpy as np
import pickle

app = FastAPI()

# Modelo de datos para la entrada del usuario
class UserInput(BaseModel):
    ciudad: str
    fecha_estimacion: str  # Fecha en formato 'YYYY-MM-DD'
    orientacion: float
    inclinacion: float
    area_disponible: float
    potencia_pico_panel: float
    capacidad_bateria: float
    carga_inicial_bateria: float  # Nivel de carga inicial de la batería (kWh)
    costo_carga_inicial_bateria: float  # Costo asociado a la carga inicial (€)
    consumo_mensual: float
    tamano_panel: float

@app.post("/calcular")
def calcular(user_input: UserInput):
    # Validar y parsear la fecha de estimación
    try:
        fecha_estimacion = datetime.datetime.strptime(user_input.fecha_estimacion, '%Y-%m-%d')
    except ValueError:
        return {"error": "La fecha de estimación debe estar en formato 'YYYY-MM-DD'"}

    # Ruta a los archivos (ajusta esta ruta según tu estructura de archivos)
    base_path = 'pkl'  # Reemplaza con la ruta real de tus archivos

    # Cargar los modelos y datos
    try:
        # Cargar el modelo de irradiación solar (SARIMAX)
        irradiacion_model = joblib.load(os.path.join(base_path, 'irradiation_model.pkl'))

        # Cargar los modelos de Prophet para precio de energía y tarifa PVPC
        with open(os.path.join(base_path, 'price_model.pkl'), 'rb') as f:
            precio_energia_model = pickle.load(f)
        with open(os.path.join(base_path, 'pvpc_model.pkl'), 'rb') as f:
            tarifa_pvpc_model = pickle.load(f)

        # Cargar el modelo de perfil de consumo horario (Prophet)
        with open(os.path.join(base_path, 'profile_model.pkl'), 'rb') as f:
            perfil_consumo_horario_model = pickle.load(f)
    except Exception as e:
        return {"error": f"No se pudieron cargar los modelos o datos: {e}"}

    # Generar un rango de fechas para el día de estimación
    horas = 24
    date_str = fecha_estimacion.strftime('%Y-%m-%d')
    fechas = pd.date_range(start=date_str, periods=horas, freq='H')
    df_pred = pd.DataFrame({'ds': fechas})
    df_pred['ds'] = df_pred['ds'].dt.tz_localize(None)

    # Generar predicciones
    try:
        # Predicción de irradiación solar con SARIMAX
        irradiacion_pred = irradiacion_model.predict(start=fechas[0], end=fechas[-1])
        irradiacion_solar = np.maximum(irradiacion_pred, 0)  # Evitar valores negativos

        # Forzar a cero la irradiación en las horas nocturnas
        for i, fecha in enumerate(fechas):
            if fecha.hour < 6 or fecha.hour > 18:  # Ajusta las horas según la temporada y ubicación
                irradiacion_solar[i] = 0

        # Predicción de precio de energía con Prophet
        precio_energia_pred = precio_energia_model.predict(df_pred)
        precio_energia = precio_energia_pred['yhat'].values

        # Predicción de tarifa PVPC con Prophet
        tarifa_pvpc_pred = tarifa_pvpc_model.predict(df_pred)
        tarifa_pvpc = tarifa_pvpc_pred['yhat'].values

        # Predicción de perfil de consumo horario con Prophet
        perfil_consumo_pred = perfil_consumo_horario_model.predict(df_pred)
        perfil_consumo_horario = perfil_consumo_pred['yhat'].values

        # Normalizar el perfil de consumo horario
        total_consumo = np.sum(perfil_consumo_horario)
        if total_consumo == 0:
            return {"error": "El perfil de consumo horario tiene suma cero, no se puede normalizar"}
        perfil_consumo_horario = perfil_consumo_horario / total_consumo  # Normalizar
    except Exception as e:
        # Capturar la traza completa del error
        import traceback
        error_details = traceback.format_exc()
        return {"error": f"No se pudieron generar las predicciones: {e}", "details": error_details}

    # Asegurarnos de que los datos tengan 24 valores
    if not (len(irradiacion_solar) == len(precio_energia) == len(tarifa_pvpc) == len(perfil_consumo_horario) == horas):
        return {"error": f"Los datos para la fecha {date_str} no tienen 24 valores por hora"}

    # Cálculo del área total de paneles
    numero_paneles = user_input.area_disponible / user_input.tamano_panel
    area_total_paneles = numero_paneles * user_input.tamano_panel

    # Calcular el consumo diario
    dias_en_mes = 30  # Días del mes
    consumo_diario = user_input.consumo_mensual / dias_en_mes

    # Definir eficiencia del panel y de la batería
    eficiencia_panel = 0.2  # Eficiencia paneles
    eficiencia_bateria = 0.9  # Supongamos una eficiencia del 90%

    # Inicializar variables de la batería
    capacidad_bateria_max = user_input.capacidad_bateria
    capacidad_bateria_actual = user_input.carga_inicial_bateria  # Nivel de carga inicial de la batería

    # Validaciones
    if capacidad_bateria_actual > capacidad_bateria_max:
        return {"error": "La carga inicial de la batería no puede exceder su capacidad máxima"}

    if capacidad_bateria_actual < 0:
        return {"error": "La carga inicial de la batería no puede ser negativa"}

    # Convertir tarifas de PVPC y precios de energía a €/kWh
    tarifa_pvpc_kwh = tarifa_pvpc / 1000  # De €/MWh a €/kWh
    precio_energia_kwh = precio_energia / 1000

    # Inicializar listas para los resultados horarios
    energia_generada_por_hora = []
    consumo_por_hora = []
    energia_autoconsumida = []
    energia_almacenada = []
    energia_vendida = []
    energia_bateria_utilizada = []
    capacidad_bateria_por_hora = []

    for hora in range(horas):
        # Energía generada por hora (kWh)
        irradiacion_kW_m2 = irradiacion_solar[hora] / 1000  # Convertir W/m² a kW/m²
        energia_generada = irradiacion_kW_m2 * area_total_paneles * eficiencia_panel
        energia_generada_por_hora.append(energia_generada)

        # Consumo por hora (kWh)
        consumo = consumo_diario * perfil_consumo_horario[hora]
        consumo_por_hora.append(consumo)

        # Energía autoconsumida directamente de la generación
        autoconsumida = min(energia_generada, consumo)
        energia_autoconsumida.append(autoconsumida)

        # Energía excedente después del autoconsumo directo
        excedente = max(energia_generada - autoconsumida, 0)

        # Energía necesaria adicional después del autoconsumo directo
        energia_necesaria = consumo - autoconsumida

        # Decisiones sobre el uso de la batería
        # Carga de la batería desde la generación solar
        almacenada = 0
        if excedente > 0 and capacidad_bateria_actual < capacidad_bateria_max:
            espacio_bateria = capacidad_bateria_max - capacidad_bateria_actual
            energia_a_almacenar = min(excedente * eficiencia_bateria, espacio_bateria)
            capacidad_bateria_actual += energia_a_almacenar
            almacenada = energia_a_almacenar
            excedente -= energia_a_almacenar / eficiencia_bateria  # Ajustar el excedente

        energia_almacenada.append(almacenada)

        # Energía vendida
        vendida = 0
        if excedente > 0:
            vendida = excedente
            excedente = 0

        energia_vendida.append(vendida)

        # Uso de la batería
        energia_bateria_usada = 0
        if energia_necesaria > 0 and capacidad_bateria_actual > 0:
            energia_disponible_bateria = capacidad_bateria_actual * eficiencia_bateria
            energia_bateria_usada = min(energia_necesaria, energia_disponible_bateria)
            capacidad_bateria_actual -= energia_bateria_usada / eficiencia_bateria
            energia_necesaria -= energia_bateria_usada

        energia_bateria_utilizada.append(energia_bateria_usada)

        capacidad_bateria_por_hora.append(capacidad_bateria_actual)

    # Calcular el costo de energía comprada a la red si no hubiera paneles solares
    costo_consumo_sin_solar = sum([consumo_por_hora[hora] * tarifa_pvpc_kwh[hora] for hora in range(horas)])

    # Calcular el costo de energía comprada a la red después de autoconsumo
    costo_consumo_con_solar = sum([energia_necesaria * tarifa_pvpc_kwh[hora] for hora in range(horas) if energia_necesaria > 0])

    # Calcular el ahorro total considerando el autoconsumo y la venta de energía excedente
    ahorro_total = costo_consumo_sin_solar - costo_consumo_con_solar + sum(energia_vendida) * np.mean(precio_energia_kwh) - user_input.costo_carga_inicial_bateria

    # Totales de cada concepto
    total_energia_generada = sum(energia_generada_por_hora)
    total_consumo = sum(consumo_por_hora)
    total_energia_autoconsumida = sum(energia_autoconsumida)
    total_energia_almacenada = sum(energia_almacenada)
    total_energia_vendida = sum(energia_vendida)
    total_energia_bateria_utilizada = sum(energia_bateria_utilizada)

    # Preparar los resultados con detalle horario
    resultados_horarios = []
    for hora in range(horas):
        resultados_horarios.append({
            "hora": hora,
            "energia_generada": energia_generada_por_hora[hora],
            "consumo": consumo_por_hora[hora],
            "energia_autoconsumida": energia_autoconsumida[hora],
            "energia_almacenada": energia_almacenada[hora],
            "energia_vendida": energia_vendida[hora],
            "energia_bateria_utilizada": energia_bateria_utilizada[hora],
            "capacidad_bateria_actual": capacidad_bateria_por_hora[hora]
        })

    # Preparar la respuesta final con los totales
    resultados = {
        "fecha": date_str,
        "carga_inicial_bateria": user_input.carga_inicial_bateria,
        "costo_carga_inicial_bateria": user_input.costo_carga_inicial_bateria,
        "ahorro_total": ahorro_total,
        "totales": {
            "total_energia_generada": total_energia_generada,
            "total_consumo": total_consumo,
            "total_energia_autoconsumida": total_energia_autoconsumida,
            "total_energia_almacenada": total_energia_almacenada,
            "total_energia_vendida": total_energia_vendida,
            "total_energia_bateria_utilizada": total_energia_bateria_utilizada,
        },
        "resultados_horarios": resultados_horarios
    }

    return resultados

"""
Universitat Carlemany - Bachelor en Data Science

Proyecto Final de Bachelor: 
"Optimización Económica del Almacenamiento de Energía en Sistemas Fotovoltaicos Mediante Algoritmos de Machine Learning"

Autor: Pablo Felipe Cano Galán
Descripción: Este código forma parte del trabajo de fin de grado (TFB), centrado en el desarrollo de modelos de machine learning para la predicción de momentos óptimos de carga y descarga de baterías en sistemas fotovoltaicos, con el fin de maximizar la eficiencia energética y la rentabilidad económica de dichos sistemas. El proyecto incluye la implementación de herramientas de simulación y una aplicación web para facilitar la gestión de sistemas fotovoltaicos.

Fecha: 07/09/2024
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import os
from datetime import datetime

# Configurar el Service para usar ChromeDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# Navega a la página web
driver.get('https://www.ree.es/en/node/13430')

# Espera a que la página cargue completamente
wait = WebDriverWait(driver, 10)

# Encuentra los menús desplegables por su identificador o nombre
combo1 = Select(wait.until(EC.presence_of_element_located((By.ID, 'edit-month'))))
combo2 = Select(wait.until(EC.presence_of_element_located((By.ID, 'edit-year'))))

# Obtener el mes y año actual
current_year = datetime.now().year
current_month = datetime.now().month

# Iterar sobre todas las combinaciones posibles hasta el mes anterior del año actual
for option2 in combo2.options:
    year = int(option2.get_attribute('value'))
    if year > current_year:
        continue  # Saltar años futuros

    for option1 in combo1.options:
        month = int(option1.get_attribute('value'))
        if year == current_year and month >= current_month:
            continue  # Saltar el mes actual y futuros meses

        combo1.select_by_value(option1.get_attribute('value'))
        combo2.select_by_value(option2.get_attribute('value'))

        # Usar JavaScript para hacer clic en el botón
        try:
            download_button = driver.find_element(By.ID, 'submit_simel')
            driver.execute_script("arguments[0].click();", download_button)
            print(f"Descargando archivo para {option1.text} {option2.text}")
            
            # Esperar a que la descarga se inicie y complete
            time.sleep(4) 

        except Exception as e:
            print(f"Error al hacer clic en el botón: {e}")

# Cierra el navegador
driver.quit()

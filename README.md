# Optimización Económica del Almacenamiento de Energía en Sistemas Fotovoltaicos Mediante Algoritmos de Machine Learning

![Logo EnergyPV](LogoProyecto.bmp)

## Introducción

Este proyecto aborda el desafío de optimizar el uso de baterías en sistemas fotovoltaicos, con el objetivo de maximizar la rentabilidad económica en un entorno de fluctuaciones de precios de la energía. Aprovechando el poder de la inteligencia artificial y los algoritmos de machine learning, esta herramienta proporciona predicciones precisas que permiten a los usuarios gestionar de forma eficiente sus sistemas fotovoltaicos.

### Objetivo del Proyecto

El principal objetivo es predecir los momentos óptimos para la carga y descarga de baterías, utilizando modelos que consideran factores como:

- Radiación solar
- Condiciones meteorológicas
- Precios dinámicos de la electricidad

### Características Principales

- **Aplicación Web**: Una interfaz intuitiva para que los usuarios puedan configurar sus sistemas y recibir predicciones personalizadas.
- **Predicciones basadas en Machine Learning**: Implementación de algoritmos de machine learning para optimizar el uso de las baterías, maximizando el ahorro y los ingresos por la venta de excedentes a la red.
- **Simulación de Estrategias**: Un entorno de simulación para probar diferentes estrategias de optimización y mejorar la eficiencia energética.
- **Validación con Datos Reales**: Utilización de datos de consumo y generación solar de varios municipios para validar el modelo en escenarios reales.

### Beneficios

Este sistema ha demostrado aumentar significativamente la eficiencia energética y los beneficios económicos de los usuarios, logrando:

- Reducción de costos operativos.
- Incremento de ingresos mediante la venta de excedentes energéticos.
- Mejora en la sostenibilidad de los sistemas de energía renovable.

### Requisitos Técnicos

- **Backend**: Implementación en FastAPI para el procesamiento de los datos y ejecución de los modelos de machine learning.
- **Frontend**: Interfaz desarrollada con Mendix para una experiencia de usuario sencilla y visualmente atractiva.
- **Base de Datos**: Uso de PostgreSQL para almacenar datos históricos de radiación solar, precios de la energía y consumo.

### Conclusión

Este proyecto demuestra el potencial de la inteligencia artificial para transformar la gestión energética, ofreciendo una solución innovadora para mejorar el uso de las energías renovables de manera rentable y sostenible.

---

## Estructura del Repositorio

1. **Descarga de los datos de PVGIS**  
   [descarga_datos_radiacion_temperatura.py](https://github.com/pablo-cano/Energy-Optimization-PV-ML/blob/main/descarga_datos_radiacion_temperatura.py)
   
2. **Análisis de los datos de PVGIS**  
   [analisis_pvgis.py](https://github.com/pablo-cano/Energy-Optimization-PV-ML/blob/main/analisis_pvgis.py)
   
3. **Descarga de los datos de precios REE**  
   [descarga_datos_precios_ree.py](https://github.com/pablo-cano/Energy-Optimization-PV-ML/blob/main/descarga_datos_precios_ree.py)
   
4. **Análisis de los datos de precios de REE**  
   [analisis_precios_ree.py](https://github.com/pablo-cano/Energy-Optimization-PV-ML/blob/main/analisis_precios_ree.py)
   
5. **Descarga de los datos del perfil de consumo de REE**  
   [descarga_datos_perfiles_ree.py](https://github.com/pablo-cano/Energy-Optimization-PV-ML/blob/main/descarga_datos_perfiles_ree.py)
   
6. **Análisis de los datos del perfil de consumo de REE**  
   [analisis_perfiles_ree.py](https://github.com/pablo-cano/Energy-Optimization-PV-ML/blob/main/analisis_perfiles_ree.py)
   
7. **Preparación de los datos PVGIS**  
   [preparacion_pvgis.py](https://github.com/pablo-cano/Energy-Optimization-PV-ML/blob/main/preparacion_pvgis.py)
   
8. **Preparación de los datos para unificar todos los orígenes**  
   [preparacion_union_datos.py](https://github.com/pablo-cano/Energy-Optimization-PV-ML/blob/main/preparacion_union_datos.py)
   
9. **Generación del modelo ARIMA**  
   [modelo_arima.py](https://github.com/pablo-cano/Energy-Optimization-PV-ML/blob/main/modelo_arima.py)
   
10. **Generación del modelo ARIMAX con la variable exógena de la temperatura**  
    [modelo_arimax.py](https://github.com/pablo-cano/Energy-Optimization-PV-ML/blob/main/modelo_arimax.py)
   
11. **Generación del modelo ARIMAX sin la variable exógena de la temperatura**  
    [modelo_arimax_sin_exo.py](https://github.com/pablo-cano/Energy-Optimization-PV-ML/blob/main/modelo_arimax_sin_exo.py)
    
12. **Generación del modelo Prophet**  
    [modelo_prophet.py](https://github.com/pablo-cano/Energy-Optimization-PV-ML/blob/main/modelo_prophet.py)
   
13. **Generación del modelo Prophet para la tarifa PVPC**  
    [modelo_prophet_pvpc.py](https://github.com/pablo-cano/Energy-Optimization-PV-ML/blob/main/modelo_prophet_pvpc.py)
   
14. **Generación del modelo Prophet para el precio de la energía**  
    [modelo_prophet_pe.py](https://github.com/pablo-cano/Energy-Optimization-PV-ML/blob/main/modelo_prophet_pe.py)
   
15. **Generación del modelo Prophet para el perfil de consumo**  
    [modelo_prophet_perfil.py](https://github.com/pablo-cano/Energy-Optimization-PV-ML/blob/main/modelo_prophet_perfil.py)
    
16. **Comparativa de modelos irradiación**  
    [comparativa_modelos.py](https://github.com/pablo-cano/Energy-Optimization-PV-ML/blob/main/comparativa_modelos.py)

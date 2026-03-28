# Predicción de Precios de Criptomonedas con LSTM

Este proyecto utiliza redes neuronales LSTM para predecir el precio de cierre de Bitcoin (BTC-USD) usando indicadores técnicos como SMA y RSI.

## Estructura del Proyecto

- `Modelo_Predicciones/Generar_Modelo.py`: Entrena el modelo LSTM multivariable y guarda el modelo y el scaler.
- `Modelo_Predicciones/Empleo_Modelo.py`: Usa el modelo entrenado para predecir el precio 1 y 3 pasos adelante.
- `crypto_multi_model.keras`: Archivo del modelo entrenado.
- `scaler.pkl`: Escalador de características para normalizar los datos.
- `training_loss.png`: Gráfico de la pérdida durante el entrenamiento.
- `.gitignore`: Exclusiones recomendadas para Python y ML.

## Requisitos

- Python 3.8+
- Paquetes: numpy, pandas, yfinance, scikit-learn, tensorflow, matplotlib

Instala dependencias con:

```
pip install -r requirements.txt
```

## Uso

### 1. Entrenar el modelo

Ejecuta:

```
python Modelo_Predicciones/Generar_Modelo.py
```

Esto descargará los datos, entrenará el modelo y generará los archivos necesarios.

### 2. Realizar predicciones

Ejecuta:

```
python Modelo_Predicciones/Empleo_Modelo.py
```

Obtendrás la predicción para 1 y 3 pasos adelante del precio de cierre de BTC-USD.

## Notas

- El modelo utiliza los últimos 60 días de datos para predecir el siguiente valor.
- El entrenamiento muestra la evolución de la pérdida y guarda un gráfico en `training_loss.png`.
- Puedes modificar los scripts para usar otros activos o ajustar los hiperparámetros.

## Autor

- Proyecto para fines educativos.

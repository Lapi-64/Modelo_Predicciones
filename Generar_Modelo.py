import numpy as np
import pandas as pd
import yfinance as yf
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Configuración
CRYPTO = 'BTC-USD'
START_DATE = '2021-01-01'
WINDOW_SIZE = 60

# 1. Obtener datos y calcular indicadores
data = yf.download(CRYPTO, start=START_DATE)
df = data[['Close']].copy()

# Añadir Media Móvil Simple (SMA) de 20 días
df['SMA_20'] = df['Close'].rolling(window=20).mean()

# Añadir RSI (Relative Strength Index)
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# Limpiar valores nulos creados por los indicadores
df.dropna(inplace=True)

# 2. Escalado de múltiples características (Features)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Guardar el scaler para usarlo después en la predicción
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 3. Preparar datos para LSTM [muestras, tiempo, características]
X, y = [], []
for i in range(WINDOW_SIZE, len(scaled_data)):
    X.append(scaled_data[i-WINDOW_SIZE:i, :]) # Todas las columnas (Precio, SMA, RSI)
    y.append(scaled_data[i, 0]) # Solo predecimos el Precio de cierre

X, y = np.array(X), np.array(y)

# 4. Modelo con múltiples entradas
model = Sequential([
    LSTM(60, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(60, return_sequences=False),
    Dropout(0.2),
    Dense(20, activation='relu'),
    Dense(1)
])

import matplotlib.pyplot as plt

model.compile(optimizer='adam', loss='mse')
print("Entrenando modelo multivariable...")
history = model.fit(X, y, epochs=15, batch_size=32, verbose=1)

# Mostrar la pérdida final
final_loss = history.history['loss'][-1]
print(f"Pérdida final de entrenamiento: {final_loss}")

# Graficar la pérdida durante el entrenamiento
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.title('Evolución de la pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida (MSE)')
plt.legend()
plt.tight_layout()
plt.savefig('training_loss.png')
plt.show()

model.save('crypto_multi_model.keras')
print("Modelo y Scaler guardados.")
import numpy as np
import yfinance as yf
import pickle
from tensorflow.keras.models import load_model

# Cargar archivos
model = load_model('crypto_multi_model.keras')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Obtener datos recientes
raw_data = yf.download('BTC-USD', period='100d')
temp_df = raw_data[['Close']].copy()
temp_df['SMA_20'] = temp_df['Close'].rolling(window=20).mean()

delta = temp_df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
temp_df['RSI'] = 100 - (100 / (1 + (gain/loss)))

# Preparar entrada para el modelo
last_data = temp_df.dropna().tail(60)
scaled_input = scaler.transform(last_data)
X_input = np.array([scaled_input])

# Predecir
pred_scaled = model.predict(X_input)

# Invertir el escalado (solo para la columna de precio)
# Creamos un array dummy para que el scaler funcione correctamente
dummy = np.zeros((1, 3)) 
dummy[0, 0] = pred_scaled.flatten()[0]
final_pred = scaler.inverse_transform(dummy)[0, 0]


# Predicción 1 paso adelante
print(f"\nPredicción 1 paso adelante: ${final_pred:.2f}")

# Predicción 3 pasos adelante (iterativa)
future_input = scaled_input.copy()
future_preds = []
for i in range(3):
    X_future = np.array([future_input[-60:]])
    pred_scaled = model.predict(X_future, verbose=0)
    # Guardar predicción
    dummy = np.zeros((1, 3))
    dummy[0, 0] = pred_scaled.flatten()[0]
    pred_real = scaler.inverse_transform(dummy)[0, 0]
    future_preds.append(pred_real)
    # Actualizar future_input para el siguiente paso
    next_row = future_input[-1].copy()
    next_row[0] = pred_scaled.flatten()[0]
    # Opcional: mantener SMA y RSI igual que el último valor conocido
    future_input = np.vstack([future_input, next_row])

print("Predicción 3 pasos adelante:")
for i, val in enumerate(future_preds, 1):
    print(f"  Paso {i}: ${val:.2f}")
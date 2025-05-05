

# Reshape the data for LSTM
timesteps = 10
features = X_train.shape[1]

X_train_lstm = []
y_train_lstm = []
X_test_lstm = []

for i in range(timesteps, len(X_train)):
    X_train_lstm.append(X_train[i-timesteps:i].values)
    y_train_lstm.append(y_train.iloc[i])

for i in range(timesteps, len(X_test)):
    X_test_lstm.append(X_test[i-timesteps:i].values)

X_train_lstm = np.array(X_train_lstm)
y_train_lstm = np.array(y_train_lstm)
X_test_lstm = np.array(X_test_lstm)

X_train_lstm = np.reshape(X_train_lstm, (X_train_lstm.shape[0], X_train_lstm.shape[1], features))
X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], features))




X_val_lstm = X_test_lstm
y_val_lstm = y_test[timesteps:].values




import itertools
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 1. Define Hyperparameter-Raster
units_list = [32, 64, 128]
learning_rates = [0.001, 0.0005, 0.0001]
batch_sizes = [16, 32, 64]
dropout_rates = [0.0, 0.2, 0.5]


def create_lstm_model(input_shape, units, dropout_rate, learning_rate):
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))  # Output Layer 
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# 3. Preparation
best_val_loss = np.inf
best_hyperparams = None
input_shape = (X_train_lstm.shape[1], X_train_lstm.shape[2])

# 4. Grid Search Loop
for units, lr, batch_size, dropout in itertools.product(units_list, learning_rates, batch_sizes, dropout_rates):
    print(f"Training with units={units}, learning_rate={lr}, batch_size={batch_size}, dropout={dropout}")

    model = create_lstm_model(input_shape, units, dropout, lr)

    history = model.fit(
        X_train_lstm, y_train_lstm,
        validation_data=(X_val_lstm, y_val_lstm),
        epochs=20,    
        batch_size=batch_size,
        verbose=0
    )

    val_loss = min(history.history['val_loss'])

    print(f"Validation Loss: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_hyperparams = {
            'units': units,
            'learning_rate': lr,
            'batch_size': batch_size,
            'dropout': dropout
        }

# 5. Beste Hyperparameter ausgeben

print(best_hyperparams)
print(f"Best Validation Loss: {best_val_loss:.6f}")




final_model = Sequential()
final_model.add(LSTM(best_hyperparams['units'], return_sequences=True, input_shape=(X_train_lstm.shape[1], features)))
final_model.add(Dropout(best_hyperparams['dropout']))
final_model.add(LSTM(best_hyperparams['units'], return_sequences=False))
final_model.add(Dense(1))

final_model.compile(optimizer=Adam(learning_rate=best_hyperparams['learning_rate']), loss='mse')

# Trainiere das Modell erneut mit den besten Hyperparametern
final_model.fit(
    X_train_lstm, y_train_lstm,
    epochs=50,  # Du kannst das anpassen
    batch_size=best_hyperparams['batch_size'],
    verbose=1
)

# Neue Vorhersagen machen
y_pred_lstm = final_model.predict(X_test_lstm).flatten()

# Neue Vorhersagen in predictions_new schreiben
predictions_new["Predicted_LSTM"] = np.nan
predictions_new["Predicted_LSTM"].iloc[-len(y_pred_lstm):] = y_pred_lstm

#BLOCK 20

# The first ten values for the LSTM model will be NaN, as they are not predicted
print(predictions_new.head(20))

# I will therefore not select the first ten values for the LSTM model
predictions_new = predictions_new.reset_index()
predictions_new = predictions_new.loc[10:,:]
predictions_new = predictions_new.set_index("Date")
predictions_new = predictions_new.sort_index()

#Block 21

# Plot the predictions vs actual values for LSTM
plt.figure(figsize=(12, 6))
plt.plot(predictions_new.index, predictions_new["Actual"].values, label="Actual", color="blue")
plt.plot(predictions_new.index, predictions_new["Predicted_LSTM"].values, label="LSTM", color="purple")
plt.legend()
plt.title("Predictions vs Actual Values for LSTM")
plt.show()

# Plot everything together
plt.figure(figsize=(12, 6))
plt.plot(predictions_new.index, predictions_new["Actual"].values, label="Actual", color="blue")
plt.plot(predictions_new.index, predictions_new["Predicted_GPR"].values, label="Gaussian Process", color="green")
plt.plot(predictions_new.index, predictions_new["Predicted"].values, label="Neural Network", color="orange")
plt.plot(predictions_new.index, predictions_new["Predicted_LSTM"].values, label="LSTM", color="purple")
plt.legend()
plt.title("Predictions vs Actual Values")
plt.show()
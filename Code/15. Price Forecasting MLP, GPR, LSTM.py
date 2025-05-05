import matplotlib.pyplot as plt

gpr_price = GridSearchCV(
    GaussianProcessRegressor(alpha=0.003),
    {"kernel": kernel_options},
    cv=5,
    verbose=2,
    n_jobs=-1
)

gpr_price.fit(X_train_price, y_train_price)
print("Best Parameter for GPR (Price):", gpr_price.best_params_)
y_pred_price_gpr = gpr_price.predict(X_test_price)



X_train_lstm_price = []
y_train_lstm_price = []
X_test_lstm_price = []

for i in range(timesteps, len(X_train_price_scaled)):
    X_train_lstm_price.append(X_train_price_scaled[i-timesteps:i])
    y_train_lstm_price.append(y_train_price_scaled[i])

for i in range(timesteps, len(X_test_price_scaled)):
    X_test_lstm_price.append(X_test_price_scaled[i-timesteps:i])

X_train_lstm_price = np.array(X_train_lstm_price)
y_train_lstm_price = np.array(y_train_lstm_price)
X_test_lstm_price = np.array(X_test_lstm_price)

X_train_lstm_price = np.reshape(X_train_lstm_price, (X_train_lstm_price.shape[0], timesteps, X_train_price.shape[1]))
X_test_lstm_price = np.reshape(X_test_lstm_price, (X_test_lstm_price.shape[0], timesteps, X_train_price.shape[1]))

# y_test for validation
y_val_price = y_test_price_scaled[timesteps:]





print("X_train_lstm_price shape:", X_train_lstm_price.shape)
print("y_train_lstm_price shape:", y_train_lstm_price.shape)


example_idx = 50  

plt.figure(figsize=(12, 4))
for i in range(X_train_lstm_price.shape[2]):
    plt.plot(X_train_lstm_price[example_idx, :, i], label=f"Feature {i}", alpha=0.5)
plt.title("Input Sequence (X) für Beispielindex 50")
plt.legend()
plt.show()


print("y_train_lstm_price:", y_train_lstm_price[example_idx])



simple_model = Sequential()
simple_model.add(LSTM(64, return_sequences=True, input_shape=(X_train_lstm_price.shape[1], X_train_lstm_price.shape[2])))
simple_model.add(Dropout(0.2))
simple_model.add(LSTM(64, return_sequences=False))
simple_model.add(Dense(1))

simple_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Training
history = simple_model.fit(
    X_train_lstm_price, y_train_lstm_price,
    validation_data=(X_test_lstm_price, y_val_price),
    epochs=100,
    batch_size=32,
    verbose=1
)



y_pred_price_lstm = simple_model.predict(X_test_lstm_price).flatten()


y_pred_price_lstm = scaler_y_price.inverse_transform(y_pred_price_lstm.reshape(-1, 1)).flatten()
y_val_price = scaler_y_price.inverse_transform(y_val_price.reshape(-1, 1)).flatten()


index_valid_price = X_test_price.index[timesteps:]

# predictions_price_clean: including LSTM
predictions_price_clean = pd.DataFrame({
    "Actual": y_val_price,
    "MLP": y_pred_price_mlp[timesteps:],
    "GPR": y_pred_price_gpr[timesteps:],
    "LSTM": y_pred_price_lstm
}, index=index_valid_price)


plt.figure(figsize=(12, 6))
plt.scatter(predictions_price_clean.index, predictions_price_clean["Actual"], label="Actual", color="black", s=10)
plt.scatter(predictions_price_clean.index, predictions_price_clean["MLP"], label="MLP", alpha=0.6, s=10)
plt.scatter(predictions_price_clean.index, predictions_price_clean["GPR"], label="GPR", alpha=0.6, s=10)
plt.scatter(predictions_price_clean.index, predictions_price_clean["LSTM"], label="LSTM", alpha=0.6, s=10, color="green")
plt.title("Predicted vs Actual (log Price) – MLP, GPR & LSTM (Scatter)")
plt.legend()
plt.show()
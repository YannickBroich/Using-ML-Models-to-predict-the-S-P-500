Block 27

X_train_lstm_cls = []
y_train_lstm_cls = []
X_test_lstm_cls = []
timesteps = 10
features_cls = X_train_cls.shape[1]

for i in range(timesteps, len(X_train_cls)):
    X_train_lstm_cls.append(X_train_cls[i-timesteps:i].values)
    y_train_lstm_cls.append(y_train_cls.iloc[i])

for i in range(timesteps, len(X_test_cls)):
    X_test_lstm_cls.append(X_test_cls[i-timesteps:i].values)

X_train_lstm_cls = np.array(X_train_lstm_cls)
y_train_lstm_cls = np.array(y_train_lstm_cls)
X_test_lstm_cls = np.array(X_test_lstm_cls)

X_train_lstm_cls = np.reshape(X_train_lstm_cls, (X_train_lstm_cls.shape[0], X_train_lstm_cls.shape[1], features_cls))
X_test_lstm_cls = np.reshape(X_test_lstm_cls, (X_test_lstm_cls.shape[0], X_test_lstm_cls.shape[1], features_cls))



# y_val for Evaluation
y_val_lstm_cls = y_test_cls[timesteps:].values

# Grid Search Setup
best_val_acc = 0
best_hyper_cls = None

for units, lr, batch_size, dropout in itertools.product(units_list, learning_rates, batch_sizes, dropout_rates):
    print(f"Training (LSTM-Cls) with units={units}, learning_rate={lr}, batch_size={batch_size}, dropout={dropout}")

    model = Sequential()
    model.add(LSTM(units, input_shape=(X_train_lstm_cls.shape[1], X_train_lstm_cls.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        X_train_lstm_cls, y_train_lstm_cls,
        validation_data=(X_test_lstm_cls, y_val_lstm_cls),
        epochs=20,
        batch_size=batch_size,
        verbose=0
    )

    val_acc = max(history.history["val_accuracy"])

    print(f"Validation Accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_hyper_cls = {
            "units": units,
            "learning_rate": lr,
            "batch_size": batch_size,
            "dropout": dropout
        }


print("\nBest Hyperparameter (LSTM Klassifikation):")
print(best_hyper_cls)
print(f"Best Accuracy: {best_val_acc:.4f}")


final_model_cls = Sequential()
final_model_cls.add(LSTM(best_hyper_cls['units'], return_sequences=True, input_shape=(X_train_lstm_cls.shape[1], features_cls)))
final_model_cls.add(Dropout(best_hyper_cls['dropout']))
final_model_cls.add(LSTM(best_hyper_cls['units'], return_sequences=False))
final_model_cls.add(Dense(1, activation="sigmoid"))

final_model_cls.compile(optimizer=Adam(learning_rate=best_hyper_cls['learning_rate']), loss="binary_crossentropy", metrics=["accuracy"])

final_model_cls.fit(
    X_train_lstm_cls, y_train_lstm_cls,
    epochs=50,
    batch_size=best_hyper_cls['batch_size'],
    verbose=1
)

#Predictions
y_pred_lstm_cls = final_model_cls.predict(X_test_lstm_cls).flatten()
y_pred_lstm_cls_binary = (y_pred_lstm_cls > 0.5).astype(int)

# Results
accuracy_lstm = accuracy_score(y_val_lstm_cls, y_pred_lstm_cls_binary)
print(f"\nAccuracy (LSTM Classifier): {accuracy_lstm:.4f}")
print("\nClassification Report (LSTM):\n", classification_report(y_val_lstm_cls, y_pred_lstm_cls_binary))


index_valid_cls = X_test_cls.index[timesteps:]


predictions_cls = pd.DataFrame({
    "Actual": y_val_lstm_cls,
    "MLP": y_pred_cls[timesteps:],
    "GPC": y_pred_gpc[timesteps:],
    "LSTM": y_pred_lstm_cls_binary
}, index=index_valid_cls)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(predictions_cls.index, predictions_cls["Actual"], label="Actual (Up=1)", color="black", linestyle="--")
plt.plot(predictions_cls.index, predictions_cls["MLP"], label="MLP", alpha=0.7)
plt.plot(predictions_cls.index, predictions_cls["GPC"], label="GPC", alpha=0.7)
plt.plot(predictions_cls.index, predictions_cls["LSTM"], label="LSTM", alpha=0.7)
plt.title("Predicted vs Actual Direction (Up = 1, Down = 0)")
plt.legend()
plt.show()
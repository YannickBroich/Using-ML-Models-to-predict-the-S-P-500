# Fit ARMA model and GARCH model simultaneously
from arch import arch_model
from tqdm import tqdm

scale = 100

p = 1
q = 1
g_p = 1
g_q = 1

y_pred_arma_garch = pd.Series(index=y_test.index, dtype=float)

history = y_train.copy() * scale

for t in tqdm(range(len(y_test)), desc="Fitting ARIMA-GARCH model"):

    am = arch_model(
        history,
        mean="ARX", lags=p,
        vol="Garch", p=g_p, q=g_q,
        dist="Normal",
        rescale=False
    )
    am_fit = am.fit(disp="off")

    # Forecast the next value
    forecast = am_fit.forecast(horizon=1, reindex=True)
    y_pred_arma_garch.iloc[t] = forecast.mean.values[-1, 0] / scale

    # Update the history with the actual value
    history.iloc[t] = y_test.iloc[t] * scale

y_pred_arma_garch = y_pred_arma_garch.rename("Predicted_ARMA_GARCH")
predictions_new = pd.concat([predictions_new, y_pred_arma_garch], axis=1)

# Drop NAs
predictions_new = predictions_new.dropna(axis=0, how="any")

# Plot the predictions vs actual values for ARMA-GARCH
plt.figure(figsize=(12, 6))
plt.plot(predictions_new.index, predictions_new["Actual"].values, label="Actual", color="blue")
plt.plot(predictions_new.index, predictions_new["Predicted_ARMA_GARCH"].values, label="ARMA-GARCH", color="brown")
plt.legend()
plt.title("Predictions vs Actual Values for ARMA-GARCH")
plt.show()

# Plot everything together
plt.figure(figsize=(12, 6))
plt.plot(predictions_new.index, predictions_new["Actual"].values, label="Actual", color="blue")
plt.plot(predictions_new.index, predictions_new["Predicted_GPR"].values, label="Gaussian Process", color="green")
plt.plot(predictions_new.index, predictions_new["Predicted"].values, label="Neural Network", color="orange")
plt.plot(predictions_new.index, predictions_new["Predicted_LSTM"].values, label="LSTM", color="purple")
plt.plot(predictions_new.index, predictions_new["Predicted_ARMA_GARCH"].values, label="ARMA-GARCH", color="brown")
plt.legend()
plt.title("Predictions vs Actual Values")
plt.show()

# Evaluate the models using several metrics

predictions_new = predictions_new.rename(columns={"Predicted": "Predicted_NN"})

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0,1]
    return {"MSE": mse, "MAE": mae, "Corr": corr}

metrics = {}
for model in ["NN", "GPR", "LSTM", "ARMA_GARCH"]:
    metrics[model] = evaluate_model(predictions_new["Actual"], predictions_new[f"Predicted_{model}"])

metrics_df = pd.DataFrame(metrics).T
print(metrics_df)





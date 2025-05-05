

# Hyperparameter tuning for Gaussian Process Regression

possible_values = {"kernel": [(C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))), (1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1, alpha_bounds=(1e-5, 1e15)))]}

# Run GridSearch to determine the best set of hyperparameters

model = GridSearchCV(GaussianProcessRegressor(alpha = 0.003), possible_values, cv=5, n_jobs=-1, verbose=2)

model.fit(X_train, y_train)

# Print the best parameters
print("Best parameters found: ", model.best_params_)

# Predict using the best model
y_pred_gpr = model.predict(X_test)



# Concatenate the predictions and actual values from before
predictions_gpr = pd.DataFrame({"Predicted_GPR": y_pred_gpr}, columns=["Predicted_GPR"], index=X_test.index)
predictions_new = pd.concat([predictions, predictions_gpr], axis=1)
print(predictions_new)

# Sort by index to ensure the order is correct
predictions_new = predictions_new.sort_index()

# Plot the predictions vs actual values for GPR
plt.figure(figsize=(12, 6))
plt.plot(predictions_new.index, predictions_new["Actual"].values, label="Actual", color="blue")
plt.plot(predictions_new.index, predictions_new["Predicted_GPR"].values, label="Gaussian Process", color="green")
plt.legend()
plt.title("Predictions vs Actual Values for GPR")
plt.show()
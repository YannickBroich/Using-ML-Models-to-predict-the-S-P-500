
# Define the model
possible_values = {"hidden_layer_sizes": [(100, 100, 100), (200, 200, 200), (300, 300, 300)],
                    "activation": ["relu", "tanh", "logistic"]}

# Run GridSearchCV
model = GridSearchCV(MLPRegressor(max_iter=100000, shuffle = False, solver = "adam", alpha = 10 ** -2), possible_values, cv=5, n_jobs=-1, verbose=2)



# Fit the best model
model.fit(X_train, y_train)

# Print the best parameters
print("Best parameters found: ", model.best_params_)

# Make predictions
y_pred = model.predict(X_test)



# Create a dataframe with the predictions and actual values
predictions = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

predictions = predictions.set_index(X_test.index)
predictions = predictions.sort_index()

# Plot the predictions vs actual values
plt.figure(figsize=(12, 6))
plt.plot(predictions.index, predictions["Actual"].values, label="Actual", color="blue")
plt.plot(predictions.index, predictions["Predicted"].values, label="Neural Network", color="orange")
plt.legend()
plt.title("Predictions vs Actual Values")
plt.show()
from sklearn.preprocessing import StandardScaler

Z["target_price"] = np.log(data.loc[Z.index, "Index"])

#Features
X_price = Z.drop(columns=["target", "target_direction", "target_price"])
y_price = Z["target_price"]

# Train/Test-Split
X_train_price, X_test_price, y_train_price, y_test_price = train_test_split(X_price, y_price, test_size=0.2, train_size=0.8)



#StandardScaler
scaler_X_price = StandardScaler()
scaler_y_price = StandardScaler()

# Scale X
X_train_price_scaled = scaler_X_price.fit_transform(X_train_price)
X_test_price_scaled = scaler_X_price.transform(X_test_price)

# Scale y 
y_train_price_scaled = scaler_y_price.fit_transform(y_train_price.values.reshape(-1, 1)).flatten()
y_test_price_scaled = scaler_y_price.transform(y_test_price.values.reshape(-1, 1)).flatten()





# GridSearchCV for Price-Model
param_grid_price = {
    "hidden_layer_sizes": [(100, 100), (200, 200)],
    "activation": ["relu", "tanh"]
}

mlp_price = GridSearchCV(MLPRegressor(max_iter=100000, solver="adam", alpha=0.01, shuffle=False), param_grid_price, cv=5, n_jobs=-1, verbose=2)
mlp_price.fit(X_train_price, y_train_price)


print("Beste Parameter für MLP (Price):", mlp_price.best_params_)
y_pred_price_mlp = mlp_price.predict(X_test_price)

# Lasso Regression for feature selection
lambdas = np.logspace(-4, 4, 100)
empty = np.zeros((len(lambdas), X.shape[1]))

for i, lamb in enumerate(lambdas):
    lasso = Lasso(alpha=lamb, fit_intercept=True, max_iter=10000)
    lasso.fit(X, y)
    empty[i, :] = lasso.coef_

# Plot the coefficients
plt.figure(figsize=(12, 6))
for i in range(X.shape[1]):
    plt.plot(lambdas, empty[:, i], label=X.columns[i])
plt.xscale("log")
plt.xlabel("Lambda")
plt.ylabel("Coefficient")
plt.title("Lasso Coefficients vs. Lambda")
plt.legend(loc = "upper right")
plt.show()



# Select the relevant features based on Lasso regression
X = X[["dp_lag1", "target_lag2", "dp_lag2", "ep", "tbl", "ep_lag1"]]
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size = 0.8)

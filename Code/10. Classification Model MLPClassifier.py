from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier



Z["target_direction"] = (Z["target"] > 0).astype(int)


X_cls = Z.drop(columns=["target", "target_direction"])
y_cls = Z["target_direction"]

# Train-Test-Split 
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, train_size=0.8)






# Grid Search 
param_grid_cls = {
    "hidden_layer_sizes": [(100, 100), (200, 200)],
    "activation": ["relu", "tanh"]
}

mlp_clf = GridSearchCV(MLPClassifier(max_iter=1000, solver="adam"), param_grid_cls, cv=5, verbose=2, n_jobs=-1)
mlp_clf.fit(X_train_cls, y_train_cls)

# Best Parameter
print("Best Parameter for MLP-Classification:", mlp_clf.best_params_)

#Predictions
y_pred_cls = mlp_clf.predict(X_test_cls)



# Accuracy
accuracy = accuracy_score(y_test_cls, y_pred_cls)
print(f"Accuracy (MLP Classifier): {accuracy:.4f}")

# Confusion Matrix
conf_mat = confusion_matrix(y_test_cls, y_pred_cls)
print("Confusion Matrix:\n", conf_mat)


print("\nClassification Report:\n", classification_report(y_test_cls, y_pred_cls))

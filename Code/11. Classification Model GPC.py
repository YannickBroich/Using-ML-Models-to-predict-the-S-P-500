

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ConstantKernel as C

# Define Kernel
kernel_options = [
    C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)),
    1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1)
]

param_grid_gpc = {"kernel": kernel_options}

gpc = GridSearchCV(GaussianProcessClassifier(), param_grid_gpc, cv=5, verbose=2, n_jobs=-1)
gpc.fit(X_train_cls, y_train_cls)

# Beste Parameter 
print("Best Parameter:", gpc.best_params_)

# Prediction
y_pred_gpc = gpc.predict(X_test_cls)



# Accuracy
accuracy_gpc = accuracy_score(y_test_cls, y_pred_gpc)
print(f"Accuracy (Gaussian Process Classifier): {accuracy_gpc:.4f}")

# Confusion Matrix
conf_mat_gpc = confusion_matrix(y_test_cls, y_pred_gpc)
print("Confusion Matrix (GPC):\n", conf_mat_gpc)


print("\nClassification Report (GPC):\n", classification_report(y_test_cls, y_pred_gpc))

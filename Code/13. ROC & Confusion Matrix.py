from sklearn.metrics import roc_curve, auc
import seaborn as sns


y_prob_mlp = mlp_clf.predict_proba(X_test_cls)[:, 1][timesteps:]
y_prob_gpc = gpc.predict_proba(X_test_cls)[:, 1][timesteps:]
y_prob_lstm = y_pred_lstm_cls  # already probability

# ROC Curve
fpr_mlp, tpr_mlp, _ = roc_curve(y_val_lstm_cls, y_prob_mlp)
fpr_gpc, tpr_gpc, _ = roc_curve(y_val_lstm_cls, y_prob_gpc)
fpr_lstm, tpr_lstm, _ = roc_curve(y_val_lstm_cls, y_prob_lstm)

# AUC-Values
auc_mlp = auc(fpr_mlp, tpr_mlp)
auc_gpc = auc(fpr_gpc, tpr_gpc)
auc_lstm = auc(fpr_lstm, tpr_lstm)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(fpr_mlp, tpr_mlp, label=f"MLP (AUC = {auc_mlp:.2f})")
plt.plot(fpr_gpc, tpr_gpc, label=f"GPC (AUC = {auc_gpc:.2f})")
plt.plot(fpr_lstm, tpr_lstm, label=f"LSTM (AUC = {auc_lstm:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()


# Confusion Matrices
mat_mlp = confusion_matrix(y_val_lstm_cls, y_pred_cls[timesteps:])
mat_gpc = confusion_matrix(y_val_lstm_cls, y_pred_gpc[timesteps:])
mat_lstm = confusion_matrix(y_val_lstm_cls, y_pred_lstm_cls_binary)

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(mat_mlp, annot=True, fmt="d", ax=axs[0], cmap="Blues")
axs[0].set_title("MLP")
axs[0].set_xlabel("Predicted")
axs[0].set_ylabel("Actual")

sns.heatmap(mat_gpc, annot=True, fmt="d", ax=axs[1], cmap="Greens")
axs[1].set_title("GPC")
axs[1].set_xlabel("Predicted")
axs[1].set_ylabel("Actual")

sns.heatmap(mat_lstm, annot=True, fmt="d", ax=axs[2], cmap="Purples")
axs[2].set_title("LSTM")
axs[2].set_xlabel("Predicted")
axs[2].set_ylabel("Actual")

plt.tight_layout()
plt.show()

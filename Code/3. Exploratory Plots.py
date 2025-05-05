


# Plot of dp
plt.figure(figsize=(12, 6))
plt.plot(Z["dp"], label="dp", color="blue")
plt.plot(Z["ep"], label="ep", color="orange")
plt.plot(Z["tbl"], label="tbl", color="green")
plt.plot(Z["target"], label="target", color="red")
plt.legend()
plt.title("Plotted Variables")

# Create lagged features

# dp lagged
Z["dp_lag1"] = Z["dp"].shift(1)
Z["dp_lag2"] = Z["dp"].shift(2)

# ep lagged
Z["ep_lag1"] = Z["ep"].shift(1)
Z["ep_lag2"] = Z["ep"].shift(2)

# tbl lagged
Z["tbl_lag1"] = Z["tbl"].shift(1)
Z["tbl_lag2"] = Z["tbl"].shift(2)

# target lagged
Z["target_lag1"] = Z["target"].shift(1)
Z["target_lag2"] = Z["target"].shift(2)

# Concatenate the lagged features with the original DataFrame
Z = Z[["dp", "ep", "tbl", "target", "dp_lag1", "dp_lag2", "ep_lag1", "ep_lag2", "tbl_lag1", "tbl_lag2", "target_lag1", "target_lag2"]]
Z = Z.dropna(axis=0, how="any")
print(Z.head())

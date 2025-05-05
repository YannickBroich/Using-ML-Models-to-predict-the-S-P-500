# -*- coding: utf-8 -*-
"""
Created on Mon May  5 22:09:56 2025

@author: Yannick
"""



# Load the data

data = pd.read_csv('DatasetQ3ML.csv')
# Convert the date column to datetime format and select the relevant rows
data["yyyymm_str"] = data["yyyymm"].astype(str).str.zfill(6)
data["Date"] = pd.to_datetime(data["yyyymm_str"], format = "%Y%m")

data = data.drop(columns=["yyyymm", "yyyymm_str"])

data = data[(data["Date"] >= "1927-01-01") & (data["Date"] <= "2021-12-31")]
print(data.head())


# Change datatype of Index column
data["Index"] = data["Index"].replace(",", "", regex=True)
data["Index"] = data["Index"].astype(float)


# Calculate log returns and equity premium

logreturns = np.log(data["Index"]).diff()
IndexDiv = data["Index"] + data["D12"]
logreturnsdiv = np.log(IndexDiv).diff()
logRiskFree = np.log(data["Rfree"] + 1)
logEquityPremium =  logreturnsdiv - logRiskFree
logEquityPremium.name = "Log Returns Premium"

print(logreturns.head())


# Select the relevant columns: dp, ep, and tbl, logreturns

dp = np.log(data["D12"]) - np.log(data["Index"])
dp.name = "dp"
dp = pd.Series(dp.values, index=data["Date"])

ep = np.log(data["E12"] / data["Index"])
ep.name = "ep"
ep = pd.Series(ep.values, index=data["Date"])

tbl = data["tbl"]
tbl.name = "tbl"
tbl = pd.Series(tbl.values, index=data["Date"])

# Create a DataFrame with the selected columns
Z = pd.DataFrame({"dp": dp, "ep": ep, "tbl": tbl, "target": pd.Series(logreturns.values, index=data["Date"])})
Z = Z.dropna(axis = 0, how = "any")
print(Z)
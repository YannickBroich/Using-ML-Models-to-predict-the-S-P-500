# S&P 500 Prediction Using Machine Learning

This repository contains a comprehensive machine learning pipeline for predicting the **S&P 500 index returns**, **log-price**, and **market direction** using macroeconomic indicators. It was developed as part of a Data Science & Machine Learning course project.

---

##  Project Goals

The main objectives are to:

- Predict **log returns** of the S&P 500.
- Forecast the **log price** of the index.
- Classify the **direction** of the market (up/down).
- Compare the performance of several models across tasks.



##  Data Overview

- **Source:** Provided CSV file (`DatasetQ3ML.csv`) with macro-financial indicators.
- **Timeframe:** January 1927 – December 2021
- **Features Used:**
  - `dp`: Dividend-Price Ratio  
  - `ep`: Earnings-Price Ratio  
  - `tbl`: 1-month Treasury Bill Rate
- **Targets:**
  - Log returns (`target`)
  - Log price (`target_price`)
  - Direction (binary: up = 1, down = 0)



##  Models Implemented

The following models were trained and evaluated:

###  Regression (Log Returns & Log Price)
- **Multi-Layer Perceptron (MLP)**
- **Gaussian Process Regression (GPR)**
- **Long Short-Term Memory Networks (LSTM)**
- **ARMA-GARCH hybrid models**

###  Classification (Market Direction)
- **MLP Classifier**
- **Gaussian Process Classifier (GPC)**
- **LSTM (binary classification)**




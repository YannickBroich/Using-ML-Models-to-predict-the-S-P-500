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



# Results Summary

This document summarizes the findings of a machine learning forecasting study on the S&P 500 index using monthly data from 1927 to 2021.

## Forecasting Targets

We investigated three key financial prediction tasks:
- Log returns (equity premium)
- Log price (index level)
- Market direction (binary classification: up or down)

## Models Applied

The following models were implemented and tuned:
- Multi-Layer Perceptron (MLP)
- Long Short-Term Memory networks (LSTM)
- Gaussian Process Regression / Classification (GPR / GPC)
- ARMA-GARCH (baseline model for comparison)

## Feature Selection

Lasso regression was used for variable selection. Lagged values of the predictors `dp`, `ep`, `tbl` and the target variable itself were included in the final models.

## Results Summary

### Log Returns Forecasting

| Model        | MSE     | MAE     | Correlation |
|--------------|---------|---------|-------------|
| MLP          | 0.00347 | 0.03946 | 0.692       |
| GPR          | 0.00088 | 0.01848 | 0.874       |
| LSTM         | 0.00395 | 0.04740 | 0.050       |

Gaussian Process Regression achieved the best performance, likely due to its ability to capture non-linear structures.

### Market Direction Classification

| Model | Accuracy | F1-Score | AUC  |
|-------|----------|----------|------|
| MLP   | 0.90     | 0.91     | 0.96 |
| GPC   | 0.92     | 0.93     | 1.00 |
| LSTM  | 0.56     | 0.72     | 0.48 |

The Gaussian Process Classifier performed best, achieving perfect separation (AUC = 1.0). LSTM underperformed in this task.

### Log Price Forecasting

| Model | MSE    | MAE    | Correlation |
|-------|--------|--------|-------------|
| MLP   | 0.0427 | 0.1574 | 0.987       |
| GPR   | 0.0166 | 0.1035 | 0.995       |
| LSTM  | 4.1092 | 1.7506 | 0.676       |

Predicting the log price proved easier due to its smoother structure. Again, GPR outperformed all other models.

### Comparison to ARMA-GARCH Baseline

| Model        | MSE     | MAE     | Correlation |
|--------------|---------|---------|-------------|
| MLP          | 0.00346 | 0.03904 | 0.6909      |
| GPR          | 0.00088 | 0.01816 | 0.8752      |
| LSTM         | 0.00395 | 0.04736 | 0.1347      |
| ARMA-GARCH   | 0.00376 | 0.04119 | -0.089      |

The ARMA-GARCH model underperformed compared to the machine learning models, especially in terms of correlation and error metrics.

## Interpretation

- GPR and GPC consistently performed best across all tasks, indicating that kernel-based methods are highly effective in capturing non-linear relationships in macro-financial time series.
- MLP models delivered stable and competitive results, particularly in classification.
- LSTM struggled in both regression and classification tasks, likely due to the short lag structure and low signal-to-noise ratio in monthly returns.
- Traditional econometric models like ARMA-GARCH fell short, reinforcing the strength of machine learning approaches for predictive tasks.

###  Requirements:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- arch
- tensorflow
- tqdm



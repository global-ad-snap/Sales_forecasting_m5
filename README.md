# Sales Forecasting — Retail Demand (M5 Dataset)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-Data%20Analysis-brightgreen.svg)](https://pandas.pydata.org/)
[![statsmodels](https://img.shields.io/badge/statsmodels-Time%20Series-orange.svg)](https://www.statsmodels.org/)
[![Prophet](https://img.shields.io/badge/Prophet-Forecasting-lightgrey.svg)](https://facebook.github.io/prophet/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


## Project Overview

This project demonstrates an end-to-end **retail sales forecasting pipeline** using real-world data from the **M5 Forecasting Competition (Walmart)**. The objective is to forecast future sales to support **inventory planning, staffing, and revenue management**, while comparing statistical and machine-learning time series models in a transparent, defensible way.

The project is designed as a **client-ready portfolio case**, emphasizing reproducibility, proper benchmarking, and business-relevant interpretation.

---

## Business Objective

Retail organizations require accurate short- and medium-term sales forecasts to:

* Optimize inventory levels
* Reduce stock-outs and overstocking
* Plan workforce and promotions
* Support revenue forecasting

This project answers the question:

> *How accurately can weekly store-level sales be forecasted using progressively more advanced time-series models?*

---

## Dataset

**Source:** M5 Forecasting – Accuracy (Kaggle)

**Description:**

* Daily unit sales from Walmart stores across the USA
* Hierarchical structure: state → store → category → item
* Time span: ~5 years

**Files used:**

* `sales_train_validation.csv`
* `calendar.csv`
* `sell_prices.csv` (not used in baseline models)

**Forecasting scope:**

* Store-level aggregation
* Weekly sales frequency
* Example store analyzed: `CA_1`

*Raw data handling and reproduction instructions are provided in the Dataset Access section below.*

---

## Project Structure

```
Sales_forecasting_m5/
├── data/
│   ├── raw/                 # Original M5 CSV files (not tracked)
│   └── processed/           # Cleaned, aggregated datasets
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_trend_seasonality_analysis.ipynb
│   ├── 03_sarima_baseline.ipynb
│   ├── 04_prophet_model.ipynb
│   └── 05_lstm_model.ipynb
├── src/                     # Reusable pipeline modules
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── evaluation.py
│   └── plotting.py
├── visuals/
│   ├── weekly_sales_trend.png
│   ├── seasonal_decomposition.png
│   ├── sarima_forecast.png
│   ├── prophet_forecast_CA_1.png
│   ├── prophet_components_CA_1.png
│   └── lstm_forecast_CA_1.png
├── .gitignore
├── report                   # Retail_Sales_Forecasting (PDF)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Dataset Access

Due to GitHub file size limits and Kaggle licensing restrictions, the full raw M5 dataset
is not included in this repository.

### Data Source
- M5 Forecasting – Accuracy Dataset (Kaggle)

### Raw Files (Not Tracked)
- calendar.csv
- sales_train_validation.csv
- sales_train_evaluation.csv
- sell_prices.csv

### How to Reproduce
1. Download the dataset from Kaggle
2. Place raw files into:
   data/raw/
3. Run:
   notebooks/01_data_preprocessing.ipynb

---

## Modeling Approach

Models were implemented in **increasing order of complexity**, following best practices in forecasting.

### 1. Trend & Seasonality Analysis

* Daily data aggregated to weekly frequency
* Visual inspection of trends
* Statistical seasonal decomposition (52-week period)

**Purpose:**
To confirm the presence of trend and annual seasonality before model selection.

---

### 2. SARIMA — Statistical Baseline

**Model:** Seasonal ARIMA (SARIMA)

* Order: (1, 1, 1)
* Seasonal order: (1, 1, 1, 52)
* Train/test split: 80% / 20% (time-aware)

**Why SARIMA:**

* Explicitly models trend and seasonality
* Standard baseline in demand forecasting literature
* Provides a statistical benchmark

**Important clarification:**
SARIMA was used **as a forecasting benchmark**, not for coefficient interpretation. Some parameters exhibit numerical instability, which is a known issue with long seasonal periods and does not invalidate its role as a baseline.

**Evaluation metrics:**

* RMSE
* MAE

---

### 3. Prophet — Business-Oriented Forecasting

**Model:** Facebook / Meta Prophet

* Additive trend and seasonality
* Automatic changepoint detection
* Robust to missing data and outliers

**Why Prophet:**

* Designed for retail and business time series
* Strong interpretability of trend and seasonal components
* Commonly used in industry settings

---

### 4. LSTM — Deep Learning Model

**Model:** Long Short-Term Memory (LSTM) neural network

* Weekly sales transformed into supervised sequences
* Scaled inputs
* Single or multi-layer recurrent architecture (depending on experiment)

**Why LSTM:**

* Captures non-linear temporal dependencies
* Serves as a high-capacity model for comparison

---

## Evaluation Strategy

All models were evaluated using:

* **Out-of-sample test set**
* **Identical forecast horizon**
* **Same error metrics (RMSE, MAE)**

This ensures a **fair and reproducible comparison**.

---

## Model Performance Comparison

| Model   | RMSE    | MAE     | Interpretation                                              |
|--------|---------|---------|--------------------------------------------------------------|
| SARIMA | 1785.78 | 1431.01 | Statistical baseline capturing trend and annual seasonality |
| Prophet | 1506.02 | 1185.88 | Business-oriented model with adaptive trend and seasonality |
| LSTM   | 1831.50 | 1442.17 | Nonlinear model capturing complex temporal patterns         |

*Metrics correspond to weekly out-of-sample forecasts for Store CA_1 using identical train–test splits.*

---

## Key Insights (Example)

* Weekly sales show strong annual seasonality
* SARIMA provides a reasonable statistical baseline
* Prophet improves trend adaptability and forecast stability
* LSTM captures complex patterns but requires careful tuning and sufficient data

*(Exact numerical results depend on store and training window.)*

---

## Key Visualizations

### Weekly Sales Trend (Store CA_1)
![Weekly Sales Trend](visuals/weekly_sales_trend.png)

### Seasonal Decomposition
![Seasonal Decomposition](visuals/seasonal_decomposition.png)

### SARIMA Forecast vs Actual
![SARIMA Forecast](visuals/sarima_forecast.png)

---

## Business Implications

Accurate weekly forecasts can support:

* Inventory replenishment planning
* Promotion scheduling
* Workforce allocation
* Revenue projections

The comparison demonstrates how model choice impacts forecast reliability and operational decisions.

---

## Business Impact

### Potential Value Drivers

This project is designed to support measurable operational or financial impact, including:

- Improved decision accuracy
- Operational efficiency gains
- Risk reduction
- Resource optimization
- Revenue protection or growth

### Example Deployment Benefits

Actual impact depends on deployment context, data quality, and operational integration. Potential benefits may include:

- Reduced operational costs through earlier risk identification
- Improved allocation of staff, inventory, or marketing resources
- Enhanced decision support for clinical or business stakeholders
- Increased transparency and confidence in analytics-driven decisions

### Measurement Considerations

Typical ROI evaluation would include:

- Baseline vs post-deployment performance comparison
- Cost savings analysis
- Revenue uplift measurement
- Error reduction metrics
- Operational efficiency indicators

Formal ROI validation requires real-world deployment data.

---

## Reproducibility

* Python ≥ 3.9
* All dependencies listed in `requirements.txt`
* Random seeds fixed where applicable

---

## Disclaimer

This project is for **educational and portfolio demonstration purposes only**. It does not represent production-ready forecasts or official Walmart analyses.

---
## Author

**Medical AI & Healthcare Data Science Consultant**

Physician (MBBS) with a Master’s in Global Communication and professional training in Machine Learning, Deep Learning, NLP, and AI for Medicine. Experienced in building interpretable risk models and decision-support systems for regulated, data-sensitive environments.

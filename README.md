# 🚗 BMW Used Car Price Analysis & Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-2.0-150458?logo=pandas)
![License](https://img.shields.io/badge/license-MIT-green)

A full end-to-end data science project that explores a BMW used car listing dataset, uncovers pricing patterns through exploratory data analysis, and builds a machine learning model to predict vehicle prices with **R² ≈ 0.88**.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)

---

## Project Overview

This project answers two core questions:

1. **What factors most strongly influence the resale price of a BMW?**
2. **Can we predict the price of a used BMW accurately given its attributes?**

We achieve this through rigorous EDA (distributions, correlations, outlier analysis) followed by a Random Forest regression pipeline that reaches an R² of ~0.88 on held-out test data.

---

## Dataset

| Column | Type | Description |
|---|---|---|
| `model` | Categorical | BMW model series (e.g. 1 Series, 5 Series, X5) |
| `year` | Integer | Year of manufacture (1996–2020) |
| `price` | Integer | Listed price in GBP |
| `transmission` | Categorical | Automatic / Manual / Semi-Auto |
| `mileage` | Integer | Odometer reading in miles |
| `fuelType` | Categorical | Diesel / Petrol / Hybrid / Electric / Other |
| `tax` | Integer | Annual road tax in GBP |
| `mpg` | Float | Fuel efficiency (miles per gallon) |
| `engineSize` | Float | Engine displacement in litres |

**Size:** 10,781 rows × 9 columns — no missing values.

---

## Project Structure

```
bmw-analysis/
├── data/
│   └── bmw.csv                  # Raw dataset
├── src/
│   └── eda_modeling.py          # Full EDA + ML pipeline
├── outputs/                     # Generated charts & metrics (auto-created)
├── requirements.txt
└── README.md
```
---

## Key Findings

- **Mileage & Year** are the two strongest price predictors — each additional year of age costs roughly £1,200; each 10,000 extra miles costs ~£800.
- **Engine size** has a strong positive correlation with price (larger engines command premium prices).
- **Diesel** listings account for ~57 % of the dataset, with slightly higher average mileage than petrol equivalents.
- **Electric & Hybrid** listings fetch significantly higher prices on average, reflecting newer model years.
- The **8 Series** is the highest average-priced model; the **1 Series** has the most listings.

---

## Model Performance

| Metric | Value |
|---|---|
| R² (test set) | **~0.88** |
| RMSE | **~£3,900** |
| MAE | **~£2,600** |

Top predictive features (by importance): `year` > `mileage` > `engineSize` > `model` > `fuelType`

---

## Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading, wrangling |
| `numpy` | Numerical operations |
| `matplotlib` | Base visualisation |
| `seaborn` | Statistical plots |
| `scikit-learn` | ML pipeline, model training, evaluation |

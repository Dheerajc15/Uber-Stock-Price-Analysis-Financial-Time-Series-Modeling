# Statistical Hypothesis Testing & Time-Series Analysis of Uber Stock Prices

## 🔍 Project Overview

### 🧾 Data Acquisition & Preprocessing

* Fetched daily Uber stock data from Yahoo Finance (May 10, 2019 – Feb 5, 2025)
* Cleaned data using:

  * Date formatting and parsing
  * Handling missing values and duplicates
  * Outlier detection using Interquartile Range (IQR)

### 📊 Exploratory Data Analysis (EDA)

* Time-series visualizations of stock trends
* Distribution plots (histograms, KDEs, box plots)
* Heatmaps for feature correlations
* Computed summary statistics, skewness, and kurtosis

### 📈 Statistical Inference

* **Normality Test**: Shapiro–Wilk test
* **Hypothesis Testing**:

  * Independent t-tests (e.g., weekday vs. weekend closing prices)
  * Chi-square tests (e.g., price vs. volume category relationships)
  * One-way and Two-way ANOVA (e.g., effect of market period on prices)

### 🤖 Regression Modeling

* Built **OLS regression models** using StatsModels:

  * **Simple Linear Regression** (e.g., Close \~ Open)
  * **Multiple Linear Regression** (e.g., Close \~ Open + High + Low + Volume)
* Evaluated models using:

  * R² Score
  * Mean Squared Error (MSE)
  * Root Mean Squared Error (RMSE)
  * Mean Absolute Error (MAE)
* Visualized residuals and predicted vs. actual price values

---

## 📦 Technologies Used

* Python
* Jupyter Notebook
* Pandas, NumPy
* Matplotlib, Seaborn
* SciPy, StatsModels

---

```


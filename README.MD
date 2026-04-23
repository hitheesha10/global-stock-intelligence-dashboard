# 📊 Global Stock Market Intelligence Dashboard

## 🚀 Overview

This project is an **end-to-end financial analytics system** built using Python, designed to analyze and visualize global stock market data across multiple companies, industries, and countries.

It transforms raw stock price data into **actionable insights** through data cleaning, feature engineering, and interactive visualizations.

The final output is an **interactive dashboard built with Streamlit**, enabling users to explore trends, volatility, correlations, and market behavior dynamically.

---

## 🎯 Problem Statement

Financial datasets are complex and high-dimensional. The goal of this project is to:

* Analyze stock price movements across multiple global companies
* Identify trends, volatility, and performance differences
* Understand relationships between stocks and industries
* Build an interactive tool for real-time exploration of insights

---

## 📂 Dataset Description

The dataset contains daily stock trading data with the following fields:

* **Date** – Trading date
* **Open / High / Low / Close** – Stock price metrics
* **Volume** – Number of shares traded
* **Brand_Name** – Company name
* **Ticker** – Stock symbol
* **Industry_Tag** – Industry classification
* **Country** – Company’s origin

---

## 🛠️ Tech Stack

* **Python**
* **Pandas** – Data manipulation & analysis
* **NumPy** – Numerical computations
* **Plotly** – Interactive visualizations
* **Streamlit** – Dashboard development

---

## 🔄 Project Workflow

### 1. Data Cleaning

* Converted `Date` to datetime format (handled timezone inconsistencies)
* Removed invalid and missing values
* Sorted data chronologically
* Optimized categorical columns

### 2. Feature Engineering

* **Daily Returns** – Percentage change in closing price
* **Rolling Volatility** – 30-day standard deviation
* **Moving Average (MA-50)** – Trend smoothing
* **Year / Month extraction** for time-based analysis

### 3. Exploratory Data Analysis (EDA)

* Time-series price trends
* Return distributions
* Volume analysis
* Industry-level comparisons

### 4. Advanced Analytics

* **Correlation Matrix** – Relationships between stocks
* **Sharpe Ratio** – Risk-adjusted performance
* **Top Gainers & Losers** – Market movers
* **Volatility Tracking** – Risk patterns

---

## 📊 Dashboard Features

The interactive dashboard includes:

* 📈 **Stock Price Trends**
* 📉 **Moving Average Analysis**
* 🕯️ **Candlestick Charts**
* 📊 **Return Distribution**
* ⚠️ **Rolling Volatility**
* 🔗 **Correlation Heatmap**
* 📊 **Sharpe Ratio Rankings**
* 🚀 **Top Gainers & Losers**
* 🧠 **Automated Market Insights**

---

## 🧠 Key Insights

* Identified **high-volatility industries**, indicating higher risk exposure
* Highlighted **top-performing stocks using risk-adjusted metrics**
* Revealed **correlation patterns** useful for portfolio diversification
* Detected **market momentum trends** through rolling analysis

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd stock-analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
python -m streamlit run app.py
```

### 4. Open in browser

```
http://localhost:8501
```

---

## 📁 Project Structure

```
stock-analysis/
│── app.py
│── requirements.txt
│── README.md
│── data/
│    └── World_Stock_Prices_Dataset.csv
│── notebooks/
```

---

## 💡 Challenges & Solutions

**Challenge:** Mixed timezone issues in date column
**Solution:** Used `pd.to_datetime(..., utc=True)` and normalized timezone

**Challenge:** Duplicate entries for Date–Ticker combinations
**Solution:** Used `pivot_table()` instead of `pivot()`

**Challenge:** Missing values in return calculations
**Solution:** Handled using group-wise transformations and filtering

---

## 🚀 Future Enhancements

* Portfolio optimization (efficient frontier)
* Machine learning for stock prediction
* Real-time data integration via APIs
* Deployment on cloud platforms

---

## 📌 Conclusion

This project demonstrates the ability to:

* Handle real-world messy data
* Perform advanced financial analysis
* Build interactive dashboards
* Translate data into actionable insights

It bridges the gap between **data analysis and product development**, showcasing practical, industry-relevant skills.

---

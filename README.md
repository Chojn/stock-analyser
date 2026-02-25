# Stock Analyser & Portfolio Optimisation Tool

## Overview

This project is a comprehensive stock analysis and portfolio optimisation tool built in Python. It combines:

* Technical Indicators (RSI, MACD, Bollinger Bands, P/E, Beta)
* AI-Generated Executive Summaries (via OpenAI)
* Modern Portfolio Theory (Maximum Sharpe Optimisation)
* Black-Litterman Portfolio Model
* Data retrieval from Yahoo Finance

It allows users to analyse multiple financial assets, generate key performance indicators (KPIs), and compute optimised portfolio allocations.

## Architecture Overview

The project is divided into **three major components**:
* Technical KPI Analysis
* Modern Portfolio Theory (MPT) Optimisation
* Black-Litterman Model Optimisation
* Optional LLM-based Executive Summary

## 1. Technical KPI Analysis

The following financial indicators are implemented:

### Relative Strength Index (RSI)
* Momentum oscillator (0–100 scale)
* Above 70 → Overbought
* Below 30 → Oversold
* Default lookback window: 14 days

Functions:
* `plot_rsi()`
* `display_rsi()`
* Included in `calculate_kpis()`

### Bollinger Bands

* 20-day SMA (middle band)
* ±2 standard deviations
* Identifies volatility & breakout conditions

Functions:
* `plot_bollinger_bands()`
* `display_bollinger_bands()`
* Included in `calculate_kpis()`

### MACD (Moving Average Convergence Divergence)

* 12-day EMA
* 26-day EMA
* 9-day Signal Line
* Used to detect momentum shifts

Functions:
* `plot_macd()`
* `display_macd()`
* Included in `calculate_kpis()`

### P/E Ratio

* Uses trailing EPS from Yahoo Finance
* Identifies relative valuation
* Skips if EPS is unavailable

Functions:
* `plot_pe_ratios()`
* `display_pe_ratios()`
* Included in calculate_kpis()

### Beta Comparison

* Measures volatility relative to market
* Beta > 1 → More volatile
* Beta < 1 → Less volatile

Function:
* `plot_beta_comparison()`

## 2. Modern Portfolio Theory (MPT)

The code implements:

### Portfolio Performance

* Expected Returns
* Portfolio Volatility

### Sharpe Ratio Optimization
* Maximizes risk-adjusted returns
* Uses scipy.optimize.minimize
* Constraints:
  * Weights sum to 1
  * No short-selling (0 ≤ weight ≤ 1)

Key Functions:
* `portfolio_performance()`
* `negative_sharpe_ratio()`
* max_sharpe_ratio()

Sample Output:

```
{'AAPL': 0.18, 'AMZN': 0.12, ...}
```

## 3. Black-Litterman Model

Enhances traditional MPT by incorporating investor views.

### In This Implementation:
* Market implied prior returns
* Manual market caps


### Workflow:

* Calculate market implied risk aversion
* Compute equilibrium returns
* Inject investor views (Q and P matrices)
* Generate adjusted returns
* Optimise Sharpe ratio

### Libraries Used:
* PyPortfolioOpt
* BlackLittermanModel
* EfficientFrontier

Sample Output:
```
{'AAPL': 0.15, 'MSFT': 0.27, ...}
```

## 4. AI-Generated Executive Summary

Function:
* `display_stocks_report(kpi_data)`

It sends KPI data to the LLM and generates:

* Executive Summary
* Buy/Sell/Hold recommendations

## Installation

### Requirements
pip install pandas numpy matplotlib yfinance python-dateutil scipy rich
pip install pypfopt langchain-openai

## How To Run

1. Insert your OpenAI API key:
```
api_key = "Insert your api key here"
```

2. Modify:
```
years = 2
assets = [...]
risk_free_rate = 0.04
```

3. Run:
```
python stocks.py
```

Example Assets Included:

* Apple (AAPL)
* Amazon (AMZN)
* Bitcoin (BTC-USD)
* Alphabet (GOOGL)
* Meta (META)
* Microsoft (MSFT)
* Nvidia (NVDA)
* SPY (S&P 500 ETF)
* Tesla (TSLA)

## Important Notes

* Internet connection required (Yahoo Finance API)
* OpenAI API key required for LLM summaries
* No short-selling allowed in optimisation
* SPY market cap manually set

## Potential Improvements
* Add Monte Carlo simulation
* Add Value-at-Risk (VaR)
* Add drawdown analysis
* Add risk parity portfolio
* Add interactive dashboard (Streamlit)
* Add backtesting framework

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
import yfinance as yf
from langchain_openai import ChatOpenAI
from scipy.optimize import minimize
from pypfopt import risk_models, expected_returns, BlackLittermanModel, EfficientFrontier, black_litterman
from rich import print as rprint

def initialise_llm(api_key):
    """
    Function to initialise the OpenAI LLM model

    Args:
        api_key (str): The api key linked to your personal/business OpenAI account
    
    Returns:
        ChatOpenAI: The initialised language model
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)
    return llm

def calculate_date_range(years):
    """
    Function that calculates the start and end dates based on the number of years set

    Args:
        years (int): The number of years that the range will be set
    
    Returns:
        The start date and the end date between the years set
    """
    end_date = datetime.today()
    start_date = end_date - relativedelta(years=years)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def get_llm_response(llm, prompt):
    """
    Get the response from the language model for a given prompt.

    Args:
        llm (ChatOpenAI): The initialized language model.
        prompt (str): The prompt to send to the language model.
    """
    response = llm.invoke(prompt)
    rprint(response.content)

# KPI section
def plot_rsi(data, ticker):
    """
    Plot the Relative Strength Index (RSI) for a given stock.

    Args:
        data (DataFrame): The stock data.
        ticker (str): The stock ticker symbol.

    Returns:
        None

    Notes:
        RSI is a momentum oscillator that measures the speed and change of price movements. 
        It ranges from 0 to 100, with values above 70 indicating overbought conditions and below 
        30 indicating oversold conditions.
    """
    # Define the lookback window for RSI calculation
    window = 14

    # Calculate the RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Plot the RSI
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, rsi, label='RSI', color='purple')
    plt.axhline(70, color='red', linestyle='--')        # Add a horizontal line at RSI = 70 (overbought threshold)
    plt.axhline(30, color='green', linestyle='--')      # Add a horizontal line at RSI = 30 (oversold threshold)

    # Add title, labels, and legend
    plt.title(f'RSI of {ticker}')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.show()

def display_rsi(tickers, start_date, end_date):
    """
    Displays historical data for each ticker and plot its RSI

    Args:
        tickers (lst): List of all ticker symbols from the assets
        start_date (str): The starting date
        end_date (str): The ending date
    """
    for ticker in tickers:
        data_ticker = yf.download(ticker, start=start_date, end=end_date)
        plot_rsi(data_ticker, ticker)

def plot_bollinger_bands(data, ticker):
    """
    Plot the Bollinger Bands for a given stock.

    Args:
        data (DataFrame): The stock data.
        ticker (str): The stock ticker symbol.

    Returns:
        None

    Notes:
        Bollinger Bands consist of a middle band (SMA) and two outer bands (standard deviations away from the SMA). 
        They help identify volatility and potential overbought or oversold conditions.
    """
    # Define the rolling window period for Bollinger Bands
    window = 20

    # Calculate the middle band (simple moving average) and outer bands (±2 standard deviations)
    data['Middle Band'] = data['Close'].rolling(window=window).mean()
    data['Upper Band'] = data['Middle Band'] + 2 * data['Close'].rolling(window=window).std()  # Upper band
    data['Lower Band'] = data['Middle Band'] - 2 * data['Close'].rolling(window=window).std()  # Lower band

    # Plot the closing price and Bollinger Bands
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['Close'], label='Closing Price')  # Plot closing price
    plt.plot(data.index, data['Middle Band'], label='Middle Band', color='blue')  # Plot middle band
    plt.plot(data.index, data['Upper Band'], label='Upper Band', color='red')  # Plot upper band
    plt.plot(data.index, data['Lower Band'], label='Lower Band', color='green')  # Plot lower band

    # Add title, labels, and legend
    plt.title(f'Bollinger Bands of {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def display_bollinger_bands(tickers, start_date, end_date):
    """
    Displays historical data for each ticker and plot its Bollinger Bands

    Args:
        tickers (lst): List of all ticker symbols from the assets
        start_date (str): The starting date
        end_date (str): The ending date
    """
    for ticker in tickers:
        data_ticker = yf.download(ticker, start=start_date, end=end_date)
        plot_bollinger_bands(data_ticker, ticker)

def plot_pe_ratios(data, ticker, eps):
    """
    Plot the Price-to-Earnings (P/E) ratio for a given stock.

    Args:
        data (DataFrame): The stock data.
        ticker (str): The stock ticker symbol.
        eps (float): The earnings per share of the stock.

    Returns:
        None

    Notes:
        The P/E ratio measures a company's current share price relative to its per-share earnings. 
        A high P/E might indicate overvaluation, while a low P/E might suggest undervaluation.
    """
    # Check if EPS value is valid
    if eps is None or eps == 0:
        print(f"Warning: EPS for {ticker} is not available or zero. PE ratio can't be calculated.")
        return

    # Calculate the P/E ratio
    pe_ratio = data['Close'] / eps

    # Create and customize the plot
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, pe_ratio, label=f'{ticker} PE Ratio')
    plt.title('PE Ratios of Selected Stocks')
    plt.xlabel('Date')
    plt.ylabel('PE Ratio')
    plt.legend()
    plt.show()

def display_pe_ratios(tickers, start_date, end_date):
    """
    Displays historical data for each ticker and plot its P/E ratios

    Args:
        tickers (lst): List of all ticker symbols from the assets
        start_date (str): The starting date
        end_date (str): The ending date
    """
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        eps = stock.info.get('trailingEps') # Retrieve the trailing EPS value from the stock's info
        data_ticker = yf.download(ticker, start=start_date, end=end_date)
        plot_pe_ratios(data_ticker, ticker, eps)

def plot_beta_comparison(tickers, start_date, end_date):
    """
    Plots a bar chart comparing the beta values of selected stocks.

    Args:
        tickers (list of str): List of stock ticker symbols.
        start_date (str): Start date for historical data retrieval (YYYY-MM-DD format).
        end_date (str): End date for historical data retrieval (YYYY-MM-DD format).

    Notes:
        Beta measures a stock's volatility relative to the market.
        A beta > 1 means the stock is more volatile than the market, while < 1 means less volatile.
    """
    betas = {}

    for ticker in tickers:
        stock = yf.Ticker(ticker)
        try:
            # Retrieve historical data
            data = stock.history(start=start_date, end=end_date)
            # Get the beta value from stock's info
            beta = stock.info.get('beta')

            # Skip to the next ticker if beta is not available
            if beta is None:
                print(f"Warning: Beta for {ticker} is not available.")
                continue

            betas[ticker] = beta # Store the beta value

        # Handle errors related to missing data
        except KeyError as e:
            print(f"Error retrieving data for {ticker}: {e}")
        # Handle other unexpected errors
        except Exception as e:
            print(f"An error occurred with ticker {ticker}: {e}")

    # Plotting the bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(betas.keys(), betas.values(), color='blue')
    plt.title('Beta Comparison of Selected Stocks')
    plt.xlabel('Ticker')
    plt.ylabel('Beta')
    plt.show()

def plot_macd(data, ticker):
    """
    Plot the Moving Average Convergence Divergence (MACD) for a given stock.

    Args:
        data (DataFrame): The stock data.
        ticker (str): The stock ticker symbol.

    Returns:
        None

    Notes:
        The Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that 
        shows the relationship between two moving averages of a stock's price. To calculate the MACD, you 
        typically use the 12 day and 26 day exponential moving averages (EMAs), and also plot the signal line, 
        which is a 9 day EMA of the MACD. A positive MACD indicates a positive momentum.
    """
    # Calculate the 12-day and 26-day EMA
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()

    # Calculate the MACD
    macd = ema_12 - ema_26

    # Calculate the signal line
    signal = macd.ewm(span=9, adjust=False).mean()

    # Plot MACD and signal line
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, macd, label=f'{ticker} MACD')
    plt.plot(data.index, signal, label=f'{ticker} Signal Line')
    plt.title(f'MACD and Signal Line of {ticker}')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend()
    plt.show()

def display_macd(tickers, start_date, end_date):
    """
    Displays historical data for each ticker and plot its MACD

    Args:
        tickers (lst): List of all ticker symbols from the assets
        start_date (str): The starting date
        end_date (str): The ending date
    """
    for ticker in tickers:
        data_ticker = yf.download(ticker, start=start_date, end=end_date)
        plot_macd(data_ticker, ticker)


def calculate_kpis(tickers, start_date, end_date):
    """
    Calculate KPIs for a list of stocks over a given time period.

    Args:
        tickers (list): A list of stock ticker symbols.
        start_date (str): The start date for the analysis.
        end_date (str): The end date for the analysis.

    Returns:
        dict: A dictionary containing the KPIs for each stock.
    """
    kpi_data = {}
    for ticker in tickers:
        # Download historical stock data
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        kpi_data[ticker] = {}

        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()      # Average gains
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()     # Average losses
        rs = gain / loss                                                  # Relative strength
        rsi = 100 - (100 / (1 + rs))                                      # RSI formula
        kpi_data[ticker]['RSI'] = rsi

        # Calculate Bollinger Bands
        middle_band = data['Close'].rolling(window=20).mean()                     # Middle band (SMA)
        upper_band = middle_band + 2 * data['Close'].rolling(window=20).std()     # Upper band
        lower_band = middle_band - 2 * data['Close'].rolling(window=20).std()     # Lower band
        kpi_data[ticker]['Bollinger Bands'] = {
            'Middle Band': middle_band,
            'Upper Band': upper_band,
            'Lower Band': lower_band
        }

        # Calculate P/E Ratio
        try:
            eps = stock.info.get('trailingEps') # Get trailing EPS
            if eps and eps != 0:
                pe_ratio = data['Close'] / eps
                kpi_data[ticker]['P/E Ratio'] = pe_ratio # Calculate P/E ratio
            else:
                kpi_data[ticker]['P/E Ratio'] = None
        except Exception as e:
            kpi_data[ticker]['P/E Ratio'] = None
            print(f"An error occurred with ticker {ticker} P/E Ratio: {e}")

        # Calculate Beta
        try:
            beta = stock.info.get('beta') # Get beta value
            kpi_data[ticker]['Beta'] = beta
        except Exception as e:
            kpi_data[ticker]['Beta'] = None
            print(f"An error occurred with ticker {ticker} Beta: {e}")

        # Calculate MACD
        ema_12 = data['Close'].ewm(span=12, adjust=False).mean()    # 12-day EMA
        ema_26 = data['Close'].ewm(span=26, adjust=False).mean()    # 26-day EMA
        macd = ema_12 - ema_26    # MACD line
        signal = macd.ewm(span=9, adjust=False).mean()   # Signal line
        kpi_data[ticker]['MACD'] = {
            'MACD': macd,
            'Signal Line': signal
        }
    return kpi_data

def display_stocks_report(kpi_data):
    """
    Displays recommendations of what stocks to hold, sell or buy based on the kpis

    Args:
        kpi_data (dict): A dictionary of all stocks and its kpis
    """
    prompt = f""" Read this data {kpi_data} and provide an executive summary with recommendations"""
    get_llm_response(llm = llm, prompt = prompt)


# Modern Portfolio Theory section 

def portfolio_performance(weights, mean_returns, cov_matrix):
    """
    Calculate portfolio performance metrics.

    Argss:
        weights (array): Asset weights in the portfolio.
        mean_returns (Series): Mean returns for each asset.
        cov_matrix (DataFrame): Covariance matrix of asset returns.

    Returns:
        float: Portfolio returns.
        float: Portfolio standard deviation.
    """
    # Calculate the expected portfolio return
    returns = np.sum(mean_returns * weights)

    # Calculate the portfolio standard deviation (volatility)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std


def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    """
    Calculate the negative Sharpe ratio for a given portfolio.

    Argss:
        weights (array): Asset weights in the portfolio.
        mean_returns (Series): Mean returns for each asset.
        cov_matrix (DataFrame): Covariance matrix of asset returns.
        risk_free_rate (float): Risk-free rate.

    Returns:
        float: Negative Sharpe ratio.
    """
    # Calculate the negative Sharpe ratio of the portfolio
    p_returns, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    # Return the negative Sharpe ratio (to minimize in optimization problems)
    return -(p_returns - risk_free_rate) / p_std


def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    """
    Find the portfolio with the maximum Sharpe ratio.

    Argss:
        mean_returns (Series): Mean returns for each asset.
        cov_matrix (DataFrame): Covariance matrix of asset returns.
        risk_free_rate (float): Risk-free rate.

    Returns:
        OptimizeResult: The optimization result containing the portfolio weights.
    """
    # Number of assets in the portfolio
    num_assets = len(mean_returns)

    # Define the arguments for the optimization function
    args = (mean_returns, cov_matrix, risk_free_rate)

    # Set up constraints (weights must sum to 1)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Define bounds for each asset weight (between 0 and 1)
    bounds = tuple((0, 1) for asset in range(num_assets))

    # Perform optimization to maximize the Sharpe ratio (minimize the negative Sharpe ratio)
    result = minimize(negative_sharpe_ratio, num_assets * [1. / num_assets,], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

if __name__ == "__main__":
    api_key = "Insert your api key here"
    llm = initialise_llm(api_key=api_key)

    #Define the year range (change to suit your needs)
    years = 2

    #Define the list of assets to analyse 
    assets = [
    "Apple (AAPL)",
    "Amazon (AMZN)",
    "Bitcoin (BTC-USD)",
    "Alphabet (GOOGL)",
    "Meta (META)",
    "Microsoft (MSFT)",
    "Nvidia (NVDA)",
    "S&P 500 index (SPY)",
    "Tesla (TSLA)"]

    #Convert asset names to their ticker symbols
    tickers = [asset.split("(")[-1].strip(")") for asset in assets]
    tickers.sort()

    start_date, end_date = calculate_date_range(years=years)

    # insert what you want to do from here:
    kpi_data = calculate_kpis(tickers, start_date, end_date)
    #display_stocks_report(kpi_data)

    # change the risk free rate to get different results (must be a float) for modern portfolio theory and Black Litterman model
    risk_free_rate = 0.04

    # Fetch the adjusted close prices for the tickers
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Adj Close']

    # Calculate daily returns
    returns = data.pct_change(fill_method=None).dropna()

    # Calculate mean returns and covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Optimize the portfolio for maximum Sharpe ratio
    optimal_portfolio = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    optimal_weights = optimal_portfolio.x

    # Store the optimal weights in a dictionary and print the result
    weights_dict = {tickers[i]: round(optimal_weights[i], 2) for i in range(len(tickers))}
    print(weights_dict)

    #Black Litterman model

    # Fetch historical stock data
    df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Adj Close']

    # Calculate the sample mean returns and the covariance matrix
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)

    ## Define market capitalizations
    mcap = {}
    # Iterate over each ticker symbol to retrieve market capitalization
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        try:
            # Attempt to get the market capitalization from stock info
            mcap[ticker] = stock.info['marketCap']
        except KeyError:
            # If the market capitalization is not available, set it to None
            mcap[ticker] = None

    # Manually set the market capitalization for the S&P 500 index (SPY)
    mcap['SPY'] = 45000000000000

    # Define beliefs (Microsoft will outperform Google by 5%)
    Q = np.array([0.05])                # Define the vector of expected returns differences (our belief)
    P = np.zeros((1, len(tickers)))     # Initialize the matrix of constraints
    P[0, tickers.index('MSFT')] = 1     # Set the coefficient for Microsoft to 1
    P[0, tickers.index('GOOGL')] = -1   # Set the coefficient for Google to -1

    # Calculate the market implied returns
    market_prices = df["SPY"]
    delta = black_litterman.market_implied_risk_aversion(market_prices)
    market_prior = black_litterman.market_implied_prior_returns(mcap, delta, S, risk_free_rate)

    # Create the Black-Litterman model
    bl = BlackLittermanModel(S,                                   # Covariance matrix of asset returns
                            Q  = Q,                              # Vector of expected returns differences (our beliefs)
                            P = P,                               # Matrix representing the assets involved in the beliefs
                            pi = market_prior,                   # Equilibrium market returns
                            market_weights = market_prior,       # Market capitalization weights (used for the equilibrium returns)
                            risk_free_rate = risk_free_rate)     # Risk-free rate for the model

    # Get the adjusted returns and covariance matrix
    bl_returns = bl.bl_returns()
    bl_cov = bl.bl_cov()

    # Optimize the portfolio for maximum Sharpe ratio
    ef = EfficientFrontier(bl_returns, bl_cov)              # Create an Efficient Frontier object with the adjusted returns and covariance matrix
    weights = ef.max_sharpe(risk_free_rate=risk_free_rate)  # Calculate the optimal portfolio weights that maximize the Sharpe ratio, considering the risk-free rate
    cleaned_weights = ef.clean_weights()                    # Clean up the weights to remove very small values for better interpretability

    # Print the optimal weights and portfolio performance
    print(cleaned_weights)
    ef.portfolio_performance(verbose=True)




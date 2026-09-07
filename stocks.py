import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
import yfinance as yf
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from scipy.optimize import minimize
from pypfopt import risk_models, expected_returns, BlackLittermanModel, EfficientFrontier, black_litterman
from rich import print as rprint
import sys
from dotenv import load_dotenv
import os

warnings.filterwarnings("ignore", category=UserWarning, module="pypfopt")
warnings.filterwarnings("ignore", category=FutureWarning, module="pypfopt")
console = Console()
load_dotenv() # load env file
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_TIMEOUT_SECONDS = 30

def get_tickers(text_file):
    """
    Reads the assets text file and converts it into a list of tickers

    Args:
        text_file (str): The text file path which stores the assets
    Returns:
        a list of tickers
    """
    if not os.path.isfile(text_file):
        raise FileNotFoundError(f"Asset file not found at: '{text_file}'.")
    assets = []
    with open(text_file, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            elif '(' not in line or line[-1] != ')':
                print(f"Warning, skipped asset {line} as didn't meet the expect format: Name (ticker)")
            else:
                assets.append(line)
    if not assets:
        raise ValueError(f"{text_file} exists but no usable assets")
    tickers = [asset.split("(")[-1].strip(")") for asset in assets]
    tickers.sort()
    return tickers


def initialise_llms(api_key=OPENROUTER_API_KEY):
    """
    Initialising GPT, Claude and Kimi K3 LLM models

    Args:
        api_key (str): The api key linked to the OpenRouter account
    
    Returns:
        ChatOpenAI: The initialised language model
    """
    gpt_model = ChatOpenRouter(model="openai/gpt-5.6-sol", api_key=api_key, base_url=OPENROUTER_BASE_URL, temperature=0)
    claude_model = ChatOpenRouter(model="anthropic/claude-opus-5", api_key=api_key, base_url=OPENROUTER_BASE_URL, temperature=0) 
    kimi_model = ChatOpenRouter(model="moonshotai/kimi-k3", api_key=api_key, base_url=OPENROUTER_BASE_URL, temperature=0)
    return gpt_model, claude_model, kimi_model

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

def build_kpi_section_for_llm(kpi_data):
    def trend_stats(series, decimals=2):
        """
        Reduce a Series to today's value plus a short trend summary:
        recent average vs a longer baseline average, the recent range, and
        the direction (rising/falling/flat). Returns None if there's no
        usable data. Avoids data truncation when passed in LLM.

        Args:
            series (pd.Series): the specific kpi data 

        Return json object that describes the kpi data
        """
        if series is None or not isinstance(series, pd.Series):
            return None
        clean = series.dropna()
        if clean.empty:
            return None
 
        today = round(clean.iloc[-1], decimals)
        short_window = clean.tail(min(7, len(clean)))
        long_window = clean.tail(min(30, len(clean)))
        recent_avg = round(short_window.mean(), decimals)
        baseline_avg = round(long_window.mean(), decimals)
        recent_low = round(long_window.min(), decimals)
        recent_high = round(long_window.max(), decimals)
        trend = "rising" if recent_avg > baseline_avg else ("falling" if recent_avg < baseline_avg else "flat")
 
        return {
            "today": today,
            "recent_avg": recent_avg,
            "baseline_avg": baseline_avg,
            "recent_low": recent_low,
            "recent_high": recent_high,
            "trend": trend,
        }
 
    def fmt(stats):
        """
        Format a trend_stats() result as compact text, or None if absent.

        Args:
            stats: kpi data
        
        Return:
            formatted text 
        """
        if stats is None:
            return "None"
        return (f"today={stats['today']}, trend={stats['trend']} "
                f"(recent_avg={stats['recent_avg']} vs baseline_avg={stats['baseline_avg']}), "
                f"range=[{stats['recent_low']}, {stats['recent_high']}]")
 
    rsi_lines, bollinger_lines, pe_lines, beta_lines, macd_lines = [], [], [], [], []
 
    for ticker, kpis in kpi_data.items():
        rsi_lines.append(f"{ticker}: {fmt(trend_stats(kpis.get('RSI')))}")
 
        bb = kpis.get('Bollinger Bands', {})
        bollinger_lines.append(
            f"{ticker}: Middle {fmt(trend_stats(bb.get('Middle Band')))}; "
            f"Upper {fmt(trend_stats(bb.get('Upper Band')))}; "
            f"Lower {fmt(trend_stats(bb.get('Lower Band')))}"
        )
 
        pe_lines.append(f"{ticker}: {fmt(trend_stats(kpis.get('P/E Ratio')))}")
        beta_lines.append(f"{ticker}: {kpis.get('Beta')}")  # already a scalar, no Series - no trend possible
 
        macd = kpis.get('MACD', {})
        macd_lines.append(
            f"{ticker}: MACD {fmt(trend_stats(macd.get('MACD')))}; "
            f"Signal {fmt(trend_stats(macd.get('Signal Line')))}"
        )
 
    return {
        "rsi": "\n".join(rsi_lines),
        "bollinger": "\n".join(bollinger_lines),
        "pe": "\n".join(pe_lines),
        "beta": "\n".join(beta_lines),
        "macd": "\n".join(macd_lines),
    }


def get_combined_recommendation(kpi_data, api_key=OPENROUTER_API_KEY):
    """
    Get an executive summary and recommendation from both GPT-5.6 Sol and Claude
    (routed through OpenRouter), then have Kimi K3 act as an independent
    decider that synthesises the two views into a single combined
    recommendation.
 
    Workflow:
        1. GPT and Claude both receive the same KPI data, broken into named
           sections (RSI, Bollinger Bands, P/E, Beta, MACD) rather than one
           raw dict dump, in parallel, and independently produce their own
           summary + buy/sell/hold view.
        2. A final independent model that took no part in producing either view. 
           It then compares the two, states where they agree or disagree, it 
           flags the disagreement as a signal worth investigating rather than 
           hiding it, and gives the final combined recommendation. 
 
    Args:
        kpi_data (dict): Output of calculate_kpis().
        api_key (str): OpenRouter API key.
 
    Returns:
        dict: {
            "gpt_view": str,
            "claude_view": str,
            "combined_recommendation": str
        }
    """
    gpt_model, claude_model, decider_model = initialise_llms(api_key)
    sections = build_kpi_section_for_llm(kpi_data)
 
    analysis_prompt = ChatPromptTemplate.from_template(
        "You are a financial analyst. Here is the latest KPI data across all tickers:\n\n"
        "RSI (0-100, >70 overbought, <30 oversold):\n{rsi}\n\n"
        "Bollinger Bands (price relative to volatility bands):\n{bollinger}\n\n"
        "P/E Ratio (valuation):\n{pe}\n\n"
        "Beta (volatility vs market):\n{beta}\n\n"
        "MACD (momentum, line vs signal):\n{macd}\n\n"
        "Provide a short executive summary and a buy/sell/hold recommendation "
        "for each ticker, with brief reasoning."
    )
 
    gpt_chain = analysis_prompt | gpt_model | StrOutputParser()
    claude_chain = analysis_prompt | claude_model | StrOutputParser()
 
    # Run both analyst models in parallel on the same input
    parallel = RunnableParallel(gpt_view=gpt_chain, claude_view=claude_chain)
 
    try:
        results = parallel.invoke(sections)
    except Exception as e:
        print(f"Warning: analyst models failed or timed out ({type(e).__name__}: {e}). Skipping LLM recommendation.")
        return None
 
    # Decider step: Kimi K3 compares the two independent views and produces the final combined answer. 
    decider_prompt = ChatPromptTemplate.from_template(
        "Two independent analysts reviewed the same stock KPI data and gave these views:\n\n"
        "Analyst A (GPT-5.6 Sol):\n{gpt_view}\n\n"
        "Analyst B (Claude Opus 5):\n{claude_view}\n\n"
        "You are a third, independent decider with no stake in either analyst's "
        "answer. Compare the two. Where they agree, state the consensus "
        "recommendation clearly. Where they disagree, explain the disagreement "
        "and flag it as worth further investigation rather than picking a side "
        "arbitrarily. End with one combined recommendation per ticker."
    )
    decider_chain = decider_prompt | decider_model | StrOutputParser()
 
    try:
        combined = decider_chain.invoke(results)
    except Exception as e:
        print(f"Warning: Kimi K3 decider failed or timed out ({type(e).__name__}: {e}).")
        combined = None
 
    output = {
        "gpt_view": results.get("gpt_view"),
        "claude_view": results.get("claude_view"),
        "combined_recommendation": combined,
    }
 
    console.print(Panel("GPT-5.6 Sol", style="bold cyan"))
    console.print(Markdown(output["gpt_view"] or "No response."))
 
    console.print(Panel("Claude Opus 5", style="bold magenta"))
    console.print(Markdown(output["claude_view"] or "No response."))
 
    console.print(Panel("Kimi K3 (decider) - combined recommendation", style="bold green"))
    console.print(Markdown(output["combined_recommendation"] or "No response."))
 
    return output


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
    Find the portfolio with the maximum Sharpe ratio. (Return per unit of risk)

    Argss:
        mean_returns (Series): Mean returns for each asset.
        cov_matrix (DataFrame): Covariance matrix of asset returns.
        risk_free_rate (float): Risk-free rate.

    Returns:
        result: The optimization result containing the portfolio weights.
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

def display_weights_table(title, weights_dict):
    table = Table(title=title)
    table.add_column("Ticker", style="cyan")
    table.add_column("Weight", justify="right", style="green")
    for ticker, w in weights_dict.items():
        table.add_row(ticker, f"{float(w) * 100:.1f}%")
    console.print(table)

def run_mpt_optimisation(tickers, start_date, end_date, risk_free_rate):
    """
    Fetch price history and run classical Modern Portfolio Theory
    (max-Sharpe) optimization for the given tickers.
 
    Args:
        tickers (list[str]): Ticker symbols to optimize over.
        start_date (str): Start of the historical price window.
        end_date (str): End of the historical price window.
        risk_free_rate (float): Annual risk-free rate, e.g. 0.04 for 4%.
 
    Returns:
        the max-Sharpe portfolio weights
    """
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)['Adj Close']
 
    returns = data.pct_change(fill_method=None).dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
 
    optimal_portfolio = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    optimal_weights = optimal_portfolio.x
 
    weights_dict = {tickers[i]: round(optimal_weights[i], 2) for i in range(len(tickers))}
    display_weights_table("Optimal Portfolio (Max Sharpe / MPT)", weights_dict)
    return weights_dict

def run_black_litterman_optimisation(tickers, start_date, end_date, risk_free_rate,
                                      view_long_ticker='MSFT', view_short_ticker='GOOGL',
                                      view_pct=0.05, market_ticker='SPY',
                                      market_cap_override=70000000000000):
    """
    Fetch price history and run a Black-Litterman optimisation, blending
    market-implied equilibrium returns with one manually specified view.
 
    The view defaults to "view_long_ticker will outperform view_short_ticker
    by view_pct" (e.g. MSFT beating GOOGL by 5%) - both tickers must be
    present in `tickers`, or this raises a clear error up front rather than
    a raw ValueError from a failed .index() lookup deep in the function.
 
    Args:
        tickers (list[str]): Ticker symbols to optimize over.
        start_date (str): Start of the historical price window.
        end_date (str): End of the historical price window.
        risk_free_rate (float): Annual risk-free rate, e.g. 0.04 for 4%.
        view_long_ticker (str): Ticker expected to outperform. Must be in `tickers`.
        view_short_ticker (str): Ticker expected to underperform. Must be in `tickers`.
        view_pct (float): The expected outperformance, e.g. 0.05 for 5%.
        market_ticker (str): Ticker used as the market proxy (for risk aversion
            and as the source of `market_cap_override`). Must be in `tickers`.
        market_cap_override (float): Manual market cap for `market_ticker`,
            since index ETFs don't report one the way a company does.
 
    Returns:
        tuple(dict, EfficientFrontier): (cleaned_weights, ef) - the cleaned
        weights dict for display, and the EfficientFrontier object itself
        (so the caller can still call ef.portfolio_performance(verbose=True)
        if wanted).
    """
    missing = [t for t in (view_long_ticker, view_short_ticker) if t not in tickers]
    if missing:
        raise ValueError(f"Black-Litterman view requires {missing} in the ticker list, but they're missing from {tickers}.")
 
    df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)['Adj Close']
 
    S = risk_models.sample_cov(df) # covriance matrix showing how the stocks moves together

    # market capitalisation for each stock
    mcap = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        try:
            mcap[ticker] = stock.info['marketCap']
        except KeyError:
            mcap[ticker] = None

    mcap[market_ticker] = market_cap_override
 
    Q = np.array([view_pct]) # size of the belief (e.g 0.05 for outperformance)
    P = np.zeros((1, len(tickers))) # stocks that contribute to that belief
    P[0, tickers.index(view_long_ticker)] = 1 # find the stock that is expected to increase
    P[0, tickers.index(view_short_ticker)] = -1 # find the stock that is expected to decrease
 
    market_prices = df[market_ticker]
    delta = black_litterman.market_implied_risk_aversion(market_prices) # estimate risk aversion of investors
    market_prior = black_litterman.market_implied_prior_returns(mcap, delta, S, risk_free_rate)
 
    bl = BlackLittermanModel(S,Q=Q,P=P,pi=market_prior,market_weights=market_prior, risk_free_rate=risk_free_rate)
 
    bl_returns = bl.bl_returns()
    bl_cov = bl.bl_cov()
 
    ef = EfficientFrontier(bl_returns, bl_cov) # similar to max sharpe ratio but for black litterman
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    cleaned_weights = ef.clean_weights()
 
    display_weights_table("Optimal Portfolio (Black-Litterman)", cleaned_weights)
    return cleaned_weights, ef

if __name__ == "__main__":
    #Define the year range (change to suit your needs)
    years = 2

    #Define the list of assets to analyse 
    assets_file = sys.argv[1]
    tickers = get_tickers(assets_file)

    start_date, end_date = calculate_date_range(years=years)

    # insert what you want to do from here:
    kpi_data = calculate_kpis(tickers, start_date, end_date)
    combined = get_combined_recommendation(kpi_data)

    # change the risk free rate to get different results (must be a float) for modern portfolio theory and Black Litterman model
    risk_free_rate = 0.04

    mpt_weights = run_mpt_optimisation(tickers, start_date, end_date, risk_free_rate)
 
    bl_weights, ef = run_black_litterman_optimisation(tickers, start_date, end_date, risk_free_rate)
    ef.portfolio_performance(verbose=True)




import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from stonks_modules import *
from numba import njit
import os

def fetch_and_save_data(ticker_symbol : str) -> (str, str): 
    """
    Fetch historical stock data and save to CSV files.
    
    This function retrieves both hourly and daily historical stock price data
    for a given ticker symbol using the yfinance API and saves the data to
    separate CSV files.
    
    Parameters
    ----------
    ticker_symbol : str
        The ticker symbol of the stock to fetch data for.
    
    Returns
    -------
    hourly_filename : str
        The filename of the saved hourly data CSV file.
    daily_filename : str
        The filename of the saved daily data CSV file.
    
    Notes
    -----
    - Files are saved with the naming convention:
      - Hourly: `{ticker}_Hourly_Data.csv`
      - Daily: `{ticker}_5Year_Daily_Data.csv`
    """

    stock_data_folder = "Stock Data"

    if not os.path.exists(stock_data_folder):
        os.mkdir(stock_data_folder)

    # --- 1. Fetching the Maximum Hourly Data ---
    # Note: Free APIs limit 1-hour intervals to a maximum of 730 days.
    try:
        hourly_data = ticker_symbol.history(period="730d", interval="1h")
        if not hourly_data.empty:
            hourly_filename = f"{stock_data_folder}/{ticker_symbol.ticker}_Hourly_Data.csv"
            hourly_data.to_csv(hourly_filename)
            print(f"Success! Saved {len(hourly_data)} rows of hourly data to '{hourly_filename}'.")
        else:
            print("No hourly data found.")
    except Exception as e:
        print(f"Error fetching hourly data: {e}")


    # --- 2. Fetching the 5-Year Daily Data ---
    try:
        daily_data = ticker_symbol.history(period="5y", interval="1d")
        if not daily_data.empty:
            daily_filename = f"{stock_data_folder}/{ticker_symbol.ticker}_5Year_Daily_Data.csv"
            daily_data.to_csv(daily_filename)
            print(f"Success! Saved {len(daily_data)} rows of daily data to '{daily_filename}'.")
        else:
            print("No daily data found.")
    except Exception as e:
        print(f"Error fetching daily data: {e}")

    return hourly_filename, daily_filename

# Use Numba's "Just-In-Time" compiler to turn this into C-speed code
@njit
def buy_basic_dip_strategy(prices : np.ndarray, buy_multiplier: float = 0.98, sell_multiplier: float = 1.02, wait_period : int = 5) -> float:
    """
    Execute a basic dip-buying trading strategy with profit-taking and cooldown.

    This function simulates a rule-based trading strategy that:
    - Starts fully invested in the asset.
    - Sells when the price increases by a specified `sell_multiplier`.
    - Buys when the price drops below a specified `buy_multiplier`
      OR after a fixed waiting period since the last sell.
    - Tracks portfolio value assuming full capital allocation per trade.

    Parameters
    ----------
    prices : np.ndarray
        Array of historical price data (e.g., hourly close prices).
        Must be a one-dimensional NumPy array.
    
    buy_multiplier : float, optional (default=0.98)
        Multiplier applied to the last transaction price that determines
        the threshold for re-entering a long position.
        Example: 0.98 means buy if price drops 2%.
    
    sell_multiplier : float, optional (default=1.02)
        Multiplier applied to the last transaction price that determines
        the threshold for selling.
        Example: 1.02 means sell if price rises 2%.
    
    wait_period : int, optional (default=5)
        Minimum number of time steps to wait after a sell before buying again,
        unless the buy threshold condition is triggered.

    Returns
    -------
    portfolio_value : float
        Final portfolio value after iterating through all price observations.

    Notes
    -----
    - The strategy assumes no transaction costs or slippage.
    - Full capital is allocated at each buy signal.
    - Only long positions are allowed (no short selling).
    """
    # Initialize the strategy by starting fully invested at the first price point
    current_stocks = 100 / prices[0]
    portfolio_value = 0.0
    last_transaction_price = prices[0]
    last_index_for_sell = 0

    long_position = True

    for i in range(1, len(prices)):
        current_price = prices[i] # Accessing the raw array is incredibly fast
        
        # Sell Signal
        if long_position:
            if current_price >= (sell_multiplier * last_transaction_price):
                portfolio_value += current_stocks * current_price
                long_position = False
                current_stocks = 0.0
                last_transaction_price = current_price
                last_index_for_sell = i
                
        # Buy Signal
        else:
            if (current_price <= (buy_multiplier * last_transaction_price)) or (i - last_index_for_sell >= wait_period):
                current_stocks += portfolio_value / current_price
                long_position = True
                portfolio_value = 0.0
                last_transaction_price = current_price

    if portfolio_value == 0.0:
        portfolio_value = current_stocks * prices[-1]

    return portfolio_value

def backtest_basic_dip_strategy(
    closes_array: np.ndarray,
    buy_multipliers: np.ndarray,
    sell_multipliers: np.ndarray,
    wait_period: int = 5
) -> np.ndarray:
    """
    Perform a grid-search backtest of the basic dip-buying strategy.

    This function evaluates the performance of the
    `buy_basic_dip_strategy` over all combinations of
    buy and sell multipliers for a fixed waiting period.
    The results are stored in a 2D matrix where each entry
    corresponds to the final portfolio value for a specific
    (buy_multiplier, sell_multiplier) pair.

    Parameters
    ----------
    closes_array : np.ndarray
        One-dimensional array of closing prices used for backtesting.

    buy_multipliers : np.ndarray
        Array of buy threshold multipliers to test.
        Each value determines the price drop required to trigger a buy.

    sell_multipliers : np.ndarray
        Array of sell threshold multipliers to test.
        Each value determines the price increase required to trigger a sell.

    wait_period : int, optional (default=5)
        Unless the buy condition is met, minimum number of time steps to wait 
        after selling before allowing a new buy.

    Returns
    -------
    profits_matrix : np.ndarray
        A 2D array of shape (len(buy_multipliers), len(sell_multipliers))
        where each entry represents the final portfolio value produced
        by the strategy for a given parameter combination.
    """
    profits_matrix = np.zeros((len(buy_multipliers), len(sell_multipliers)))

    for i, buy_mult in enumerate(tqdm(buy_multipliers, desc="Backtesting Grid")):    
        for j, sell_mult in enumerate(sell_multipliers):

                # Run the backtest for this specific combination
                final_value = buy_basic_dip_strategy(
                    closes_array, 
                    buy_mult, 
                    sell_mult, 
                    wait_period)
                
                # Store the result
                profits_matrix[i, j] = final_value

    return profits_matrix

def main():
    pass

if __name__ == "__main__":
    print("This file is not supposed to be executed")
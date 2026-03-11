import yfinance as yf
import pandas as pd
import numpy as np
from numba import njit
import functools

@functools.lru_cache(maxsize=32)
def fetch_hourly_data(ticker_symbol: str) -> pd.DataFrame:
    """
    Fetch historical stock data straight into memory.
    Cached to avoid spamming Yahoo Finance API.
    """
    ticker = yf.Ticker(ticker_symbol)
    try:
        # Free APIs limit 1-hour intervals to a maximum of 730 days.
        hourly_data = ticker.history(period="730d", interval="1h")
        if not hourly_data.empty:
            return hourly_data
        else:
            print(f"No hourly data found for {ticker_symbol}.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching hourly data for {ticker_symbol}: {e}")
        return pd.DataFrame()

@njit
def buy_basic_dip_strategy(prices: np.ndarray, buy_multiplier: float = 0.98, sell_multiplier: float = 1.02, wait_period: int = 5) -> float:
    """
    Execute a basic dip-buying trading strategy with profit-taking and cooldown. (Numba JIT optimized)
    """
    current_stocks = 100 / prices[0]
    portfolio_value = 0.0
    last_transaction_price = prices[0]
    last_index_for_sell = 0

    long_position = True

    for i in range(1, len(prices)):
        current_price = prices[i]
        
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

@njit
def buy_basic_dip_strategy_timeseries(prices: np.ndarray, buy_multiplier: float = 0.98, sell_multiplier: float = 1.02, wait_period: int = 5):
    """
    Execute a basic dip-buying trading strategy, returning the portfolio value at every time step 
    along with arrays indicating indices where buy and sell signals occurred.
    """
    current_stocks = 100 / prices[0]
    portfolio_value = 0.0
    last_transaction_price = prices[0]
    last_index_for_sell = 0
    long_position = True
    
    portfolio_history = np.zeros(len(prices))
    portfolio_history[0] = 100.0
    
    # Store indices (using a larger array and slicing later, or appending)
    buy_signals = np.zeros(len(prices), dtype=np.int32)
    sell_signals = np.zeros(len(prices), dtype=np.int32)
    buy_idx = 0
    sell_idx = 0

    # Initial buy is at index 0
    buy_signals[buy_idx] = 0
    buy_idx += 1

    for i in range(1, len(prices)):
        current_price = prices[i]
        
        # Sell Signal
        if long_position:
            if current_price >= (sell_multiplier * last_transaction_price):
                portfolio_value += current_stocks * current_price
                long_position = False
                current_stocks = 0.0
                last_transaction_price = current_price
                last_index_for_sell = i
                
                sell_signals[sell_idx] = i
                sell_idx += 1
                
        # Buy Signal
        else:
            if (current_price <= (buy_multiplier * last_transaction_price)) or (i - last_index_for_sell >= wait_period):
                current_stocks += portfolio_value / current_price
                long_position = True
                portfolio_value = 0.0
                last_transaction_price = current_price
                
                buy_signals[buy_idx] = i
                buy_idx += 1

        # Record daily value
        if long_position:
            portfolio_history[i] = current_stocks * current_price
        else:
            portfolio_history[i] = portfolio_value

    return portfolio_history, buy_signals[:buy_idx], sell_signals[:sell_idx]

@njit
def backtest_basic_dip_strategy(
    closes_array: np.ndarray,
    buy_multipliers: np.ndarray,
    sell_multipliers: np.ndarray,
    wait_period: int = 5
) -> np.ndarray:
    """
    Perform a grid-search backtest of the basic dip-buying strategy.
    """
    profits_matrix = np.zeros((len(buy_multipliers), len(sell_multipliers)))

    for i in range(len(buy_multipliers)):
        for j in range(len(sell_multipliers)):
            # Run the backtest for this specific combination
            final_value = buy_basic_dip_strategy(
                closes_array, 
                buy_multipliers[i], 
                sell_multipliers[j], 
                wait_period)
            
            # Store the result
            profits_matrix[i, j] = final_value

    return profits_matrix

def run_all_backtests(closes, buy_arr, sell_arr, baseline_profit):
    """
    Tests wait periods from 1 to 20. Returns results dict, global min/max, and wait periods list.
    """
    wait_periods = list(range(1, 21))
    results = {}
    
    for wait in wait_periods:
        profits = backtest_basic_dip_strategy(closes, buy_arr, sell_arr, wait)
        # Convert raw profit to % difference from baseline
        results[wait] = (profits - baseline_profit) * 100 / baseline_profit

    global_min = min(arr.min() for arr in results.values())
    global_max = max(arr.max() for arr in results.values())
    return results, global_min, global_max, wait_periods

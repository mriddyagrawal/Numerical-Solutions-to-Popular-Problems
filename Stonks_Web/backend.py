import yfinance as yf
import pandas as pd
import numpy as np
from numba import njit
import streamlit as st

@st.cache_data(ttl=3600)  # Cache the data for 1 hour to avoid spamming Yahoo Finance API
def fetch_hourly_data(ticker_symbol: str) -> pd.DataFrame:
    """
    Fetch historical stock data straight into memory.
    """
    ticker = yf.Ticker(ticker_symbol)
    try:
        # Free APIs limit 1-hour intervals to a maximum of 730 days.
        hourly_data = ticker.history(period="730d", interval="1h")
        if not hourly_data.empty:
            return hourly_data
        else:
            st.error("No hourly data found.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching hourly data: {e}")
        return pd.DataFrame()

@njit
def buy_basic_dip_strategy(prices: np.ndarray, buy_multiplier: float = 0.98, sell_multiplier: float = 1.02, wait_period: int = 5) -> float:
    """
    Execute a basic dip-buying trading strategy with profit-taking and cooldown. (Numba JIT optimized)
    """
    # Initialize the strategy by starting fully invested at the first price point
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
def buy_basic_dip_strategy_timeseries(prices: np.ndarray, buy_multiplier: float = 0.98, sell_multiplier: float = 1.02, wait_period: int = 5) -> np.ndarray:
    """
    Execute a basic dip-buying trading strategy, returning the portfolio value at every time step.
    """
    current_stocks = 100 / prices[0]
    portfolio_value = 0.0
    last_transaction_price = prices[0]
    last_index_for_sell = 0
    long_position = True
    
    portfolio_history = np.zeros(len(prices))
    portfolio_history[0] = 100.0

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

        # Record daily value
        if long_position:
            portfolio_history[i] = current_stocks * current_price
        else:
            portfolio_history[i] = portfolio_value

    return portfolio_history

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

    for i, buy_mult in enumerate(buy_multipliers):    
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

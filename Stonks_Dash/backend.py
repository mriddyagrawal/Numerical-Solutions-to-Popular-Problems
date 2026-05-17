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
    if ticker_symbol == "NIFTY 50":
        try:
            # Load local custom high-res dataset, avoiding YFinance API completely
            df = pd.read_csv("assets/NIFTY 50_5minute.csv")
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.rename(columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            }, inplace=True)
            return df
        except Exception as e:
            print(f"Error loading local NIFTY 50 dataset: {e}")
            return pd.DataFrame()

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
def buy_basic_dip_strategy(prices: np.ndarray, buy_multiplier: float = 0.98, sell_multiplier: float = 1.02, stop_loss_multiplier: float = 0.95, wait_period: int = 5) -> float:
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
            if current_price >= (sell_multiplier * last_transaction_price) or current_price <= (stop_loss_multiplier * last_transaction_price):
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
def buy_basic_dip_strategy_timeseries(prices: np.ndarray, buy_multiplier: float = 0.98, sell_multiplier: float = 1.02, stop_loss_multiplier: float = 0.95, wait_period: int = 5):
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
            if current_price >= (sell_multiplier * last_transaction_price) or current_price <= (stop_loss_multiplier * last_transaction_price):
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
def buy_momentum_dip_strategy(
    prices: np.ndarray, 
    buy_multiplier: float = 0.98, 
    sell_multiplier: float = 1.02, 
    stop_loss_multiplier: float = 0.95, 
    wait_period: int = 5, 
    momentum_window: int = 20,
    momentum_min: float = 0.005,
    momentum_max: float = 0.03,
    fallback_momentum_min: float = 0.0,
    fallback_momentum_max: float = 0.05
) -> float:
    """
    Execute a signal-driven momentum-based dip-buying strategy.
    Evaluates momentum strength (current_price / sma - 1) against defined thresholds
    rather than waiting rigidly.
    """
    current_stocks = 100 / prices[0]
    portfolio_value = 0.0
    last_transaction_price = prices[0]
    last_index_for_sell = 0

    long_position = True

    for i in range(1, len(prices)):
        current_price = prices[i]
        
        # Calculate SMA for Momentum Check (only if enough data exists)
        if i >= momentum_window:
            # Simple slicing sum for SMA (Numba handles this efficiently)
            sma = np.sum(prices[i - momentum_window : i]) / momentum_window
        else:
            # If not enough data, just use the average of what we have so far
            sma = np.sum(prices[:i]) / i
            
        # Calculate normalize momentum
        # Avoid division by zero if sma is 0 (unlikely for prices, but safe to check)
        momentum = (current_price / sma) - 1.0 if sma > 0 else 0.0
        
        # Sell Signal
        if long_position:
            if current_price >= (sell_multiplier * last_transaction_price) or current_price <= (stop_loss_multiplier * last_transaction_price):
                portfolio_value += current_stocks * current_price
                long_position = False
                current_stocks = 0.0
                last_transaction_price = current_price
                last_index_for_sell = i
                
        # Buy Signal
        else:
            # 1. Primary Entry
            is_primary_dip = current_price <= (buy_multiplier * last_transaction_price)
            is_primary_momentum_ok = (momentum >= momentum_min) and (momentum <= momentum_max)
            primary_entry = is_primary_dip and is_primary_momentum_ok

            # 2. Fallback Entry after waiting
            wait_elapsed = (i - last_index_for_sell) >= wait_period
            is_fallback_momentum_ok = (momentum >= fallback_momentum_min) and (momentum <= fallback_momentum_max)
            fallback_entry = wait_elapsed and is_fallback_momentum_ok
            
            if primary_entry or fallback_entry:
                current_stocks += portfolio_value / current_price
                long_position = True
                portfolio_value = 0.0
                last_transaction_price = current_price

    if portfolio_value == 0.0:
        portfolio_value = current_stocks * prices[-1]

    return portfolio_value


@njit
def buy_momentum_dip_strategy_timeseries(
    prices: np.ndarray, 
    buy_multiplier: float = 0.98, 
    sell_multiplier: float = 1.02, 
    stop_loss_multiplier: float = 0.95, 
    wait_period: int = 5, 
    momentum_window: int = 20,
    momentum_min: float = 0.005,
    momentum_max: float = 0.03,
    fallback_momentum_min: float = 0.0,
    fallback_momentum_max: float = 0.05
):
    """
    Execute a momentum-based dip-buying strategy and capture portfolio arrays.
    Evaluates momentum strength rather than wait-period bypass logic.
    """
    current_stocks = 100 / prices[0]
    portfolio_value = 0.0
    last_transaction_price = prices[0]
    last_index_for_sell = 0
    long_position = True
    
    portfolio_history = np.zeros(len(prices))
    portfolio_history[0] = 100.0
    
    sma_history = np.zeros(len(prices))
    sma_history[0] = prices[0] # Initial SMA is the first value
    
    buy_signals = np.zeros(len(prices), dtype=np.int32)
    sell_signals = np.zeros(len(prices), dtype=np.int32)
    buy_idx = 0
    sell_idx = 0

    # Initial buy is at index 0
    buy_signals[buy_idx] = 0
    buy_idx += 1

    for i in range(1, len(prices)):
        current_price = prices[i]
        
        # Calculate SMA for Momentum Check
        if i >= momentum_window:
            sma = np.sum(prices[i - momentum_window : i]) / momentum_window
        else:
            sma = np.sum(prices[:i]) / i
            
        sma_history[i] = sma
        
        # Calculate normalize momentum
        momentum = (current_price / sma) - 1.0 if sma > 0 else 0.0
        
        # Sell Signal
        if long_position:
            if current_price >= (sell_multiplier * last_transaction_price) or current_price <= (stop_loss_multiplier * last_transaction_price):
                portfolio_value += current_stocks * current_price
                long_position = False
                current_stocks = 0.0
                last_transaction_price = current_price
                last_index_for_sell = i
                
                sell_signals[sell_idx] = i
                sell_idx += 1
                
        # Buy Signal
        else:
            # 1. Primary Entry
            is_primary_dip = current_price <= (buy_multiplier * last_transaction_price)
            is_primary_momentum_ok = (momentum >= momentum_min) and (momentum <= momentum_max)
            primary_entry = is_primary_dip and is_primary_momentum_ok

            # 2. Fallback Entry after waiting
            wait_elapsed = (i - last_index_for_sell) >= wait_period
            is_fallback_momentum_ok = (momentum >= fallback_momentum_min) and (momentum <= fallback_momentum_max)
            fallback_entry = wait_elapsed and is_fallback_momentum_ok
            
            if primary_entry or fallback_entry:
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

    return portfolio_history, buy_signals[:buy_idx], sell_signals[:sell_idx], sma_history

@njit
def backtest_basic_dip_strategy(
    closes_array: np.ndarray,
    buy_multipliers: np.ndarray,
    sell_multipliers: np.ndarray,
    stop_loss_multiplier: float = 0.95,
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
                stop_loss_multiplier,
                wait_period)
            
            # Store the result
            profits_matrix[i, j] = final_value

    return profits_matrix

@njit
def backtest_momentum_dip_strategy(
    closes_array: np.ndarray,
    buy_multipliers: np.ndarray,
    sell_multipliers: np.ndarray,
    stop_loss_multiplier: float = 0.95,
    wait_period: int = 5,
    momentum_window: int = 20,
    momentum_min: float = 0.005,
    momentum_max: float = 0.03,
    fallback_momentum_min: float = 0.0,
    fallback_momentum_max: float = 0.05
) -> np.ndarray:
    """
    Perform a grid-search backtest of the momentum dip-buying strategy.
    """
    profits_matrix = np.zeros((len(buy_multipliers), len(sell_multipliers)))

    for i in range(len(buy_multipliers)):
        for j in range(len(sell_multipliers)):
            # Run the backtest for this specific combination
            final_value = buy_momentum_dip_strategy(
                closes_array, 
                buy_multipliers[i], 
                sell_multipliers[j], 
                stop_loss_multiplier,
                wait_period,
                momentum_window,
                momentum_min,
                momentum_max,
                fallback_momentum_min,
                fallback_momentum_max)
            
            # Store the result
            profits_matrix[i, j] = final_value

    return profits_matrix

def run_all_backtests(closes, buy_arr, sell_arr, stop_loss_multiplier, baseline_profit, strategy_type="basic", momentum_window=20, momentum_min=0.005, momentum_max=0.03, fallback_momentum_min=0.0, fallback_momentum_max=0.05):
    """
    Tests wait periods from 1 to 20. Returns results dict, global min/max, and wait periods list.
    Supports strategy_type of 'basic' or 'momentum'.
    """
    wait_periods = list(range(1, 21))
    results = {}
    
    for wait in wait_periods:
        if strategy_type == "momentum":
            profits = backtest_momentum_dip_strategy(closes, buy_arr, sell_arr, stop_loss_multiplier, wait, momentum_window, momentum_min, momentum_max, fallback_momentum_min, fallback_momentum_max)
        else:
            profits = backtest_basic_dip_strategy(closes, buy_arr, sell_arr, stop_loss_multiplier, wait)
            
        # Convert raw profit to % difference from baseline
        results[wait] = (profits - baseline_profit) * 100 / baseline_profit

    global_min = min(arr.min() for arr in results.values())
    global_max = max(arr.max() for arr in results.values())
    return results, global_min, global_max, wait_periods

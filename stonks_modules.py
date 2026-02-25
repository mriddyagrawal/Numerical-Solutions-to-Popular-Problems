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
    # --- 1. Fetching the Maximum Hourly Data ---
    # Note: Free APIs limit 1-hour intervals to a maximum of 730 days.
    try:
        hourly_data = ticker_symbol.history(period="730d", interval="1h")
        if not hourly_data.empty:
            hourly_filename = f"{ticker_symbol.ticker}_Hourly_Data.csv"
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
            daily_filename = f"{ticker_symbol.ticker}_5Year_Daily_Data.csv"
            daily_data.to_csv(daily_filename)
            print(f"Success! Saved {len(daily_data)} rows of daily data to '{daily_filename}'.")
        else:
            print("No daily data found.")
    except Exception as e:
        print(f"Error fetching daily data: {e}")

    return hourly_filename, daily_filename


def main():
    pass

if __name__ == "__main__":
    print("This file is not supposed to be executed")
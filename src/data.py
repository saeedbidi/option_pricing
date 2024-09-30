import yfinance as yf
import numpy as np

class DataHandler:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = None
        self.S = None

    def get_stock_data(self):
        """
        Fetch stock data from Yahoo Finance.
        """
        try:
            # Fetch data for a short period to get the latest price
            self.data = yf.download(self.ticker, period="1d")
            if self.data.empty:
                raise ValueError("No data found for the given ticker.")
            
            if 'Adj Close' in self.data.columns and not self.data['Adj Close'].empty:
                self.S = self.data['Adj Close'].iloc[-1]
                print('Current Stock Price:', self.S)
            else:
                raise ValueError("Adjusted Close data is not available.")

        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return None

    def calculate_historical_volatility(self, start_date, end_date, window=252):
        """
        Calculate the historical volatility based on stock price data.

        Args:
            start_date (str): Start date for the historical data in "YYYY-MM-DD" format.
            end_date (str): End date for the historical data in "YYYY-MM-DD" format.
            window (int): Rolling window for volatility calculation (default is 252 days).

        Returns:
            float: Annualised historical volatility.
        """
        try:
            self.data = yf.download(self.ticker, start=start_date, end=end_date)
            if self.data.empty:
                raise ValueError("No data found for the given ticker.")
            
            if 'Adj Close' in self.data.columns and not self.data['Adj Close'].empty:
                self.data['Returns'] = self.data['Adj Close'].pct_change()
                historical_volatility = self.data['Returns'].std() * np.sqrt(252)  # Annualised volatility
                sigma = historical_volatility  # Store volatility for use in other methods
                return historical_volatility
            else:
                raise ValueError("Adjusted Close data is not available.")
            
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return None


import yfinance as yf
import numpy as np

class DataHandler:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = None
        self.S = None

    def _flatten_columns(self, df):
        # yfinance >= 0.2.x returns MultiIndex columns even for single tickers
        if hasattr(df.columns, 'levels'):
            df.columns = df.columns.get_level_values(0)
        return df

    def get_stock_data(self):
        """
        Fetch stock data from Yahoo Finance.
        """
        try:
            self.data = yf.download(self.ticker, period="1d", auto_adjust=True, progress=False)
            if self.data.empty:
                raise ValueError("No data found for the given ticker.")

            self.data = self._flatten_columns(self.data)

            if 'Close' in self.data.columns and not self.data['Close'].empty:
                self.S = float(self.data['Close'].iloc[-1])
                print('Current Stock Price:', self.S)
            else:
                raise ValueError("Close data is not available.")

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
            self.data = yf.download(self.ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
            if self.data.empty:
                raise ValueError("No data found for the given ticker.")

            self.data = self._flatten_columns(self.data)

            if 'Close' in self.data.columns and not self.data['Close'].empty:
                self.data['Returns'] = self.data['Close'].pct_change()
                historical_volatility = self.data['Returns'].std() * np.sqrt(252)  # Annualised volatility
                return historical_volatility
            else:
                raise ValueError("Close data is not available.")

        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return None


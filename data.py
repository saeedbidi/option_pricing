import yfinance as yf

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

    # You can add more data-related methods here


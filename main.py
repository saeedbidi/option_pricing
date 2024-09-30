import os
from datetime import datetime

import pandas as pd
from data import DataHandler
from option_pricing.backtester import Backtester
from results import ResultsHandler

class Main:
    def __init__(self, ticker, output_folder):
        self.ticker = ticker
        self.output_folder = output_folder
        
        # Initialize the data handler, backtester, and report generator
        self.data_handler = DataHandler(ticker, output_folder)
        self.backtester = Backtester(ticker, output_folder)
        self.report_generator = ResultsHandler(ticker, output_folder)

    def run(self, data_file_path, n_data=None, n_each_day=5, risk_free_rate=0.05, num_steps=100, keep_first_n_rows_per_date=False):
        # Step 1: Load and preprocess the data
        self.data_handler.load_data(data_file_path, n_data)
        historical_volatility = self.data_handler.calculate_historical_volatility()
        
        # Step 2: Run the backtest
        backtest_results = self.backtester.backtest(data_file_path, n_data, n_each_day, risk_free_rate, num_steps, keep_first_n_rows_per_date)
        
        # Step 3: Generate a report with the calculated metrics
        for index, row in backtest_results.iterrows():
            # Extract required data for the report
            S = row['stock_price']
            K = row['strike']
            T = (pd.to_datetime(row['expiration']) - pd.to_datetime(row['date'])).days / 365  # Time to maturity
            sigma = row['implied_volatility']
            market_price = row['mid_price']
            bs_price = row['BS_price']
            delta = row['BS_delta']  # Assuming you have these calculated
            gamma_val = row['BS_gamma']
            vega_val = row['BS_vega']
            theta_val = row['BS_theta']
            rho_val = row['BS_rho']
            iv = row['implied_volatility']  # Assuming this is also part of your results
            
            # Generate report
            self.report_generator.generate_report(S, K, T, risk_free_rate, sigma, bs_price, delta, gamma_val, vega_val, theta_val, rho_val, iv, market_price)

if __name__ == "__main__":
    # User-defined parameters
    # Initialize the OptionPricing class
    OUTPUT_FOLDER = "output"

    # For historical volatility
    start_date_volatility = datetime(2023, 1, 1).strftime('%Y-%m-%d')
    end_date_volatility = datetime.today().strftime('%Y-%m-%d')

    # For Monte Carlo
    mc_num_sim = 10000
    bt_num_step = 100

   # AAPL
    ticker = "AAPL"  # Example ticker
    option_type = "call"  # Can be 'call' or 'put'
    K = 207.5  # Example strike price
    days_to_maturity = 7
    T = days_to_maturity / 365  # Time to maturity (in years)
    r = 0.05  # Example risk-free rate
    market_price = 22.25  # Example market price



    ticker = "AAPL"  # Example ticker
    output_folder = "output"
    data_file_path = "path_to_your_data.csv"  # Change to your actual data file path

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create an instance of the Main class and run the process
    main = Main(ticker, output_folder)
    main.run(data_file_path, n_data=1000, n_each_day=5, risk_free_rate=0.05, num_steps=100, keep_first_n_rows_per_date=True)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import OptionPricingModels


class Backtester:
    def __init__(self, ticker, output_folder):
        self.ticker = ticker
        self.output_folder = output_folder

    def backtest(self, file_path, n_data=None, n_each_day=5, risk_free_rate=0.05, num_steps=100, keep_first_n_rows_per_date=False):
        # Load the data
        if n_data is not None:
            self.stock_data = pd.read_csv(file_path, nrows=n_data)
        else:
            self.stock_data = pd.read_csv(file_path)

        # Filter for the desired ticker
        self.stock_data = self.stock_data[self.stock_data['act_symbol'] == self.ticker]

        # Optionally keep only one row per date
        if keep_first_n_rows_per_date:
            self.stock_data = self.stock_data.groupby('date').head(n_each_day)  # Keep the first 5 records for each date

        # Calculate mid prices
        self.stock_data['mid_price'] = (self.stock_data['bid'] + self.stock_data['ask']) / 2

        # Initialize columns for theoretical prices
        self.stock_data['BS_price'] = np.nan
        self.stock_data['BT_price'] = np.nan
        self.stock_data['MC_price'] = np.nan

        for index, row in self.stock_data.iterrows():
            S = row['stock_price']
            K = row['strike']
            T = (pd.to_datetime(row['expiration']) - pd.to_datetime(row['date'])).days / 365  # Time to maturity
            sigma = row['implied_volatility']
            option_type = row['call_put'].lower()
            # Calculate theoretical prices
            option_pricing_models = OptionPricingModels(S, K, T, risk_free_rate, sigma, option_type)
            
            bs_price = option_pricing_models.black_scholes_option()
            # print(f"Black-Scholes {option_type} price: {bs_price}")
    
            
            # # Calculate Binomial Tree price
            bt_price = option_pricing_models.binomial_tree_option_price()
            # print(f"Binomial Tree {option_type} price: {bt_price}")

          # # Calculate Monte Carlo price
            mc_price = option_pricing_models.monte_carlo_option_price()
            # print(f"Monte Carlo {option_type} price: {mc_price[0]}")


            # Store the calculated prices
            self.stock_data.at[index, 'BS_price'] = bs_price
            self.stock_data.at[index, 'BT_price'] = bt_price
            self.stock_data.at[index, 'MC_price'] = mc_price[0]

        # Calculate errors
        self.stock_data['BS_error'] = self.stock_data['BS_price'] - self.stock_data['mid_price']
        self.stock_data['BT_error'] = self.stock_data['BT_price'] - self.stock_data['mid_price']
        self.stock_data['MC_error'] = self.stock_data['MC_price'] - self.stock_data['mid_price']

        # Calculate percentage errors
        self.stock_data['BS_error_pct'] = ((self.stock_data['BS_price'] - self.stock_data['mid_price']) / self.stock_data['mid_price']) * 100
        self.stock_data['BT_error_pct'] = ((self.stock_data['BT_price'] - self.stock_data['mid_price']) / self.stock_data['mid_price']) * 100
        self.stock_data['MC_error_pct'] = ((self.stock_data['MC_price'] - self.stock_data['mid_price']) / self.stock_data['mid_price']) * 100

        # Compute error metrics
        mae_bs = self.stock_data['BS_error'].abs().mean()
        rmse_bs = (self.stock_data['BS_error'] ** 2).mean() ** 0.5
        
        mae_bt = self.stock_data['BT_error'].abs().mean()
        rmse_bt = (self.stock_data['BT_error'] ** 2).mean() ** 0.5
        
        mae_mc = self.stock_data['MC_error'].abs().mean()
        rmse_mc = (self.stock_data['MC_error'] ** 2).mean() ** 0.5

        print(f'Black-Scholes MAE: {mae_bs}, RMSE: {rmse_bs}')
        print(f'Binomial Tree MAE: {mae_bt}, RMSE: {rmse_bt}')
        print(f'Monte Carlo MAE: {mae_mc}, RMSE: {rmse_mc}')

        # Create output directories
        current_dir = os.getcwd()

        backtest_folder = os.path.join(os.path.dirname(current_dir), 'backtesting')
        os.makedirs(backtest_folder, exist_ok=True)

        # Save results to output folder
        self.stock_data.to_csv(os.path.join(backtest_folder, f'{self.ticker}_backtest_results.csv'), index=False)
        backtest_results = self.stock_data

        # Convert 'expiration' to datetime format
        backtest_results['expiration'] = pd.to_datetime(backtest_results['expiration'])

        # Sort the DataFrame based on the 'expiration' column
        backtest_results = backtest_results.sort_values(by='expiration')

        # Set common styling for plots
        plt.rcParams.update({
            'font.size': 14,           # Font size
            'lines.linewidth': 5,      # Line width
            'figure.dpi': 300          # Image quality
        })

        # 1. Scatter Plot of Mid Price vs. Model Prices
        plt.figure(figsize=(12, 6))
        plt.plot(backtest_results['mid_price'], backtest_results['BS_price'], 
                label='Black-Scholes Price', marker='o', color='blue', linestyle='-')
        plt.plot(backtest_results['mid_price'], backtest_results['BT_price'], 
                label='Binomial Tree Price', marker='o', color='black', linestyle='--')
        plt.plot(backtest_results['mid_price'], backtest_results['MC_price'], 
                label='Monte Carlo Price', marker='o', color='red', linestyle=':')

        plt.xlabel('Mid Option Price')
        plt.ylabel('Predicted Option Price')
        plt.title(f'Option Mid Price vs. Model Prices for {self.ticker} (2013)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(backtest_folder, 'mid_price_vs_model_prices.png'))
        plt.show()  # Display the plot
        plt.close()

        # Sort the DataFrame based on the 'strike' column
        backtest_results_sorted = backtest_results.sort_values(by='strike')
        # 2. Line Plot of Price Across Dates
        backtest_results_sorted_zero_removed = backtest_results_sorted[backtest_results_sorted['mid_price'] > 5]
        backtest_results_sorted_zero_removed = backtest_results_sorted_zero_removed.sort_values(by='date')
        plt.figure(figsize=(12, 6))

        # Plot Black-Scholes Prediction
        plt.plot(backtest_results_sorted_zero_removed['date'], 
                backtest_results_sorted_zero_removed['MC_price'], 
                label='Monte Carlo Prediction', marker='o', color='red', linestyle='-')

        # Plot Mid Price
        plt.plot(backtest_results_sorted_zero_removed['date'], 
                backtest_results_sorted_zero_removed['mid_price'], 
                label='Mid Price', marker='o', color='black', linestyle='--')

        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'Price Across Dates for {self.ticker} (2013)')
        # Customize x-ticks: Show only every nth date to avoid overcrowding
        n_ticks = 10  # Adjust this to control the number of dates shown
        dates = backtest_results_sorted_zero_removed['date']
        plt.xticks(dates[::n_ticks], rotation=45)  # Show every nth date, rotated for readability
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(backtest_folder, 'price_vs_date_plot.png'))
        plt.show()
        plt.close()

        return backtest_results

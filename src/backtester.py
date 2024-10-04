import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import OptionPricingModels

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class Backtester:
    def __init__(self, ticker, output_folder):
        self.ticker = ticker
        self.output_folder = output_folder

    def train_machine_learning_model(self, csv_file, risk_free_rate=0.05):
        """
        Predict mid price using machine learning on the provided CSV data.
        """
        # Load the data
        data = pd.read_csv(csv_file)

        # Convert 'call_put' to numerical values
        data['call_put'] = data['call_put'].map({'call': 1, 'put': 0})

        # Calculate mid price
        data['mid_price'] = (data['bid'] + data['ask']) / 2
        data['T'] = (pd.to_datetime(data['expiration']) - pd.to_datetime(data['date'])).dt.days / 365  # Time to maturity
        data['risk_free_rate'] = risk_free_rate

        # Select features and target
        features = data[['stock_price', 'strike', 'T', 'risk_free_rate', 'implied_volatility', 'call_put']]
        target = data['mid_price']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Train the Random Forest model
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)

        # Calculate the mean squared error
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"Machine Learning Model MSE: {mse}")

        # Return the trained model
        return model

    def predict_columns_mid_price_with_ml(self, model, ml_features):
        """
        Predict mid price using the trained machine learning model for multiple rows.
        """
        ml_price = model.predict(ml_features)
        return ml_price  # Return the predicted values


    def backtest(self, file_path, n_data=None, n_each_day=5, risk_free_rate=0.05, num_steps=100, keep_first_n_rows_per_date=False):
        # Load the data
        if n_data is not None:
            self.stock_data = pd.read_csv(file_path, nrows=n_data)
        else:
            self.stock_data = pd.read_csv(file_path)

        # Filter for the desired ticker
        self.stock_data = self.stock_data[self.stock_data['act_symbol'] == self.ticker]
        self.stock_data['T'] = (pd.to_datetime(self.stock_data['expiration']) - pd.to_datetime(self.stock_data['date'])).dt.days / 365

        # Optionally keep only n_each_day rows per date
        if keep_first_n_rows_per_date:
            self.stock_data = self.stock_data.groupby('date').head(n_each_day).reset_index(drop=True)
        # Calculate mid prices just once
        self.stock_data['mid_price'] = (self.stock_data['bid'] + self.stock_data['ask']) / 2
        # Convert 'call_put' to lower case for consistency
        self.stock_data['call_put'] = self.stock_data['call_put'].str.lower()

        # Add the risk-free rate column
        self.stock_data['risk_free_rate'] = risk_free_rate
        # Prepare vectors for pricing models
        S_array = self.stock_data['stock_price'].values
        K_array = self.stock_data['strike'].values
        T_array = self.stock_data['T'].values
        sigma_array = self.stock_data['implied_volatility'].values
        option_type_array = self.stock_data['call_put'].values

        # Instantiate the OptionPricingModels class with the vectors
        option_pricing_models = OptionPricingModels(S_array, K_array, T_array, risk_free_rate, sigma_array, option_type_array)

        # Calculate Black-Scholes prices
        self.stock_data['BS_price'] = option_pricing_models.black_scholes_option()

        # Calculate Binomial Tree prices
        self.stock_data['BT_price'] = option_pricing_models.binomial_tree_option_price(N=num_steps)

        # Calculate Monte Carlo prices (optional adjustment for outputs)
        self.stock_data['MC_price'], _ = option_pricing_models.new_monte_carlo_option_price(num_simulations=10000)

        # --- Machine Learning Predictions for Mid Price ---

        # Step 1: Train the ML model (this can be done outside the method for efficiency)
        ml_model = self.train_machine_learning_model(file_path, risk_free_rate=risk_free_rate)

        # Step 2: Prepare features for ML price prediction
        ml_features = self.stock_data[['stock_price', 'strike', 'T', 'risk_free_rate', 'implied_volatility', 'call_put']].copy()
        
        # Convert 'call_put' column to numerical values (1 for call, 0 for put)
        ml_features['call_put'] = ml_features['call_put'].map({'call': 1, 'put': 0})

        # Step 3: Predict ML mid prices using the trained model
        self.stock_data['ML_price'] = self.predict_columns_mid_price_with_ml(ml_model, ml_features)


        display(self.stock_data)
        # Calculate errors
        self.stock_data['BS_error'] = self.stock_data['BS_price'] - self.stock_data['mid_price']
        self.stock_data['BT_error'] = self.stock_data['BT_price'] - self.stock_data['mid_price']
        self.stock_data['MC_error'] = self.stock_data['MC_price'] - self.stock_data['mid_price']
        self.stock_data['ML_error'] = self.stock_data['ML_price'] - self.stock_data['mid_price']

        # Calculate percentage errors
        self.stock_data['BS_error_pct'] = ((self.stock_data['BS_price'] - self.stock_data['mid_price']) / self.stock_data['mid_price']) * 100
        self.stock_data['BT_error_pct'] = ((self.stock_data['BT_price'] - self.stock_data['mid_price']) / self.stock_data['mid_price']) * 100
        self.stock_data['MC_error_pct'] = ((self.stock_data['MC_price'] - self.stock_data['mid_price']) / self.stock_data['mid_price']) * 100
        self.stock_data['ML_error_pct'] = ((self.stock_data['ML_price'] - self.stock_data['mid_price']) / self.stock_data['mid_price']) * 100

        # Compute error metrics
        mae_bs = self.stock_data['BS_error'].abs().mean()
        rmse_bs = (self.stock_data['BS_error'] ** 2).mean() ** 0.5
        
        mae_bt = self.stock_data['BT_error'].abs().mean()
        rmse_bt = (self.stock_data['BT_error'] ** 2).mean() ** 0.5
        
        mae_mc = self.stock_data['MC_error'].abs().mean()
        rmse_mc = (self.stock_data['MC_error'] ** 2).mean() ** 0.5

        mae_ml = self.stock_data['ML_error'].abs().mean()
        rmse_ml = (self.stock_data['ML_error'] ** 2).mean() ** 0.5

        print(f'Black-Scholes MAE: {mae_bs}, RMSE: {rmse_bs}')
        print(f'Binomial Tree MAE: {mae_bt}, RMSE: {rmse_bt}')
        print(f'Monte Carlo MAE: {mae_mc}, RMSE: {rmse_mc}')
        print(f'Machine Learning MAE: {mae_ml}, RMSE: {rmse_ml}')

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
        backtest_results = backtest_results.sort_values(by='mid_price')

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
                label='Binomial Tree Price', marker='o', color='yellow', linestyle='--')
        plt.plot(backtest_results['mid_price'], backtest_results['MC_price'], 
                label='Monte Carlo Price', marker='o', color='purple', linestyle=':')
        plt.plot(backtest_results['mid_price'], backtest_results['ML_price'], 
                label='Machine Learning Price', marker='o', color='green', linestyle='-.')
        
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
                backtest_results_sorted_zero_removed['BS_price'], 
                label='Black-Scholes Prediction', marker='o', color='blue', linestyle='-')
        
        plt.plot(backtest_results_sorted_zero_removed['date'], 
                backtest_results_sorted_zero_removed['BT_price'], 
                label='Binomial Tree Prediction', marker='o', color='yellow', linestyle='--')
        
        plt.plot(backtest_results_sorted_zero_removed['date'], 
                backtest_results_sorted_zero_removed['MC_price'], 
                label='Monte Carlo Prediction', marker='o', color='purple', linestyle=':')

        # Plot Machine Learning Prediction
        plt.plot(backtest_results_sorted_zero_removed['date'], 
                backtest_results_sorted_zero_removed['ML_price'], 
                label='Machine Learning Prediction',  marker='o', color='green', linestyle='-.')
        
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

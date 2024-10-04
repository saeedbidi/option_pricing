import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import OptionPricingModels

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch

class Backtester:
    def __init__(self, ticker, output_folder):
        self.ticker = ticker
        self.output_folder = output_folder

    def train_machine_learning_model(self, csv_file, risk_free_rate=0.05):
        """
        Train multiple machine learning models (Random Forest, XGBoost, Linear Regression, Polynomial Regression, and SVR) 
        to predict the mid price using the provided CSV data.
        """
        print('Training started...')
        # Record the total time for training all models
        start_time = time.time()
        # Load the data
        data = pd.read_csv(csv_file)

        # Sample 100,000 rows from the dataset for training
        # data = data.sample(n=100000, random_state=42)    

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

        # Scale features (for SVR)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)


        # Define a dictionary to store the models and their names
        models = {
            # 'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            # 'XGBoost': XGBRegressor(n_estimators=100, max_depth=10, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'XGBoost': XGBRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            # 'XGBoost': XGBRegressor(n_estimators=100, max_depth=10, random_state=42, tree_method='gpu_hist'),

            'Linear Regression': LinearRegression(),
            # 'Polynomial Regression (degree 2)': Pipeline([('poly', PolynomialFeatures(degree=2)),
            #                                             ('linear', LinearRegression())]),
            'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
        }

        # Dictionary to store the MSE of each model
        mse_results = {}

        # Loop through each model, train it, and calculate MSE
        for name, model in models.items():
            # Use scaled features only for SVR; for other models, use the original features
            if name == 'SVR':
                model.fit(X_train_scaled, y_train)  # Train the model on scaled data
                predictions = model.predict(X_test_scaled)  # Make predictions on scaled data
            else:
                model.fit(X_train, y_train)  # Train the model on original data
                predictions = model.predict(X_test)  # Make predictions on original data
            
            mse = mean_squared_error(y_test, predictions)  # Calculate MSE
            mse_results[name] = mse
            # print(f"{name} Model MSE: {mse}")
        print('Training finished...')
        # Print total training time for all models
        end_time = time.time()
        print(f"Total training time: {end_time - start_time:.2f} seconds")
        # Return the models and their MSE results
        return models, mse_results

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

        # Step 1: Train all ML models (this can be done outside the method for efficiency)
        ml_models, mse_results = self.train_machine_learning_model(file_path, risk_free_rate=risk_free_rate)

        # Step 2: Prepare features for ML price prediction
        ml_features = self.stock_data[['stock_price', 'strike', 'T', 'risk_free_rate', 'implied_volatility', 'call_put']].copy()

        # Convert 'call_put' column to numerical values (1 for call, 0 for put)
        ml_features['call_put'] = ml_features['call_put'].map({'call': 1, 'put': 0})

        # Step 3: Predict ML mid prices using each trained model and store results in new columns
        for model_name, model in ml_models.items():
            self.stock_data[f'{model_name}_ML_price'] = self.predict_columns_mid_price_with_ml(model, ml_features)
        # Optionally print the MSE results for each model
        for model_name, mse in mse_results.items():
            print(f"{model_name} Model MSE: {mse}")

        display(self.stock_data)
        # Calculate errors
        self.stock_data['BS_error'] = self.stock_data['BS_price'] - self.stock_data['mid_price']
        self.stock_data['BT_error'] = self.stock_data['BT_price'] - self.stock_data['mid_price']
        self.stock_data['MC_error'] = self.stock_data['MC_price'] - self.stock_data['mid_price']
        # self.stock_data['ML_error'] = self.stock_data['ML_price'] - self.stock_data['mid_price']
        # Loop through ML models to calculate errors
        for model_name in ml_models.keys():
            self.stock_data[f'{model_name}_ML_error'] = self.stock_data[f'{model_name}_ML_price'] - self.stock_data['mid_price']

        # Calculate percentage errors
        self.stock_data['BS_error_pct'] = ((self.stock_data['BS_price'] - self.stock_data['mid_price']) / self.stock_data['mid_price']) * 100
        self.stock_data['BT_error_pct'] = ((self.stock_data['BT_price'] - self.stock_data['mid_price']) / self.stock_data['mid_price']) * 100
        self.stock_data['MC_error_pct'] = ((self.stock_data['MC_price'] - self.stock_data['mid_price']) / self.stock_data['mid_price']) * 100
        # self.stock_data['ML_error_pct'] = ((self.stock_data['ML_price'] - self.stock_data['mid_price']) / self.stock_data['mid_price']) * 100
        # Loop through ML models to calculate percentage errors
        for model_name in ml_models.keys():
            self.stock_data[f'{model_name}_ML_error_pct'] = ((self.stock_data[f'{model_name}_ML_price'] - self.stock_data['mid_price']) / self.stock_data['mid_price']) * 100

        # Compute error metrics
        mae_bs = self.stock_data['BS_error'].abs().mean()
        rmse_bs = (self.stock_data['BS_error'] ** 2).mean() ** 0.5
        
        mae_bt = self.stock_data['BT_error'].abs().mean()
        rmse_bt = (self.stock_data['BT_error'] ** 2).mean() ** 0.5
        
        mae_mc = self.stock_data['MC_error'].abs().mean()
        rmse_mc = (self.stock_data['MC_error'] ** 2).mean() ** 0.5

        # mae_ml = self.stock_data['ML_error'].abs().mean()
        # rmse_ml = (self.stock_data['ML_error'] ** 2).mean() ** 0.5

        # Loop through ML models to compute error metrics
        for model_name in ml_models.keys():
            mae_ml = self.stock_data[f'{model_name}_ML_error'].abs().mean()
            rmse_ml = (self.stock_data[f'{model_name}_ML_error'] ** 2).mean() ** 0.5
            
            print(f'{model_name} MAE: {mae_ml}, RMSE: {rmse_ml}')

        print(f'Black-Scholes MAE: {mae_bs}, RMSE: {rmse_bs}')
        print(f'Binomial Tree MAE: {mae_bt}, RMSE: {rmse_bt}')
        print(f'Monte Carlo MAE: {mae_mc}, RMSE: {rmse_mc}')
        print(f'Machine Learning MAE: {mae_ml}, RMSE: {rmse_ml}')

        # Create output directories
        # current_dir = os.getcwd()

        backtest_folder = self.output_folder
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
        # plt.plot(backtest_results['mid_price'], backtest_results['BS_price'], 
        #         label='Black-Scholes Price', marker='o', color='blue', linestyle='-')
        # plt.plot(backtest_results['mid_price'], backtest_results['BT_price'], 
        #         label='Binomial Tree Price', marker='o', color='yellow', linestyle='--')
        plt.plot(backtest_results['mid_price'], backtest_results['MC_price'], 
                label='Monte Carlo Price',  marker='o', color='blue', linestyle=':', markersize=11)
        plt.plot(backtest_results['mid_price'], backtest_results['XGBoost_ML_price'], 
                label='Machine Learning Price (XGBoost)', marker='o', color='red', linestyle='-')
        # plt.plot(backtest_results['mid_price'], backtest_results['Random Forest_ML_price'], 
        #         label='Machine Learning Price (Random Forest)', marker='o', color='green', linestyle='-.')

        plt.xlabel('Mid Option Price')
        plt.ylabel('Predicted Option Price')
        plt.title(f'Backtesting Option Mid Price vs. Model Prices for {self.ticker} (2013)')
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
        # plt.plot(backtest_results_sorted_zero_removed['date'], 
        #         backtest_results_sorted_zero_removed['BS_price'], 
        #         label='Black-Scholes Prediction', marker='o', color='blue', linestyle='-')
        
        # plt.plot(backtest_results_sorted_zero_removed['date'], 
        #         backtest_results_sorted_zero_removed['BT_price'], 
        #         label='Binomial Tree Prediction', marker='o', color='yellow', linestyle='--')
        
        # plt.plot(backtest_results_sorted_zero_removed['date'], 
        #         backtest_results_sorted_zero_removed['MC_price'], 
        #         label='Monte Carlo Prediction', marker='o', color='blue', linestyle=':')
        plt.plot(backtest_results_sorted_zero_removed['date'], 
         backtest_results_sorted_zero_removed['MC_price'], 
         label='Monte Carlo Prediction', marker='o', color='blue', linestyle=':', markersize=11)


        # Plot Machine Learning Prediction

        plt.plot(backtest_results_sorted_zero_removed['date'], 
                backtest_results_sorted_zero_removed['XGBoost_ML_price'], 
                label='Machine Learning Prediction (XGBoost)',  marker='o', color='red', linestyle='-')
        

        # plt.plot(backtest_results_sorted_zero_removed['date'], 
        #         backtest_results_sorted_zero_removed['Random Forest_ML_price'], 
        #         label='Machine Learning Prediction (Random Forest)',  marker='o', color='green', linestyle='-.')
        
        # Plot Mid Price
        plt.plot(backtest_results_sorted_zero_removed['date'], 
                backtest_results_sorted_zero_removed['mid_price'], 
                label='Mid Price', marker='o', color='black', linestyle='--')

        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'Backtesting Price Across Dates for {self.ticker} (2013)')
        # Customize x-ticks: Show only every nth date to avoid overcrowding
        n_ticks = 10  # Adjust this to control the number of dates shown
        dates = backtest_results_sorted_zero_removed['date']
        plt.xticks(dates[::n_ticks], rotation=45)  # Show every nth date, rotated for readability
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(backtest_folder, 'price_vs_date_plot.png'))
        plt.show()
        plt.close()


        self.generate_backtest_report(backtest_results, backtest_folder, self.ticker)

        return backtest_results
    

    def generate_backtest_report(self, backtest_results, backtest_folder, ticker):
        # Define the file path for the PDF report
        report_path = os.path.join(backtest_folder, f'{ticker}_backtest_report.pdf')
        
        # Create a canvas for PDF
        c = canvas.Canvas(report_path, pagesize=letter)
        
        # Title of the report
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, f"Backtesting Report for {ticker} (2013)")
        c.setFont("Helvetica", 12)
        
        # Add a general description
        c.drawString(100, 730, f"Backtest Date: {pd.to_datetime('today').strftime('%Y-%m-%d')}")
        
        # Section 1: Error Metrics
        c.drawString(100, 700, "Error Metrics")
        error_metrics = {
            'Black-Scholes': (backtest_results['BS_error'].abs().mean(), (backtest_results['BS_error'] ** 2).mean() ** 0.5),
            'Binomial Tree': (backtest_results['BT_error'].abs().mean(), (backtest_results['BT_error'] ** 2).mean() ** 0.5),
            'Monte Carlo': (backtest_results['MC_error'].abs().mean(), (backtest_results['MC_error'] ** 2).mean() ** 0.5)
        }
        
        y_position = 680
        for model_name, (mae, rmse) in error_metrics.items():
            c.drawString(100, y_position, f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
            y_position -= 20
        
        # Section 2: Machine Learning Models Error Metrics
        c.drawString(100, y_position - 20, "Machine Learning Error Metrics")

        # Find all columns related to ML model prices (ends with '_ML_price')
        ml_model_columns = [col for col in backtest_results.columns if '_ML_price' in col]

        # Loop through ML model columns to calculate and display MAE and RMSE
        y_position -= 40  # Adjust y_position for the next section

        for model_column in ml_model_columns:
            # Extract model name by removing '_ML_price' from the column name
            model_name = model_column.replace('_ML_price', '')

            # Calculate MAE and RMSE for each ML model based on its respective error column
            mae_ml = backtest_results[f'{model_name}_ML_error'].abs().mean()
            rmse_ml = (backtest_results[f'{model_name}_ML_error'] ** 2).mean() ** 0.5

            # Display the results for the ML model
            c.drawString(100, y_position, f"{model_name} - MAE: {mae_ml:.2f}, RMSE: {rmse_ml:.2f}")
            y_position -= 20
        # Section 3: Add Plots
        # Add the scatter plot of mid price vs model prices
        plot_image_path = os.path.join(backtest_folder, 'mid_price_vs_model_prices.png')
        c.drawImage(plot_image_path, 100, y_position - 200, width=400, height=200)
        
        # Add the price vs date plot
        date_plot_image_path = os.path.join(backtest_folder, 'price_vs_date_plot.png')
        c.drawImage(date_plot_image_path, 100, y_position - 450, width=400, height=200)
        
        # Finalize the PDF
        c.showPage()
        c.save()

        print(f"Backtesting report saved at {report_path}")

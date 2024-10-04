import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class OptionPricingModels():
    def __init__(self, S, K, T, r, sigma, option_type):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
    def black_scholes_option(self, q=0):
        """
        Calculate the option price using the Black-Scholes formula.
        """
        d1 = (np.log(self.S / self.K) + (self.r - q + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if self.option_type == 'call':
            price = self.S * np.exp(-q * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif self.option_type == 'put':
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * np.exp(-q * self.T) * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        return price

    def binomial_tree_option_price(self, N=100):
        """
        Calculate option price using the Binomial Tree method.
        """
        dt = self.T / N  # Time step
        u = np.exp(self.sigma * np.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (np.exp(self.r * dt) - d) / (u - d)  # Risk-neutral probability

        # Initialise asset prices at maturity
        ST = np.array([self.S * (u ** (N - i)) * (d ** i) for i in range(N + 1)])

        # Initialise option values at maturity
        option_values = np.maximum(0, ST - self.K) if self.option_type == 'call' else np.maximum(0, self.K - ST)

        # Backward induction
        for j in range(N - 1, -1, -1):
            option_values = np.exp(-self.r * dt) * (p * option_values[:-1] + (1 - p) * option_values[1:])

        return option_values[0]

    def monte_carlo_option_price(self, ticker=None, output_folder=None, num_simulations=10000):
        """
            Calculate option price using Monte Carlo simulation.

            Args:
                S (float): Current stock price.
                K (float): Strike price.
                T (float): Time to maturity in years.
                r (float): Risk-free rate.
                sigma (float): Volatility.
                num_simulations (int): Number of simulations. Default is 10,000.
                option_type: Call or put.
            Returns:
                float: Option price and list of plot filenames.
        """
        dt = 1 / 365  # Daily steps
        num_steps = int(self.T * 365)  # Number of days until maturity
        payoffs = []
        paths = []
        estimated_prices = []


        for _ in range(num_simulations):
            ST = self.S
            path = [self.S]
            for _ in range(num_steps):
                Z = np.random.normal()
                ST *= np.exp((self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z)
                path.append(ST)

            paths.append(path)
            payoff = max(0, ST - self.K) if self.option_type == 'call' else max(0, self.K - ST)
            payoffs.append(payoff)
            estimated_prices.append(np.exp(-self.r * self.T) * np.mean(payoffs))
        
        plot_filenames = []

        if output_folder:
            # Create output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)

            # Set common styling for plots
            plt.rcParams.update({
                'font.size': 14,           # Font size
                'lines.linewidth': 2,      # Line width
                'figure.dpi': 300          # Image quality
            })
            # Plot stock price paths
            path_filename = os.path.join(output_folder, 'Monte_Carlo_Paths.png')
            plt.figure(figsize=(12, 6))
            for path in paths[:200]:  # Limit to 200 paths for clarity
                plt.plot(path, linewidth=0.5)
            plt.title(f'Monte Carlo Simulation of Stock Price Paths ({ticker})')
            plt.xlabel('Time Steps')
            plt.ylabel('Stock Price')
            plt.savefig(path_filename)
            plt.close()
            plot_filenames.append(path_filename)

            # Histogram of payoffs
            payoff_filename = os.path.join(output_folder, 'Payoff_Histogram.png')
            plt.figure(figsize=(12, 6))
            plt.hist(payoffs, bins=50)
            plt.title(f'Histogram of Simulated Payoffs ({ticker})')
            plt.xlabel('Payoff')
            plt.ylabel('Frequency')
            plt.savefig(payoff_filename)
            plt.close()
            plot_filenames.append(payoff_filename)

            # Convergence plot
            convergence_filename = os.path.join(output_folder, 'Convergence_Plot.png')
            plt.figure(figsize=(12, 6))
            plt.plot(estimated_prices, color='blue')
            plt.title(f'Convergence of Monte Carlo Option Price ({ticker})')
            plt.xlabel('Number of Simulations')
            plt.ylabel('Option Price Estimate')
            plt.savefig(convergence_filename)
            plt.close()
            plot_filenames.append(convergence_filename)

        return np.exp(-self.r * self.T) * np.mean(payoffs), plot_filenames if output_folder is not None else None
    
    def machine_learning_predict_mid_price(self, csv_file, risk_free_rate=0.05):
        """
        Predict mid price using machine learning on the provided CSV data.
        """
        # Load the data
        data = pd.read_csv(csv_file)

        # Calculate mid price
        data['mid_price'] = (data['bid'] + data['ask']) / 2
        display(data.head())

        # data['T'] = (pd.to_datetime(data['expiration']) - pd.to_datetime(data['date'])).days / 365  # Time to maturity
        data['T'] = (pd.to_datetime(data['expiration']) - pd.to_datetime(data['date'])).dt.days / 365  # Time to maturity
        data['risk_free_rate'] = risk_free_rate
       # Select features and target
        features = data[['stock_price','strike', 'T', 'risk_free_rate','implied_volatility', 'call_put']]
        target = data['mid_price']


        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate the mean squared error
        mse = mean_squared_error(y_test, predictions)
        print(f"Machine Learning Model MSE: {mse}")

        print(X_test)

        # Return predictions and true values for comparison
        return predictions, y_test.values

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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
        Vectorised calculation of the option price using the Black-Scholes formula.
        """
        # Compute d1 and d2 using vectorised operations
        d1 = (np.log(self.S / self.K) + (self.r - q + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        # Calculate call and put prices for all rows
        call_prices = self.S * np.exp(-q * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        put_prices = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * np.exp(-q * self.T) * norm.cdf(-d1)

        # Use np.where to choose the price based on the option type ('call' or 'put')
        prices = np.where(self.option_type == 'call', call_prices, put_prices)

        return prices
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

        # Use np.where to create option values based on option_type
        option_values = np.where(
            self.option_type == 'call',
            np.maximum(0, ST - self.K),
            np.maximum(0, self.K - ST)
        )

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

    def new_monte_carlo_option_price(self, ticker=None, output_folder=None, num_simulations=10000):
        dt = 1 / 365  # Daily steps
        num_steps = np.array((self.T * 365).astype(int), ndmin=1)  # Ensure num_steps is always an array

        S_array = np.asarray(self.S).flatten()  # Convert to array and flatten
        payoffs = np.zeros(len(S_array))  # Array to store payoffs for each option
        paths = []  # List to store paths for each simulation

        # Loop through the number of simulations
        for _ in range(num_simulations):
            ST = np.copy(S_array)  # Current stock prices for this simulation
            path = [ST.copy()]  # Store the path for this simulation

            # Loop over each option's number of steps
            for idx in range(len(S_array)):
                current_steps = num_steps[idx]  # Access number of steps for the specific option
                for step in range(current_steps):  # Use current_steps for each specific option
                    Z = np.random.normal()  # Generate a random normal value for this step

                    ST[idx] *= np.exp((self.r - 0.5 * self.sigma[idx] ** 2) * dt + self.sigma[idx] * np.sqrt(dt) * Z)

            # Calculate payoffs for all options in this simulation
            for idx in range(len(ST)):
                # print(f"Payoff calculation for option {idx}, type: {self.option_type[idx]}")
                if self.option_type[idx] == 'call':
                    payoffs[idx] += max(0, ST[idx] - self.K[idx])
                else:
                    payoffs[idx] += max(0, self.K[idx] - ST[idx])

            # Store the final stock price for this option path
            path.append(ST.copy())
            paths.append(path)

        # Estimated prices calculation
        option_prices = np.exp(-self.r * np.asarray(self.T)) * (payoffs / num_simulations)

        return option_prices, paths if output_folder is not None else None
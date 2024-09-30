import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

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


# # // Single responbility 

#     class VolatilityType(Enum):
#         black_scholes = 3
#         binomial_tree = 4
#         monte_carlo = 5

#     def calculate_volatility(
#             type: VolatilityType, 
#             self, 
#             sigma, 
#             q=0,      
#     ):

#     # // Shared preparation for function 

#     # match type:

#     #     case VolatilityType.black_scholes:

#     #     case VolatilityType.binomial_tree:

#     #     case VolatilityType.monte_carlo

#     #     case _:

# import os
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.stats import norm
# from enum import Enum

# class ModelType(Enum):
#     BLACK_SCHOLES = 1
#     BINOMIAL_TREE = 2
#     MONTE_CARLO = 3


# class OptionPricingModels:
#     def __init__(self, S, K, T, r, sigma):
#         """
#         Constructor for the OptionPricingModels class.
#         """
#         self.S = S       # Current stock price
#         self.K = K       # Strike price
#         self.T = T       # Time to maturity (in years)
#         self.r = r       # Risk-free rate
#         self.sigma = sigma  # Volatility

#     def calculate_option_price(self, option_type, model_type: ModelType, **kwargs):
#         """
#         Main function to calculate option price based on the model type (Black-Scholes, Binomial Tree, Monte Carlo).
#         """
#         if model_type == ModelType.BLACK_SCHOLES:
#             return self.black_scholes_option(option_type, kwargs.get('q', 0))

#         elif model_type == ModelType.BINOMIAL_TREE:
#             return self.binomial_tree_option_price(option_type, kwargs['N'])

#         elif model_type == ModelType.MONTE_CARLO:
#             return self.monte_carlo_option_price(option_type, kwargs['num_simulations'], kwargs['ticker'], kwargs['output_folder'])

#         else:
#             raise ValueError("Invalid model_type. Use BLACK_SCHOLES, BINOMIAL_TREE, or MONTE_CARLO from ModelType Enum.")

#     def black_scholes_option(self, option_type, q=0):
#         """
#         Calculate the option price using the Black-Scholes formula.
#         """
#         d1 = (np.log(self.S / self.K) + (self.r - q + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
#         d2 = d1 - self.sigma * np.sqrt(self.T)

#         if option_type == 'call':
#             price = self.S * np.exp(-q * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
#         elif option_type == 'put':
#             price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * np.exp(-q * self.T) * norm.cdf(-d1)
#         else:
#             raise ValueError("option_type must be 'call' or 'put'")

#         return price

#     def binomial_tree_option_price(self, option_type, N):
#         """
#         Calculate option price using the Binomial Tree method.
#         """
#         dt = self.T / N  # Time step
#         u = np.exp(self.sigma * np.sqrt(dt))  # Up factor
#         d = 1 / u  # Down factor
#         p = (np.exp(self.r * dt) - d) / (u - d)  # Risk-neutral probability

#         # Initialise asset prices at maturity
#         ST = np.array([self.S * (u ** (N - i)) * (d ** i) for i in range(N + 1)])

#         # Initialise option values at maturity
#         option_values = np.maximum(0, ST - self.K) if option_type == 'call' else np.maximum(0, self.K - ST)

#         # Backward induction
#         for j in range(N - 1, -1, -1):
#             option_values = np.exp(-self.r * dt) * (p * option_values[:-1] + (1 - p) * option_values[1:])

#         return option_values[0]

#     def monte_carlo_option_price(self, option_type, num_simulations, ticker, output_folder):
#         """
#         Calculate option price using Monte Carlo simulation.
#         """
#         dt = 1 / 365  # Daily steps
#         num_steps = int(self.T * 365)  # Number of days until maturity
#         payoffs = []
#         paths = []
#         estimated_prices = []

#         for _ in range(num_simulations):
#             ST = self.S
#             path = [self.S]
#             for _ in range(num_steps):
#                 Z = np.random.normal()
#                 ST *= np.exp((self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z)
#                 path.append(ST)

#             paths.append(path)
#             payoff = max(0, ST - self.K) if option_type == 'call' else max(0, self.K - ST)
#             payoffs.append(payoff)
#             estimated_prices.append(np.exp(-self.r * self.T) * np.mean(payoffs))

#         # Set common styling for plots
#         plt.rcParams.update({
#             'font.size': 14,           # Font size
#             'lines.linewidth': 2,      # Line width
#             'figure.dpi': 300          # Image quality
#         })
#         # Plot stock price paths
#         path_filename = os.path.join(output_folder, 'Monte_Carlo_Paths.png')
#         plt.figure(figsize=(12, 6))
#         for path in paths[:200]:  # Limit to 200 paths for clarity
#             plt.plot(path, linewidth=0.5)
#         plt.title(f'Monte Carlo Simulation of Stock Price Paths ({ticker})')
#         plt.xlabel('Time Steps')
#         plt.ylabel('Stock Price')
#         plt.savefig(path_filename)
#         plt.close()

#         # Histogram of payoffs
#         payoff_filename = os.path.join(output_folder, 'Payoff_Histogram.png')
#         plt.figure(figsize=(12, 6))
#         plt.hist(payoffs, bins=50)
#         plt.title(f'Histogram of Simulated Payoffs ({ticker})')
#         plt.xlabel('Payoff')
#         plt.ylabel('Frequency')
#         plt.savefig(payoff_filename)
#         plt.close()

#         # Convergence plot
#         convergence_filename = os.path.join(output_folder, 'Convergence_Plot.png')
#         plt.figure(figsize=(12, 6))
#         plt.plot(estimated_prices, color='blue')
#         plt.title(f'Convergence of Monte Carlo Option Price ({ticker})')
#         plt.xlabel('Number of Simulations')
#         plt.ylabel('Option Price Estimate')
#         plt.savefig(convergence_filename)
#         plt.close()

#         plot_filenames = [path_filename, payoff_filename, convergence_filename]
#         return np.exp(-self.r * self.T) * np.mean(payoffs), plot_filenames

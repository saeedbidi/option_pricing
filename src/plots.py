import os
import matplotlib.pyplot as plt
import numpy as np
from models import OptionPricingModels


class OptionPlots:
    def __init__(self, option_type, ticker, output_folder):
        self.option_type = option_type
        self.ticker = ticker
        self.output_folder = output_folder

    def plot_option_price_vs_stock_price(self, S, K, T, r, sigma):

        S_range = np.linspace(S * 0.8, S * 1.2, 100)
        K_list = [K, K * 1.1, K * 0.9]
        # Set common styling for plots
        plt.rcParams.update({
            'font.size': 14,           
            'lines.linewidth': 2,      
            'figure.dpi': 300         
        })
        plt.figure(figsize=(12, 6))
        for K in K_list:
            option_prices = [OptionPricingModels(S, K, T, r, sigma, self.option_type).black_scholes_option(q=0) for S in S_range]
            plt.plot(S_range, option_prices, label=f"Strike Price {K:.1f}")

        plt.title(f"{self.option_type.capitalize()} Option Price vs Stock Price using Black Scholes model ({self.ticker})")
        plt.xlabel("Stock Price")
        plt.ylabel(f"{self.option_type.capitalize()} Option Price")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(self.output_folder, 'Option_price_vs_stock_price.png')
        plt.savefig(plot_path)
        # plt.show()

    def comparative_pricing_plot(self, bs_price, mc_price, bt_price):
        methods = ['Black-Scholes', 'Monte Carlo', 'Binomial Tree']
        prices = [bs_price, mc_price[0], bt_price]
        # Set common styling for plots
        plt.rcParams.update({
            'font.size': 14,           
            'lines.linewidth': 2,      
            'figure.dpi': 300          
        })
        plt.figure(figsize=(12, 6))
        plt.bar(methods, prices, color=['blue', 'red', 'black'])
        plt.title(f'Option Pricing: Black-Scholes vs Monte Carlo vs Binomial Tree ({self.ticker})')
        plt.ylabel('Option Price')
        plot_comparison = os.path.join(self.output_folder, 'Pricing_Comparison.png')
        plt.savefig(plot_comparison)
        # plt.show()

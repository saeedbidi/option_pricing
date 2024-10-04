import os
import numpy as np
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
from models import OptionPricingModels


class GreeksVolatility:
    def __init__(self, S, K, T, r, market_price, ticker, option_type, output_folder='output'):
        """
        Initialises the Greeks and volatility calculation class.

        Args:
            S (float): Current stock price.
            K (float): Strike price.
            T (float): Time to maturity in years.
            r (float): Risk-free rate.
            market_price (float): Market price of the option.
            ticker (str): Stock ticker symbol.
            option_type (str): Option type ('call' or 'put').
            output_folder (str): Directory to save output files.
        """
        self.S = S 
        self.K = K
        self.T = T
        self.r = r
        self.market_price = market_price
        self.ticker = ticker
        self.option_type = option_type
        self.output_folder = output_folder
        self.sigma = None  # Volatility

    def _greeks(self, sigma, q=0):
        """
        Calculate the Greeks for the option.

        Args:
            S (float): Current stock price.
            K (float): Strike price.
            T (float): Time to maturity in years.
            r (float): Risk-free rate.
            sigma (float): Volatility.
            q (float): Dividend rate. Default is 0.

        Returns:
            tuple: Delta, Gamma, Vega, Theta, and Rho.
        """

        d1 = (np.log(self.S / self.K) + (self.r - q + 0.5 * sigma ** 2) * self.T) / (sigma * np.sqrt(self.T))
        d2 = d1 - sigma * np.sqrt(self.T)

        if self.option_type == "call":
            delta = norm.cdf(d1)
            theta = (- (self.S * np.exp(-q * self.T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(self.T)) -
                     self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
            rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            delta = norm.cdf(d1) - 1
            theta = (- (self.S * np.exp(-q * self.T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(self.T)) +
                     self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2))
            rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)

        gamma = norm.pdf(d1) / (self.S * sigma * np.sqrt(self.T))
        vega = self.S * np.exp(-q * self.T) * norm.pdf(d1) * np.sqrt(self.T)

        return delta, gamma, vega, theta, rho

    def implied_volatility(self):
        """
        Calculate implied volatility using the market price.
        This method tries Newton-Raphson first, then Bisection, and finally Brent's method.

        Args:
            S (float): Current stock price.
            K (float): Strike price.
            T (float): Time to maturity in years.
            r (float): Risk-free rate.
            market_price (float): Market price of the option.

        Returns:
            float: Implied volatility.
            """
        try:
            sigma = self.implied_volatility_newton()
            if sigma is None:
                raise ValueError("Newton-Raphson method returned None for sigma.")
            print('implied volatility = ', sigma)
            return sigma
        except ValueError as e:
            print(f"Newton-Raphson method failed: {e}")
            print("Falling back to Bisection method.")

        try:
            print('implied volatility using bisection.')
            sigma = self.implied_volatility_bisection()
            return sigma
        except ValueError as e:
            print(f"Bisection method failed: {e}")
            print("Falling back to Brent's method.")

        return self.implied_volatility_brent()

    def implied_volatility_newton(self, max_iterations=10000, tolerance=1e-6, relaxation_factor=0.15):
        """
        Calculate implied volatility using the Newton-Raphson method.

        Args:
            S (float): Current stock price.
            K (float): Strike price.
            T (float): Time to maturity in years.
            r (float): Risk-free rate.
            market_price (float): Market price of the option.

        Returns:
            float: Implied volatility.
        """
        sigma = self.sigma if self.sigma is not None else 0.2  # Fallback if historical volatility is not calculated

        for _ in range(max_iterations):

            option_pricing = OptionPricingModels(self.S, self.K, self.T, self.r, sigma, self.option_type)
            price = option_pricing.black_scholes_option()
            vega = self._greeks(sigma)[2]
            price_diff = price - self.market_price


            if abs(price_diff) < tolerance:
                print(f"Newton-Raphson method converged! sigma = {sigma}")
                return sigma

            if vega == 0:
                raise ValueError("Vega is zero, cannot update volatility.")

            delta_sigma = relaxation_factor * (price_diff / vega)
            sigma -= delta_sigma
            sigma = max(0.01, min(5.0, sigma))  # Limit sigma
        print(f"Newton-Raphson method failed to converge after {max_iterations} iterations. {sigma}")
    

    def implied_volatility_bisection(self, lower_bound=0.01, upper_bound=5.0, max_iterations=100, tolerance=1e-6):
        
        sigma = self.sigma if self.sigma is not None else 0.2 
        
        def price_difference(sigma):
            option_pricing = OptionPricingModels(self.S, self.K, self.T, self.r, sigma, self.option_type)
            return option_pricing.black_scholes_option() - self.market_price

        f_lower = price_difference(lower_bound)
        f_upper = price_difference(upper_bound)
        if f_lower * f_upper > 0:
            raise ValueError("The function must have different signs at the lower and upper bounds.")

        for _ in range(max_iterations):
            mid = (lower_bound + upper_bound) / 2
            f_mid = price_difference(mid)

            if abs(f_mid) < tolerance:
                return mid

            if f_lower * f_mid < 0:
                upper_bound = mid
                f_upper = f_mid
            else:
                lower_bound = mid
                f_lower = f_mid

        raise ValueError("Implied volatility could not be found within the specified iterations.")

    def implied_volatility_brent(self):
        """
        Calculate implied volatility using Brent's method (brentq).

        Args:
            S (float): Current stock price.
            K (float): Strike price.
            T (float): Time to maturity in years.
            r (float): Risk-free rate.
            market_price (float): Market price of the option.

        Returns:
            float: Implied volatility.
        """
        sigma = self.sigma if self.sigma is not None else 0.2  # Fallback if historical volatility is not calculated
        option_pricing = OptionPricingModels(self.S, self.K, self.T, self.r, sigma, self.option_type)
        def option_price_diff(sigma):
            price = option_pricing.black_scholes_option()
            return price - self.market_price

        low = 1e-6
        high = 10
        low_value = option_price_diff(low)
        high_value = option_price_diff(high)

        if low_value * high_value > 0:
            print("Volatility couldn't be found: Function does not have opposite signs at the boundaries.")
            return None
        
        implied_vol = brentq(option_price_diff, low, high)
        return implied_vol

    

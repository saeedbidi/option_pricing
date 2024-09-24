import streamlit as st
import numpy as np
import yfinance as yf
from datetime import datetime
import os
import matplotlib.pyplot as plt
from option_pricing import OptionPricing 


# Streamlit app
def app():
    st.title("Option Pricing Models")

    # Input fields
    ticker = st.text_input("Enter stock ticker:", "AAPL")
    option_type = st.radio("Select option type:", ["Call", "Put"])
    K = st.number_input("Enter strike price (K):", value=207.5)
    days_to_maturity = st.number_input("Days to expiration:", value=7)
    T = days_to_maturity / 365  # Time to maturity (in years)
    r = st.number_input("Enter risk-free rate (r):", value=0.05)
    market_price = st.number_input("Enter market price of the option:", value=22.25)

    num_simulations = st.number_input("Number of simulations for Monte Carlo method (e.g., 100000):", value=10000)
    N = st.number_input("Number of steps for Binomial Tree method (e.g., 100):", value=100)

    # ticker = st.text_input("Enter stock ticker:", "TSLA")
    # option_type = st.radio("Select option type:", ["Call", "Put"])
    # K = st.number_input("Enter strike price (K):", value=225)
    # days_to_maturity = st.number_input("Days to expiration:", value=25)
    # T = days_to_maturity / 365  # Time to maturity (in years)
    # r = st.number_input("Enter risk-free rate (r):", value=0.05)
    # market_price = st.number_input("Enter market price of the option:", value=30.8)

    # Date inputs for historical volatility calculation
    start_date = st.date_input("Select start date for historical data:", datetime(2023, 1, 1))
    end_date = st.date_input("Select end date for historical data:", datetime.today())

    # Initialise the OptionPricing class
    option_pricing = OptionPricing(ticker, option_type.lower())  # Pass the type as lowercase

    # Calculate historical volatility
    sigma = option_pricing.calculate_historical_volatility(start_date.strftime('%Y-%m-%d'), 
                                                             end_date.strftime('%Y-%m-%d'))

    # Button to calculate all results
    if st.button("Calculate All Results"):
        # Fetch the current stock price
        option_pricing.get_stock_data()
        if option_pricing.S:
            st.success(f"Current Stock Price (S): {option_pricing.S:.2f}")
            option_pricing.output_folder = "output_streamlit"
            os.makedirs(option_pricing.output_folder, exist_ok=True)
            # Calculate prices using different models
            bs_price = option_pricing.black_scholes_option(option_pricing.S, K, T, r, sigma)
            mc_price = option_pricing.monte_carlo_option_price(option_pricing.S, K, T, r, sigma, num_simulations)
            bt_price = option_pricing.binomial_tree_option_price(option_pricing.S, K, T, r, sigma, N)

            st.success(f"Black-Scholes Price: {bs_price:.2f}")
            st.success(f"Monte Carlo Price: {mc_price[0]:.2f}")
            st.success(f"Binomial Tree Price: {bt_price:.2f}")

            # Generate comparative pricing plot
            option_pricing.comparative_pricing_plot(bs_price, mc_price, bt_price)
            # Ensure the output folder exists
            # Plot option prices vs stock price
            S_range = np.linspace(option_pricing.S * 0.8, option_pricing.S * 1.2, 100)
            K_list = [K, K * 1.1, K * 0.9]
            option_pricing.plot_option_price_vs_stock_price(S_range, K_list, T, r, sigma)

            # Calculate implied volatility
            iv = option_pricing.implied_volatility(option_pricing.S, K, T, r, market_price)
            if iv is not None:
                st.success(f"Implied Volatility: {iv:.2%}")
            else:
                st.error("Could not calculate implied volatility.")

            # Display plots
            if os.path.exists(os.path.join(option_pricing.output_folder, 'Pricing_Comparison.png')):
                st.image(os.path.join(option_pricing.output_folder, 'Pricing_Comparison.png'), caption='Option Pricing: Black-Scholes vs Monte Carlo vs Binomial Tree')
            if os.path.exists(os.path.join(option_pricing.output_folder, 'Convergence_Plot.png')):
                st.image(os.path.join(option_pricing.output_folder, 'Option_price_vs_stock_price.png'), caption='Option Price vs Stock Price')
            if os.path.exists(os.path.join(option_pricing.output_folder, 'Monte_Carlo_Paths.png')):
                st.image(os.path.join(option_pricing.output_folder, 'Monte_Carlo_Paths.png'), caption='Monte Carlo Simulation Paths')
            if os.path.exists(os.path.join(option_pricing.output_folder, 'Payoff_Histogram.png')):
                st.image(os.path.join(option_pricing.output_folder, 'Payoff_Histogram.png'), caption='Histogram of Simulated Payoffs')
            if os.path.exists(os.path.join(option_pricing.output_folder, 'Convergence_Plot.png')):
                st.image(os.path.join(option_pricing.output_folder, 'Convergence_Plot.png'), caption='Convergence of Monte Carlo Option Price')


        else:
            st.error("Error fetching stock price.")

# Streamlit call
if __name__ == "__main__":
    app()

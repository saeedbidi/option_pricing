import streamlit as st

st.title("Hello, Streamlit!")
st.write("This is a test to see if Streamlit is running correctly.")

# import streamlit as st
import numpy as np
import yfinance as yf
from scipy.stats import norm
from option_pricing import OptionPricing  # Import your class here

# Title of the app
st.title("Option Pricing Calculator")

# User inputs
ticker = st.text_input("Stock Ticker:", "AAPL")
option_type = st.selectbox("Option Type:", ["call", "put"])
strike_price = st.number_input("Strike Price:", value=207.5)
days_to_maturity = st.number_input("Days to Maturity:", value=7, min_value=1)
risk_free_rate = st.number_input("Risk-Free Rate (as a decimal):", value=0.05)
market_price = st.number_input("Market Price of the Option:", value=22.25)

# Button to calculate option price
if st.button("Calculate"):
    option = OptionPricing(ticker, option_type)
    option.get_stock_data()
    historical_vol = option.calculate_historical_volatility()
    S = option.S
    T = days_to_maturity / 365

    # Calculate option prices
    bs_price = option.black_scholes_option(S, strike_price, T, risk_free_rate, historical_vol)
    mc_price = option.monte_carlo_option_price(S, strike_price, T, risk_free_rate, historical_vol)
    bt_price = option.binomial_tree_option_price(S, strike_price, T, risk_free_rate, historical_vol, N=100)

    # Display results
    st.write(f"Black-Scholes Price: {bs_price:.2f}")
    st.write(f"Monte Carlo Price: {mc_price:.2f}")
    st.write(f"Binomial Tree Price: {bt_price:.2f}")

    # Option Greeks
    delta, gamma, vega, theta, rho = option.greeks(S, strike_price, T, risk_free_rate, historical_vol)
    st.write(f"Greeks: Delta={delta:.4f}, Gamma={gamma:.4f}, Vega={vega:.4f}, Theta={theta:.4f}, Rho={rho:.4f}")

    # Implied Volatility
    implied_vol = option.implied_volatility_newton(S, strike_price, T, risk_free_rate, market_price)
    st.write(f"Implied Volatility: {implied_vol:.2f}")

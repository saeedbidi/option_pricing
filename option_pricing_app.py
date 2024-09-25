import streamlit as st
import numpy as np
import yfinance as yf
from datetime import datetime
import os
import matplotlib.pyplot as plt
from option_pricing import OptionPricing 
import time

# Streamlit app
def app():
    st.set_page_config(
        page_title="Option Pricing Models",
        page_icon="📈",
        layout="centered",  # Makes use of the full screen width
        initial_sidebar_state="auto"
    )

    st.title("Option Pricing Models 📊 (saeed.bidi@qmul.ac.uk)")

    # Description of the app and models
    st.markdown("""
    Welcome to the **Option Pricing Models** app. This tool allows you to calculate option prices 
    using various financial models such as the Black-Scholes model, Monte Carlo simulations, and Binomial Tree models.
    
    You can enter key inputs like the stock ticker, strike price, risk-free rate, and time to maturity. 
    The app will also calculate implied volatility and provide you with a comparison between different pricing methods.
    """)
  # CSS styles for button
    st.markdown("""
    <style>
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
        margin: 10px 2px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)






    # Layout 1
    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input("Stock ticker:", "AAPL")
        option_type = st.radio("Option type:", ["Call", "Put"])

    with col2:
        K = st.number_input("Strike price (K):", value=207.5)
        days_to_maturity = st.number_input("Days to expiration:", value=7)
        T = days_to_maturity / 365
        r = st.number_input("Risk-free rate (r):", value=0.05)
        market_price = st.number_input("Market price of the option:", value=22.25)
      

    with col3:
        num_simulations = st.number_input("Monte Carlo runs (e.g., 100000):", value=10000)
        N = st.number_input("Binomial Tree steps (e.g., 100):", value=100)
        # Date inputs for historical volatility calculation
        start_date = st.date_input("Select start date for historical data:", datetime(2023, 1, 1))
        end_date = st.date_input("Select end date for historical data:", datetime.today())

    # Layout 2
    # with st.sidebar:
    #     st.header("Option Pricing Inputs")
    #     ticker = st.text_input("Stock ticker:", "AAPL")
    #     option_type = st.radio("Option type:", ["Call", "Put"])
    #     K = st.number_input("Strike price (K):", value=207.5)
    #     days_to_maturity = st.number_input("Days to expiration:", value=7)
    #     r = st.number_input("Risk-free rate (r):", value=0.05)
    #     market_price = st.number_input("Market price of the option:", value=22.25)
    #     num_simulations = st.number_input("Monte Carlo runs (e.g., 100000):", value=10000)
    #     N = st.number_input("Binomial Tree steps (e.g., 100):", value=100)
    #     start_date = st.date_input("Select start date for historical data:", datetime(2023, 1, 1))
    #     end_date = st.date_input("Select end date for historical data:", datetime.today())
    
    T = days_to_maturity / 365

    # Initialise the OptionPricing class
    option_pricing = OptionPricing(ticker, option_type.lower())  # Pass the type as lowercase

    # Calculate historical volatility
    sigma = option_pricing.calculate_historical_volatility(start_date.strftime('%Y-%m-%d'), 
                                                             end_date.strftime('%Y-%m-%d'))

    # Button to calculate all results
    if st.button("Calculate All Results 🚀"):
        with st.spinner('Calculating...'):
            # Code to calculate results
            time.sleep(2)
        st.info("Fetching stock data and performing calculations, this might take a few moments...")
        
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




            # Generate comparative pricing plot
            option_pricing.comparative_pricing_plot(bs_price, mc_price, bt_price)

            # Plot option prices vs stock price
            S_range = np.linspace(option_pricing.S * 0.8, option_pricing.S * 1.2, 100)
            K_list = [K, K * 1.1, K * 0.9]
            option_pricing.plot_option_price_vs_stock_price(S_range, K_list, T, r, sigma)

            # Calculate implied volatility
            iv = option_pricing.implied_volatility(option_pricing.S, K, T, r, market_price)
            # st.success("Calculation completed successfully! 🎉")

            # st.success(f"Black-Scholes Price Prediction: {bs_price:.2f}")
            # st.success(f"Monte Carlo Price Prediction: {mc_price[0]:.2f}")
            # st.success(f"Binomial Tree Price Prediction: {bt_price:.2f}")
            
            # Create a container for the price predictions
            with st.container():
                st.success("Price Predictions:")
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**Black-Scholes Price:** {bs_price:.2f}")
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**Monte Carlo Price:** {mc_price[0]:.2f}")
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**Binomial Tree Price:** {bt_price:.2f}")

            # After calculating bt_price, display the prediction

            if iv is not None:
                st.success(f"Implied Volatility: {iv:.2%}")
            else:
                st.error("Could not calculate implied volatility.")
        

            # Display plots
            if os.path.exists(os.path.join(option_pricing.output_folder, 'Convergence_Plot.png')):
                st.image(os.path.join(option_pricing.output_folder, 'Option_price_vs_stock_price.png'), caption='Option Price vs Stock Price')
            if os.path.exists(os.path.join(option_pricing.output_folder, 'Monte_Carlo_Paths.png')):
                st.image(os.path.join(option_pricing.output_folder, 'Monte_Carlo_Paths.png'), caption='Monte Carlo Simulation Paths')
            if os.path.exists(os.path.join(option_pricing.output_folder, 'Payoff_Histogram.png')):
                st.image(os.path.join(option_pricing.output_folder, 'Payoff_Histogram.png'), caption='Histogram of Simulated Payoffs')
            if os.path.exists(os.path.join(option_pricing.output_folder, 'Convergence_Plot.png')):
                st.image(os.path.join(option_pricing.output_folder, 'Convergence_Plot.png'), caption='Convergence of Monte Carlo Option Price')
            if os.path.exists(os.path.join(option_pricing.output_folder, 'Pricing_Comparison.png')):
                st.image(os.path.join(option_pricing.output_folder, 'Pricing_Comparison.png'), caption='Option Pricing: Black-Scholes vs Monte Carlo vs Binomial Tree')
 

        else:
            st.error("Error fetching stock price.")
    # Footer with credits and GitHub link
    st.markdown("---")
    st.markdown("""
    **Developed by Dr. Saeed Bidi**  
    [GitHub Repository](https://github.com/saeedbidi/option_pricing)
    
    This app is designed to help you understand the pricing of European call and put options using various financial models.
    The content and calculations provided are for educational purposes and should not be used for actual trading without further research.
    """)

# Streamlit call
if __name__ == "__main__":
    app()

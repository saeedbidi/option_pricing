import sys
import os

# Add the src folder to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import streamlit as st
import time
from datetime import datetime
import os

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from data import DataHandler
from models import OptionPricingModels
from plots import OptionPlots
from greeks_volatility import GreeksVolatility



# Streamlit app
def app():
    st.set_page_config(
        page_title="Option Pricing Models",
        page_icon="📈",
        layout="centered",  # Makes use of the full screen width
        initial_sidebar_state="auto"
    )

    st.title("Option Pricing Models 📊")
    # st.subsubheader("Contact: saeed.bidi@qmul.ac.uk 📧",)
    st.markdown("""
    **Developed by Saeed Bidi, PhD**""")  
    st.markdown("<span style='font-size: 0.9em;'>✉️ saeed.bidi@qmul.ac.uk</span>", unsafe_allow_html=True)

    # Description of the app and models
    st.markdown("""
    Welcome to my **Option Pricing Models** app. This tool allows you to calculate option prices 
    using various financial models such as Monte Carlo, Black-Scholes, and Binomial Tree models.
    
    You can enter key inputs like the stock ticker, strike price, risk-free rate, and time to maturity. 
    The app will also calculate implied volatility and provide you with a comparison between different pricing models.
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
        option_type = option_type.lower()

    with col2:
        K = st.number_input("Strike price:", value=207.5)
        days_to_maturity = st.number_input("Days to expiration:", value=7)
        # rfr = st.number_input("Risk-free rate (%):", value=5)
        rfr = st.number_input("Risk-free rate (%):", value=5, help="Annualised value.")
        market_price = st.number_input("Market price of the option:", value=22.25)
      

    with col3:
        num_simulations = st.number_input("Monte Carlo runs (e.g., 100000):", value=10000, help="Avoid a too large number as it increases the computational time. 10000 should be fine")
        N = st.number_input("Binomial Tree steps (e.g., 100):", value=100)
        # Date inputs for historical volatility calculation
        start_date = st.date_input("Select start date for historical data:", datetime(2023, 1, 1), help="The historical data is used to compute volatility")
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
    r = rfr/100.0
    # Initialise the OptionPricing class
    # option_pricing = OptionPricing(ticker, option_type.lower())  # Pass the type as lowercase
    
    # Use DataHandler to fetch stock data
    data_handler = DataHandler(ticker)
    
    # Calculate historical volatility

    sigma = data_handler.calculate_historical_volatility(start_date.strftime('%Y-%m-%d'), 
                                                             end_date.strftime('%Y-%m-%d'))

    # Button to calculate all results
    if st.button("Calculate All Results 🚀"):
        with st.spinner('Calculating...'):
            # Code to calculate results
            time.sleep(2)
        fetching_message = st.info("Fetching stock data and performing calculations, this might take a few moments...")
        
        # Fetch the current stock price
        data_handler.get_stock_data()
        fetching_message.empty()

        computing_message = st.info("Models started! Please wait ...")


        if data_handler.S:
            st.success(f"Current Stock Price: {data_handler.S:.2f} (USD) according to the live Yahoo market data")
            # Create output directories
            # current_dir = os.getcwd()

            OUTPUT_FOLDER = os.path.join('output', 'streamlit')
            models = OptionPricingModels(data_handler.S, K, T, r, sigma, option_type)
        
            # # Calculate Black Scholes price
            bs_price = models.black_scholes_option(q=0)


            # # Calculate Binomial Tree price
            bt_price = models.binomial_tree_option_price(N)

            # # Calculate Monte Carlo price
            mc_price = models.monte_carlo_option_price(ticker=ticker, output_folder=OUTPUT_FOLDER, num_simulations=num_simulations)
            # st.success("Calculation completed successfully! 🎉")
            with st.container():
                st.success("Price Predictions (in USD):")
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**Black-Scholes Price:** {bs_price:.2f}")
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**Monte Carlo Price:** {mc_price[0]:.2f}")
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**Binomial Tree Price:** {bt_price:.2f}")

            # Generate the range of stock prices for plotting

            S_range = np.linspace(data_handler.S * 0.8, data_handler.S * 1.2, 100)
            strike_prices = [K, K * 1.1, K * 0.9]  # List of different strike prices
            
            # Create a Plotly figure
            fig = go.Figure()

            # Calculate Black-Scholes prices for each strike price
            # for K_strike in strike_prices:
            #     bs_prices = [OptionPricingModels(data_handler.S, K, T, r, sigma, option_type).black_scholes_option(q=0) for S in S_range]
            #     fig.add_trace(go.Scatter(x=S_range, y=bs_prices, mode='lines', name=f"Strike Price: {K_strike:.2f}"))
            for K_strike in strike_prices:
                bs_prices = [OptionPricingModels(S, K_strike, T, r, sigma, option_type).black_scholes_option(q=0) for S in S_range]
                fig.add_trace(go.Scatter(x=S_range, y=bs_prices, mode='lines', name=f"Strike Price: {K_strike:.2f}"))

            # Customize the layout
            fig.update_layout(title="Black-Scholes Prices for Different Strike Prices",
                              xaxis_title="Stock Price (USD)",
                              yaxis_title="Option Price (USD)",
                              legend_title="Strike Prices")
            st.plotly_chart(fig)


            # Generate comparative pricing plot
            plots = OptionPlots(option_type, ticker, OUTPUT_FOLDER)
            plots.comparative_pricing_plot(bs_price, mc_price, bt_price)

            # Plot option prices vs stock price
            plots.plot_option_price_vs_stock_price(data_handler.S, K, T, r, sigma)
        
            # Calculate implied volatility
            greeks_volatility = GreeksVolatility(data_handler.S, K, T, r, market_price, ticker, option_type, OUTPUT_FOLDER)
            iv = greeks_volatility.implied_volatility()



            # After calculating bt_price, display the prediction

            if iv is not None:
                st.success(f"Implied Volatility: {iv:.2%}")
            # else:
            #     st.error("Could not calculate implied volatility.")
        

            # Display plots
            # if os.path.exists(os.path.join(option_pricing.output_folder, 'Convergence_Plot.png')):
            #     st.image(os.path.join(option_pricing.output_folder, 'Option_price_vs_stock_price.png'))
            # if os.path.exists(os.path.join(OUTPUT_FOLDER, 'Monte_Carlo_Paths.png')):
            #     st.image(os.path.join(OUTPUT_FOLDER, 'Monte_Carlo_Paths.png'))
            if os.path.exists(os.path.join(OUTPUT_FOLDER, 'Payoff_Histogram.png')):
                st.image(os.path.join(OUTPUT_FOLDER, 'Payoff_Histogram.png'))
            if os.path.exists(os.path.join(OUTPUT_FOLDER, 'Convergence_Plot.png')):
                st.image(os.path.join(OUTPUT_FOLDER, 'Convergence_Plot.png'))
            # if os.path.exists(os.path.join(option_pricing.output_folder, 'Pricing_Comparison.png')):
            #     st.image(os.path.join(option_pricing.output_folder, 'Pricing_Comparison.png'))
 

        else:
            st.error("Error fetching stock price.")
        
        computing_message.empty()
    # Footer with credits and GitHub link
    
    st.markdown("---")
    st.markdown("""
    [My LinkedIn](https://www.linkedin.com/in/saeed-bidi/)
                
    [GitHub Repository](https://github.com/saeedbidi/option_pricing)
    
    I designed this app to model the pricing of options using various financial models.
                
    **Machine learning** algorithms are also imelemented for conducting backtesting. Here is an example for AAPL option:
    """)
    # Use os.path.join to ensure the correct path is used
    image_path = os.path.join(os.getcwd(), 'output', 'backtesting', 'Backtesting_price_vs_date_plot.png')

    # Display the image
    st.image(image_path, use_column_width=True)
    
    st.markdown("""
        The content and calculations provided are for educational purposes and should not be used for actual trading without further research.
    """)

# Streamlit call
if __name__ == "__main__":
    app()

# **Project Overview**


## Quick Note:

📢 This app is now released and deployed at: https://saeed-option-pricing.streamlit.app/


🚀 I am excited to share my project focusing on pricing options. I implemented the **Black-Scholes model**, **Monte Carlo simulations**, and the **Binomial Tree method** for pricing options. I am further developing the project by adding backtesting, a machine learning model, and deploying the app.

🎉 I believe in hands-on learning. As a Senior Data Scientist with a PhD in Computational Physics, I have a strong foundation in analytical thinking and complex problem-solving. Through hands-on work in finance, I am leveraging my data science skills to gain practical experience and build a robust understanding of financial systems.

🛠️ Real-world stock data is fetched via the **Yahoo Finance API** to compute option prices, historical volatility, and option sensitivities (the Greeks). The project also provides visualisations and comparisons of option prices using all three methods.


##  💰 **Option Pricing Model with Black-Scholes, Monte Carlo and Binomial Tree Methods**

The **Black-Scholes model**, **Monte Carlo simulations**, and the **Binomial Tree method** are vital tools in financial markets for pricing options. Each method offers a distinct approach to estimating the price of call and put options, relying on parameters such as stock price, strike price, time to maturity, risk-free rate, and volatility.

### **Features**
- Fetches **real-time stock data** (e.g., Apple stock) using `yfinance`.
- Computes option prices for **call and put options** using:
  - **Black-Scholes**
  - **Monte Carlo**
  - **Binomial Tree**
- Calculates **historical volatility** based on daily returns.
- Computes the **Greeks**: **Delta**, **Gamma**, **Vega**, **Theta**, and **Rho**.
- Calculates **Implied Volatility** based on market prices.
- Provides visualisations showing how option prices vary with stock price for **both calls and puts**.
- **Compares** pricing results from the three methods.

###  📝 **Inputs**

1. **Stock Ticker**
2. **Stock Price (S)**
3. **Strike Price (K)**
4. **Days to Expiration**
5. **Risk-Free Rate (r)**
6. **Market Price of the Option**

###  📤 **Outputs**

The script will generate the following output:
- **options_report.txt**: a report file containing:
  - **Call and Put Option Prices** using **Black-Scholes**, **Monte Carlo**, and **Binomial Tree** methods.
  - **Greeks** and **implied volatility**.

- **Plots containing:**
  - **Comparing the predictions by different models**
  - **Option price vs stock price (using Black Scholes model)**
  - **Monte Carlo payoff histogram**
  - **Monte Carlo convergence plot**
  - **Monte Carlo paths**

#### **Sample report file**

```

  Options Pricing and Greeks Calculation Report

  1. User Inputs:
  - Option type: call
  - Stock Ticker: AAPL
  - Stock Price (S): 226.5
  - Strike Price (K): 207.5
  - Days to Expiration: 7 days
  - Risk-Free Rate (r): 5.00%
  - Market Price of the Option: 22.25
  - Start and end date for calculating Historical Volatility: 2023-01-01 - 2024-09-24


  2. Calculated Intermediate Values:
  - Time to Maturity (T): 0.0192 years
  - Historical Volatility (σ): 21.91%

  3. Option Prices:
  - Option Price (Black-Scholes): 19.17
  - Option Price (Monte Carlo): 19.24
  - Option Price (Binomial Tree): 19.17

  4. Greeks:
  - Delta: 0.9983
  - Gamma: 0.0008
  - Vega: 0.1710
  - Theta: -11.3226
  - Rho: 3.9682

  5. Implied Volatility Calculation:
  - Implied Volatility (IV): 82.16%
  
    
```
#### **Sample of Option vs Stock price plot using Black-Scholes model**

![Option_price_vs_stock](/output/Option_price_vs_stock_price.png)

#### **Sample of Monte Carlo paths and convergence plot**

![Monte_Carlo_Paths](/output/Monte_Carlo_Paths.png)
![Convergence_Plot](/output/Convergence_Plot.png)


## 📂 **Project Structure**

```
.
├── option_pricing.py        # Main Python script for the project
├── option_pricing_app.py    # Streamlit app
├── setup.py        # Setup Python script to install the required libraries
├── README.md                        # This README file
├── output                       # Output folder containing the reports and plots
```

## ⚙️ **Customisation**
You can customise this project by:
- Changing the **stock ticker** (e.g., from `AAPL` to `MSFT`).
- Modifying the **strike price**, **risk-free rate**, **time to maturity**, or **volatility**.
- Adjusting the number of **simulations** for the Monte Carlo method or the number of **steps** for the Binomial Tree method.


##  🎓 **How to Run the Project**

1. Clone the repository.
2. Install the following Python setup file including the necessary libraries:

```bash
python setup.py install
```
3. Execute the Python code `option_pricing.py` or the Jupyter Notebook script `option_pricing.ipynb`.


### **Parameters**
You can modify the following parameters in the script:
- **`ticker`**: Ticker symbol of the stock (e.g., `AAPL` for Apple).
- **`K`**: Strike price of the option.
- **`T`**: Time to maturity in years.
- **`r`**: Risk-free rate (annualised).
- **`mc_num_sim`**: Number of simulations for the Monte Carlo method.
- **`bt_num_step`**: Number of steps in the Binomial Tree method.


##  📖 **Code Explanation**

### **1. Black-Scholes Model**
The **Black-Scholes model** gives a closed-form solution for pricing call and put options, based on the assumption that stock prices follow a geometric Brownian motion.

#### **Formula**
For call options:
```math
C = S \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2).
```

For put options:
```math
P = K \cdot e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1),
```
where:

```math
d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma \sqrt{T}}, 
```

```math
 d_2 = d_1 - \sigma \sqrt{T}.
```

### **2. Monte Carlo Simulations**

Monte Carlo simulation is used to estimate the price of options by simulating the future stock price paths and calculating the corresponding option payoffs. The method relies on **Geometric Brownian Motion (GBM)**, which models stock price evolution as:

```math
S_{t+1} = S_t \cdot \exp \left( (r - 0.5 \cdot \sigma^2) \cdot \Delta t + \sigma \cdot \sqrt{\Delta t} \cdot Z \right),
```

where:
- $S_t$ is the stock price at time $t$.
- $r$ is the risk-free interest rate.
- $\sigma$ is the volatility of the stock.
- $\Delta t$ is the time increment (daily or another small step).
- $Z$ is a random variable from a standard normal distribution (representing the random walk).

For each simulation:
1. Stock price over time is simulated using the GBM formula.
2. Option **payoff** at maturity is calculated:
   - **Call Option**: $\max(0, S_T - K)$
   - **Put Option**: $\max(0, K - S_T)$,

where $S_T$ is the simulated stock price at maturity and $K$ is the strike price.

3. The payoff is discounted back to present value using the risk-free rate:
```math
\text{Option Price} = \exp(-r \cdot T) \cdot \text{Average Payoff},
```

where $T$ is the time to expiration in years.

This approach is flexible and can handle a wide variety of option types though it requires many simulations for accuracy.


### **3. Binomial Tree Method**
The Binomial Tree method approximates option prices by constructing a tree of possible future stock prices, calculating option prices by working backwards from the terminal nodes.

#### **Steps**
1. Time is divided to expiration into `n` intervals.
2. Each interval allows the stock price to increase by a factor `u` or decrease by a factor `d`.
3. Option prices are computed by working backwards using risk-neutral probabilities.

### **Greeks Explanation**
- **$\Delta$**: Sensitivity of the option price to changes in the underlying stock price.
- **$\Gamma$**: Rate of change of delta with respect to the underlying stock price.
- **$\nu$**: Sensitivity of the option price to changes in volatility.
- **$\Theta$**: Sensitivity of the option price to time decay.
- **$\rho$**: Sensitivity of the option price to changes in the risk-free interest rate.

### **Implied Volatility Calculation**
Implied volatility is derived from the Black-Scholes model to match the market price of an option, calculated through trying **Newton-Raphson** first, then **Bisection**, and finally **Brent's method**. They usually converge to the same solution.

### ⚖️ **Comparison of Methods**
All methods provide alternative means to calculate option prices, especially for options with complex features. Why not running the code and observe the comparison plot?


##  📚 **References**
- [Black-Scholes Model - Investopedia](https://www.investopedia.com/terms/b/blackscholes.asp)
- [Monte Carlo Methods - Investopedia](https://www.investopedia.com/terms/m/montecarlosimulation.asp)
- [Binomial Tree Model - Investopedia](https://www.investopedia.com/terms/b/binomialoptionpricing.asp)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)

## 📜 **License**
This project is open-source and available under the [MIT License](LICENSE).
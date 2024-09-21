# **Option Pricing Model with Black-Scholes, Monte Carlo and Binomial Tree Methods**

This project implements the **Black-Scholes model** for pricing European call and put options, along with additional functionality for **Monte Carlo simulations** and the **Binomial Tree method**. Real-world stock data is fetched via the **Yahoo Finance API** to compute option prices, historical volatility, and option sensitivities (the Greeks). The project also provides visualizations and comparisons of option prices using all three methods.

## **Project Overview**

The **Black-Scholes model**, **Monte Carlo simulations**, and the **Binomial Tree method** are vital tools in financial markets for pricing options. Each method offers a distinct approach to estimating the price of European-style options, relying on parameters such as stock price, strike price, time to maturity, risk-free rate, and volatility.

### **Features**
- Fetches **real-time stock data** (e.g., Apple stock) using `yfinance`.
- Computes option prices for **European call and put options** using:
  - **Black-Scholes**
  - **Monte Carlo**
  - **Binomial Tree**
- Calculates **historical volatility** based on daily returns.
- Computes the **Greeks**: **Delta**, **Gamma**, **Vega**, **Theta**, and **Rho**.
- Calculates **Implied Volatility** based on market prices.
- Provides visualizations showing how option prices vary with stock price for both calls and puts.
- Compares pricing results from the three methods.

### **Outputs**
1. **Call and Put Option Prices** using the three pricing methods.
2. **Greeks** for the call and put options.
3. **Implied Volatility** of the option.
4. **Plots** illustrating price changes with respect to the underlying stock price.
5. **Comparison** of option prices from all three methods.

## **Installation**

To run this project, install the following Python libraries:

```bash
pip install numpy scipy yfinance matplotlib
```

## **How to Run the Project**

1. Clone the repository or download the Python script.
2. Install the required dependencies listed above.
3. Execute the script in your preferred Python environment (e.g., Jupyter Notebook, VS Code, or command line):

```bash
python option_pricing_project.py
```

### **Example Output**
The script will generate the following output:
- **Call and Put Option Prices** using **Black-Scholes**, **Monte Carlo**, and **Binomial Tree** methods.
- The computed **Greeks** and **implied volatility**.
- Two plots:
  - **Call Option Price vs. Stock Price**
  - **Put Option Price vs. Stock Price**

### **Parameters**
You can modify the following parameters in the script:
- **`ticker`**: Ticker symbol of the stock (e.g., `AAPL` for Apple).
- **`K`**: Strike price of the option.
- **`T`**: Time to maturity in years.
- **`r`**: Risk-free rate (annualized).
- **`n_simulations`**: Number of simulations for the Monte Carlo method.
- **`n_steps`**: Number of steps in the Binomial Tree method.

### **Sample Output**

```
Option Pricing Results:
----------------------------------
Method: Black-Scholes
Call Option Price: 6.50
Put Option Price: 4.30

Method: Monte Carlo (100,000 simulations)
Call Option Price: 6.48
Put Option Price: 4.33

Method: Binomial Tree (500 steps)
Call Option Price: 6.52
Put Option Price: 4.31

Greeks (Black-Scholes):
Delta: 0.63
Gamma: 0.05
Vega: 0.12
Theta: -0.02
Rho: 0.25

Implied Volatility: 21.45%
```

### **Comparison**
The script compares prices from the **Black-Scholes**, **Monte Carlo**, and **Binomial Tree** methods, showcasing their alignment or divergence under varying market conditions and parameters like volatility and time to maturity.

## **Code Explanation**

### **1. Black-Scholes Model**
The **Black-Scholes model** offers a closed-form solution for pricing European-style options, based on the assumption that stock prices follow a geometric Brownian motion.

#### **Formula**
For call options:
\[ C = S \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2) \]

For put options:
\[ P = K \cdot e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1) \]

Where:
\[ d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma \sqrt{T}} \]
\[ d_2 = d_1 - \sigma \sqrt{T} \]

### **2. Monte Carlo Simulations**
Here's a concise explanation with equations for your README file:

---

### **Monte Carlo Simulation for Option Pricing**

Monte Carlo simulation is used to estimate the price of options by simulating the future stock price paths and calculating the corresponding option payoffs. The method relies on **Geometric Brownian Motion (GBM)**, which models stock price evolution as:

\[
S_{t+1} = S_t \cdot \exp \left( (r - 0.5 \cdot \sigma^2) \cdot \Delta t + \sigma \cdot \sqrt{\Delta t} \cdot Z \right)
\]

Where:
- \( S_t \) is the stock price at time \( t \).
- \( r \) is the risk-free interest rate.
- \( \sigma \) is the volatility of the stock.
- \( \Delta t \) is the time increment (daily or another small step).
- \( Z \) is a random variable from a standard normal distribution (representing the random walk).

For each simulation:
1. Simulate the stock price over time using the GBM formula.
2. Calculate the option **payoff** at maturity:
   - **Call Option**: \( \max(0, S_T - K) \)
   - **Put Option**: \( \max(0, K - S_T) \)

Where \( S_T \) is the simulated stock price at maturity and \( K \) is the strike price.

3. Discount the payoff back to present value using the risk-free rate:
\[
\text{Option Price} = \exp(-r \cdot T) \cdot \text{Average Payoff}
\]

Where \( T \) is the time to expiration in years.

This approach is flexible and can handle a wide variety of option types, but it requires many simulations for accuracy.

--- 

This explanation should fit neatly into your README file.

### **3. Binomial Tree Method**
The Binomial Tree method approximates option prices by constructing a tree of possible future stock prices, calculating option prices by working backwards from the terminal nodes.

#### **Steps**
1. Divide the time to expiration into `n` intervals.
2. Each interval allows the stock price to increase by a factor `u` or decrease by a factor `d`.
3. Compute option prices by working backwards using risk-neutral probabilities.

### **4. Greeks Explanation**
- **Delta (Δ)**: Sensitivity of the option price to changes in the underlying stock price.
- **Gamma (Γ)**: Rate of change of delta with respect to the underlying stock price.
- **Vega (ν)**: Sensitivity of the option price to changes in volatility.
- **Theta (Θ)**: Sensitivity of the option price to time decay.
- **Rho (ρ)**: Sensitivity of the option price to changes in the risk-free interest rate.

### **5. Implied Volatility Calculation**
Implied volatility is derived from the Black-Scholes model to match the market price of an option, solved iteratively using **Brent's method**.

### **6. Comparison of Methods**
Both Monte Carlo and Binomial Tree methods provide alternative means to calculate option prices, especially for options with complex features.

## **Project Structure**

```
.
├── option_pricing_project.py        # Main Python script for the project
├── README.md                        # This README file
```

## **Customization**
You can customize this project by:
- Changing the **stock ticker** (e.g., from `AAPL` to `MSFT`).
- Modifying the **strike price**, **risk-free rate**, **time to maturity**, or **volatility**.
- Adjusting the number of **simulations** for the Monte Carlo method or the number of **steps** for the Binomial Tree method.

## **References**
- [Black-Scholes Model - Investopedia](https://www.investopedia.com/terms/b/blackscholes.asp)
- [Monte Carlo Methods - Wikipedia](https://en.wikipedia.org/wiki/Monte_Carlo_method)
- [Binomial Tree Model - Wikipedia](https://en.wikipedia.org/wiki/Binomial_options_pricing_model)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)

## **License**
This project is open-source and available under the [MIT License](LICENSE).

---

This revised `README.md` provides a clear, structured overview of the project, ensuring that users can easily understand and navigate through the information.

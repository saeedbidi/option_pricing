### **Options Overview**

**What is an Option?**
An **option** is a financial contract that gives the buyer the right, but not the obligation, to buy or sell an asset (like a stock) at a specified price (the **strike price**) before or on a certain date (the **expiration date**). There are two main types of options:

- **Call Option**: Grants the right to **buy** the asset at the strike price.
- **Put Option**: Grants the right to **sell** the asset at the strike price.

**Example**:
- If you buy a **call option** on Apple (AAPL) with a strike price of \$150, you can purchase AAPL at \$150, even if the market price rises to \$200.
- Conversely, buying a **put option** at the same strike price allows you to sell AAPL at \$150, even if the price drops to \$100.

---

### 1. Brownian Motion
**Explanation**: Brownian motion is a mathematical model used to describe random motion. In finance, it represents the unpredictable path of asset prices over time, akin to a pollen grain moving randomly in water. The movement is influenced by small, random shocks or changes in the market.

**Geometric Brownian Motion (GBM)** is used to model stock prices because it incorporates both the random nature of price movements and the tendency of prices to grow over time. The formula is:

\[
S(t) = S(0) \cdot e^{\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W(t)}
\]

- **Where**:
  - \(S(t)\) = stock price at time \(t\)
  - \(S(0)\) = initial stock price
  - \(\mu\) = expected return (drift)
  - \(\sigma\) = volatility (standard deviation of returns)
  - \(W(t)\) = standard Brownian motion (random component)

### Example Using GBM
Let’s assume:
- Current stock price \(S(0) = 100\)
- Expected return \(\mu = 0.08\) (8% annual return)
- Volatility \(\sigma = 0.2\) (20% annual volatility)
- Time \(t = 1\) year

**Calculating Expected Return (Drift)**:
The expected return (\(\mu\)) can be estimated using historical data. Here’s how:
1. **Gather historical price data** for the stock over a specific period.
2. **Calculate returns** for each period (daily, monthly, etc.):
   \[
   \text{Return} = \frac{P(t) - P(t-1)}{P(t-1)}
   \]
   Where \(P(t)\) is the price at time \(t\).
3. **Calculate the average return** over the period:
   \[
   \mu = \frac{1}{n} \sum_{i=1}^{n} \text{Return}_i
   \]
   Where \(n\) is the number of periods.

Using this method, if you find an average return of 8% over the past five years, you can use this as your expected return.

### Expanded Explanation of Each Component

#### Risk-Free Rate
The **risk-free rate** is the return expected from an investment with zero risk, typically represented by government bonds. For example, if you buy a U.S. Treasury bond that offers a 3% annual return, this is your risk-free rate. It serves as a benchmark for evaluating investment performance, indicating what investors could earn without taking risks.

#### Risk-Neutral
In finance, a **risk-neutral** investor makes decisions based solely on expected returns, ignoring risk. This means they would be indifferent between receiving a certain return or a risky investment with the same expected return. For example, if given a choice between a guaranteed $100 or a 50% chance of winning $200 (with a 50% chance of winning nothing), a risk-neutral investor would view both options as equivalent since both have the same expected value of $100.

#### Discount
**Discounting** refers to the process of determining the present value of future cash flows. It accounts for the time value of money—the principle that a dollar today is worth more than a dollar tomorrow due to its potential earning capacity. The formula for discounting a future cash flow \(C\) at a rate \(r\) over time \(t\) is:

\[
PV = \frac{C}{(1 + r)^t}
\]

Where \(PV\) is the present value. For instance, if you expect to receive $100 in one year and the discount rate is 5%, the present value would be:

\[
PV = \frac{100}{(1 + 0.05)^1} \approx 95.24
\]

#### Payoff
**Payoff** is the return from an option at expiration, based on the relationship between the stock price and the strike price. For example, consider a call option with a strike price of $100. If the stock price at expiration is $120, the payoff from exercising the option is:

\[
\text{Payoff} = S(t) - K = 120 - 100 = 20
\]

If the stock price is below $100 at expiration, the option is not exercised, resulting in a payoff of $0.

### Summary of GBM Example
Let’s summarize the GBM example using all components:

1. **Assume**:
   - Current stock price \(S(0) = 100\)
   - Expected return \(\mu = 0.08\)
   - Volatility \(\sigma = 0.2\)
   - Time \(t = 1\) year
   - Hypothetical \(W(1) \approx 0.5\)

2. **Calculating**:
   - **Calculate drift**:
     \[
     \left(\mu - \frac{\sigma^2}{2}\right)t = \left(0.08 - 0.02\right) \cdot 1 = 0.06
     \]
   - **Final stock price**:
     \[
     S(1) = 100 \cdot e^{(0.06 + 0.1)} = 100 \cdot e^{0.16} \approx 117.30
     \]

In this model, after one year, the estimated stock price would be approximately **$117.30**.

---

### **Example Calculation of a Put Option**

1. **Parameters**:
   - Current stock price (S): \$150
   - Strike price (K): \$160
   - Premium paid for the put option: \$5

2. **Scenario**: If the stock price falls to \$140:
   - You can exercise your option to sell at \$160 while buying back at \$140.
   - **Profit Calculation**:
     \[
     \text{Profit per Share} = \text{Sell Price} - \text{Buy Price} = 160 - 140 = 20
     \]
     \[
     \text{Net Profit} = 20 - 5 = 15 \text{ per share}
     \]

3. **Alternate Scenario**: If the stock price rises to \$170:
   - You may choose to hold your put option and wait for a potential drop below the strike price before expiration.

---

### **Understanding Option Premiums**

**Breakdown of an Option's Premium**:
The premium consists of two components: **intrinsic value** and **time value**.

#### **1. Intrinsic Value**:
- **For a Call Option**:
  \[
  \text{Intrinsic Value (Call)} = \max(0, S - K)
  \]
- **For a Put Option**:
  \[
  \text{Intrinsic Value (Put)} = \max(0, K - S)
  \]

#### **2. Time Value**:
\[
\text{Time Value} = \text{Option Premium} - \text{Intrinsic Value}
\]
- **Time value** reflects potential future price movements based on time to expiration and market conditions.

---

### **Example Calculation of Call Option Value**

1. **Parameters**:
   - Stock price (S): \$220
   - Strike price (K): \$200
   - Call option premium: \$30

2. **Calculations**:
   - **Intrinsic Value**:
     \[
     \text{Intrinsic Value} = S - K = 220 - 200 = 20
     \]
   - **Time Value**:
     \[
     \text{Time Value} = 30 - 20 = 10
     \]

---

### **European vs. American Options**

- **European Option**: Can only be exercised on the **expiration date**.
- **American Option**: Can be exercised at any time before expiration.

This project will focus on **European options**.

---

### **The Black-Scholes Model**

The **Black-Scholes Model** is used to price European-style options, considering factors such as:
- **Stock Price (S)**
- **Strike Price (K)**
- **Time to Maturity (T)**
- **Risk-Free Interest Rate (r)**
- **Volatility (σ)**

This model provides a **theoretical price** for call and put options, assisting traders in understanding option value based on current market conditions.

---

Here’s the revised section on the Greeks, now with more detailed explanations and equations:

---

### **The Greeks: Understanding Risk**

The **Greeks** measure how sensitive an option's price is to various factors. Each Greek provides insights into different types of risk associated with options:

1. **Delta (Δ)**:
   - **Definition**: Measures the sensitivity of the option's price to changes in the stock price.
   - **Equation**: 
     \[
     \Delta = \frac{\partial C}{\partial S}
     \]
     Where \(C\) is the price of the call option and \(S\) is the stock price.
   - **Interpretation**: A Delta of 0.6 suggests that for every \$1 increase in the stock price, the option price will increase by \$0.60. Additionally, a Delta of 0.6 implies a 60% probability that the option will finish in the money at expiration.

2. **Gamma (Γ)**:
   - **Definition**: Measures the rate of change of Delta as the stock price changes.
   - **Equation**: 
     \[
     \Gamma = \frac{\partial^2 C}{\partial S^2}
     \]
   - **Interpretation**: A high Gamma indicates that Delta can change significantly with small movements in the stock price, which is important for managing risk.

3. **Vega (ν)**:
   - **Definition**: Measures the sensitivity of the option's price to changes in the volatility of the underlying asset.
   - **Equation**: 
     \[
     \nu = \frac{\partial C}{\partial \sigma}
     \]
   - **Interpretation**: A high Vega means the option price is very sensitive to changes in volatility. For example, a Vega of 0.5 indicates that a 1% increase in volatility will increase the option price by \$0.50.

4. **Theta (Θ)**:
   - **Definition**: Measures the rate of decline in the option's price as time passes, commonly referred to as time decay.
   - **Equation**: 
     \[
     \Theta = \frac{\partial C}{\partial t}
     \]
     Where \(t\) represents time to expiration.
   - **Interpretation**: A Theta of -0.05 implies that the option's price will decrease by \$0.05 per day as expiration approaches, assuming all other factors remain constant.

5. **Rho (ρ)**:
   - **Definition**: Measures the sensitivity of the option's price to changes in the risk-free interest rate.
   - **Equation**: 
     \[
     \rho = \frac{\partial C}{\partial r}
     \]
   - **Interpretation**: A Rho of 0.1 suggests that if the risk-free interest rate increases by 1%, the option price will increase by \$0.10. Rho is generally more relevant for long-term options.

---

### **Key Terms and Concepts**

- **Stock Price (S)**: Current market price of the stock.
- **Strike Price (K)**: Price at which the option can be exercised.
- **Time to Maturity (T)**: Time remaining until expiration, measured in years.
- **Volatility (σ)**: Measure of the stock's price fluctuations.
- **Risk-Free Interest Rate (r)**: Return rate on a risk-free investment.

---

### **Project Goals**

This project aims to develop a **quantitative tool** that can:
1. Fetch real-world stock data using the **Yahoo Finance API**.
2. Calculate theoretical option prices using the **Black-Scholes model**.
3. Evaluate sensitivity through the **Greeks** (e.g., Delta).
4. Visualize the relationship between stock prices and option prices.

### **Yahoo Finance API**

The **Yahoo Finance API** allows access to historical and real-time financial data, including stock prices and trading volumes. The **yfinance** library will be utilized to gather data for our calculations.

---

### **Alternative Models: Binomial Tree**

The **Binomial Tree Model** is a versatile option pricing method that accommodates American options and simulates price changes over discrete intervals.

---


### 1) When to Use Each Pricing Model

- **Black-Scholes Model**:
  - **Use When**: Pricing European options, especially when the underlying asset follows a geometric Brownian motion (i.e., continuous price movements).
  - **Advantages**: Fast and provides a closed-form solution. Ideal for options with no dividends.
  - **Limitations**: Assumes constant volatility and interest rates, and cannot price American options or options with path-dependent features.

- **Binomial Tree Model**:
  - **Use When**: Pricing American options or options with varying conditions, like dividends. Also suitable for simpler structures and shorter time frames.
  - **Advantages**: More flexible than Black-Scholes; can model changing volatility and can handle American options effectively.
  - **Limitations**: Computationally intensive for a large number of steps, leading to longer calculation times.

- **Monte Carlo Simulation**:
  - **Use When**: Pricing complex options, such as path-dependent options (e.g., Asian options) or when dealing with high-dimensional problems.
  - **Advantages**: Can handle a wide range of options and can model complicated payoffs.
  - **Limitations**: Slower convergence and requires a large number of simulations for accuracy, making it less efficient for simpler options.

---

### 2) Example: Pricing an Option Using Monte Carlo Simulation

**Scenario**: Price a European Call Option on a stock with the following parameters:
- Current stock price (S): $100
- Strike price (K): $100
- Time to expiration (T): 1 year
- Risk-free interest rate (r): 5%
- Volatility (σ): 20%
- Number of simulations: 10,000

**Steps**:
1. **Simulate Stock Prices**:
   - Use the formula for stock price at expiration:
     \[
     S_T = S \cdot e^{(r - \frac{\sigma^2}{2})T + \sigma \sqrt{T} Z}
     \]
     where \(Z\) is a random variable from a standard normal distribution.

2. **Calculate Payoff**:
   - For each simulated stock price, calculate the payoff:
     \[
     \text{Payoff} = \max(0, S_T - K)
     \]

3. **Calculate Present Value**:
   - Discount the average payoff back to present value:
     \[
     \text{Option Price} = e^{-rT} \cdot \frac{1}{N} \sum_{i=1}^{N} \text{Payoff}_i
     \]

**Implementation** (in pseudocode):
```python
import numpy as np

# Parameters
S = 100
K = 100
T = 1
r = 0.05
σ = 0.20
N = 10000  # number of simulations

# Simulate stock prices at expiration
Z = np.random.normal(0, 1, N)
S_T = S * np.exp((r - 0.5 * σ**2) * T + σ * np.sqrt(T) * Z)

# Calculate payoffs
payoffs = np.maximum(0, S_T - K)

# Calculate present value of the option
option_price = np.exp(-r * T) * np.mean(payoffs)

print(f"Monte Carlo Option Price: {option_price:.2f}")
```

---

### 3) Example: Pricing an Option Using the Binomial Tree Model

**Scenario**: Price a European Call Option with the following parameters:
- Current stock price (S): $100
- Strike price (K): $100
- Time to expiration (T): 1 year
- Risk-free interest rate (r): 5%
- Volatility (σ): 20%
- Number of steps (N): 3

**Steps**:
1. **Set Up Parameters**:
   - Calculate \(Δt = \frac{T}{N}\)
   - Calculate \(u = e^{σ\sqrt{Δt}}\) (up factor) and \(d = \frac{1}{u}\) (down factor)
   - Calculate \(p = \frac{e^{rΔt} - d}{u - d}\) (risk-neutral probability)

2. **Build the Stock Price Tree**:
   - Calculate stock prices at each node in the tree.

3. **Calculate Option Values at Expiration**:
   - For each terminal node, calculate the option payoff:
     \[
     C_T = \max(0, S_T - K)
     \]

4. **Work Backward to Find Option Price**:
   - Use the risk-neutral probabilities to calculate the option price at each preceding node:
     \[
     C_i = e^{-rΔt} (pC_{i+1} + (1-p)C_{i})
     \]

**Implementation** (in pseudocode):
```python
import numpy as np

# Parameters
S = 100
K = 100
T = 1
r = 0.05
σ = 0.20
N = 3  # number of steps

# Calculate parameters
Δt = T / N
u = np.exp(σ * np.sqrt(Δt))
d = 1 / u
p = (np.exp(r * Δt) - d) / (u - d)

# Initialize stock price tree
stock_prices = np.zeros((N + 1, N + 1))
for i in range(N + 1):
    for j in range(i + 1):
        stock_prices[j, i] = S * (u ** (i - j)) * (d ** j)

# Initialize option values at expiration
option_values = np.zeros((N + 1))
for j in range(N + 1):
    option_values[j] = max(0, stock_prices[j, N] - K)

# Backward induction to get the option price
for i in range(N - 1, -1, -1):
    for j in range(i + 1):
        option_values[j] = np.exp(-r * Δt) * (p * option_values[j] + (1 - p) * option_values[j + 1])

print(f"Binomial Tree Option Price: {option_values[0]:.2f}")
```

---

### **Conclusion**

This project will enhance your understanding of options pricing, the application of quantitative finance tools, and the ability to analyze real market data while visualizing key relationships.

---
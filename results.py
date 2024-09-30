import os

class ResultsHandler:
    def __init__(self, ticker, output_folder, S, K, T, r, sigma, option_type, delta, gamma_val, vega_val, theta_val, rho_val, iv, market_price, bs_price, mc_price, bt_price, start_date_volatility, end_date_volatility):
        self.ticker = ticker
        self.output_folder = output_folder
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.delta = delta
        self.gamma_val = gamma_val
        self.vega_val = vega_val
        self.theta_val = theta_val
        self.rho_val = rho_val
        self.iv = iv
        self.market_price = market_price
        self.bs_price = bs_price
        self.mc_price = mc_price
        self.bt_price = bt_price
        self.start_date_volatility = start_date_volatility
        self.end_date_volatility = end_date_volatility

    def generate_report(self):
        
        os.makedirs(self.output_folder, exist_ok=True)
       
        report = f"""
        Options Pricing and Greeks Calculation Report

        1. User Inputs:
        - Option type: {self.option_type}
        - Stock Ticker: {self.ticker}
        - Stock Price (S): {self.S:.1f}
        - Strike Price (K): {self.K}
        - Days to Expiration: {int(self.T * 365)} days
        - Risk-Free Rate (r): {self.r * 100:.2f}%
        - Market Price of the Option: {self.market_price}
        - Start and end date for calculating Historical Volatility: {self.start_date_volatility} - {self.end_date_volatility}

        2. Calculated Intermediate Values:
        - Time to Maturity (T): {self.T:.4f} years
        - Historical Volatility (σ): {self.sigma * 100:.2f}%

        3. Option Prices:
        - Option Price (Black-Scholes): {self.bs_price:.2f} 
        - Option Price (Monte Carlo): {self.mc_price:.2f}
        - Option Price (Binomial Tree): {self.bt_price:.2f} 

        4. Greeks:
        - Delta: {self.delta:.4f}
        - Gamma: {self.gamma_val:.4f}
        - Vega: {self.vega_val:.4f}
        - Theta: {self.theta_val:.4f}
        - Rho: {self.rho_val:.4f}

        5. Implied Volatility Calculation:
        - Implied Volatility (IV): {'Not found (check the imported market price)' if self.iv is None else f'{self.iv * 100:.2f}%'}
        """
        
        report_path = os.path.join(self.output_folder, "options_report.txt")

        with open(report_path, "w") as file:
            file.write(report)

        print("Report saved to options_report.txt")

    # You can add more result-related methods here

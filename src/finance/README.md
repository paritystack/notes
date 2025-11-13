# Finance

## Overview

Finance is the management of money, investments, and other financial instruments. This guide covers various aspects of financial markets, investment strategies, and trading concepts essential for understanding modern finance and making informed investment decisions.

## What is Finance?

Finance encompasses the creation, management, and study of money, banking, credit, investments, assets, and liabilities. It involves:

- **Personal Finance**: Managing individual/household money
- **Corporate Finance**: Managing business finances
- **Public Finance**: Government revenue and expenditure
- **Investment Finance**: Growing wealth through financial instruments

## Financial Markets

### Market Types

1. **Stock Market**: Equity securities (shares of companies)
2. **Bond Market**: Debt securities (loans to companies/governments)
3. **Commodity Market**: Physical goods (gold, oil, agricultural products)
4. **Forex Market**: Currency exchange
5. **Derivatives Market**: Contracts based on underlying assets

### Market Participants

- **Retail Investors**: Individual investors
- **Institutional Investors**: Banks, hedge funds, pension funds
- **Market Makers**: Provide liquidity
- **Brokers**: Execute trades on behalf of clients
- **Regulators**: Ensure fair and orderly markets

## Investment Instruments

### 1. Stocks (Equities)

Ownership shares in a company.

**Types:**
- **Common Stock**: Voting rights, dividends
- **Preferred Stock**: Fixed dividends, priority over common

**Metrics:**
- **Price-to-Earnings (P/E) Ratio**: Stock price / Earnings per share
- **Dividend Yield**: Annual dividend / Stock price
- **Market Capitalization**: Share price × Shares outstanding

```python
# Calculate basic stock metrics
def calculate_pe_ratio(price, earnings_per_share):
    """Price-to-Earnings Ratio"""
    return price / earnings_per_share

def calculate_dividend_yield(annual_dividend, stock_price):
    """Dividend Yield as percentage"""
    return (annual_dividend / stock_price) * 100

def calculate_market_cap(price, shares_outstanding):
    """Market Capitalization"""
    return price * shares_outstanding

# Example
stock_price = 150.00
eps = 10.00
annual_dividend = 3.00
shares = 1_000_000_000

pe_ratio = calculate_pe_ratio(stock_price, eps)
dividend_yield = calculate_dividend_yield(annual_dividend, stock_price)
market_cap = calculate_market_cap(stock_price, shares)

print(f"P/E Ratio: {pe_ratio:.2f}")
print(f"Dividend Yield: {dividend_yield:.2f}%")
print(f"Market Cap: ${market_cap:,.0f}")
```

See: [Stocks Guide](stocks.md)

### 2. Options

Contracts giving the right (not obligation) to buy/sell at a specific price.

**Types:**
- **Call Option**: Right to buy
- **Put Option**: Right to sell

**Key Terms:**
- **Strike Price**: Exercise price
- **Premium**: Option cost
- **Expiration Date**: Contract end date
- **In-the-Money (ITM)**: Profitable to exercise
- **Out-of-the-Money (OTM)**: Not profitable to exercise
- **At-the-Money (ATM)**: Strike ≈ Current price

**Greeks:**
- **Delta**: Price sensitivity to underlying
- **Gamma**: Rate of delta change
- **Theta**: Time decay
- **Vega**: Volatility sensitivity
- **Rho**: Interest rate sensitivity

See: [Options Trading](options.md)

### 3. Futures

Obligatory contracts to buy/sell at a future date and price.

**Characteristics:**
- Standardized contracts
- Exchange-traded
- Margin requirements
- Daily settlement

**Uses:**
- Hedging risk
- Speculation
- Price discovery

**Common Futures:**
- Equity index futures (S&P 500, NASDAQ)
- Commodity futures (oil, gold, corn)
- Currency futures
- Interest rate futures

See: [Futures Trading](futures.md)

### 4. Cryptocurrencies

Digital or virtual currencies using cryptography.

**Popular Cryptocurrencies:**
- **Bitcoin (BTC)**: First cryptocurrency
- **Ethereum (ETH)**: Smart contract platform
- **Altcoins**: Alternative cryptocurrencies

**Key Concepts:**
- **Blockchain**: Distributed ledger technology
- **Mining**: Transaction verification process
- **Wallet**: Storage for private keys
- **Exchange**: Platform for trading crypto

See: [Cryptocurrency Guide](crypto.md)

## Investment Strategies

### Value Investing

Buy undervalued securities based on fundamental analysis.

**Key Principles:**
- Focus on intrinsic value
- Margin of safety
- Long-term perspective
- Fundamental analysis

**Metrics:**
- P/E ratio
- Price-to-Book (P/B) ratio
- Debt-to-Equity ratio
- Free cash flow

### Growth Investing

Invest in companies with high growth potential.

**Characteristics:**
- High P/E ratios
- Revenue growth
- Market expansion
- Innovation focus

### Dividend Investing

Focus on stocks paying regular dividends.

**Benefits:**
- Steady income stream
- Lower volatility
- Compound growth

**Metrics:**
- Dividend yield
- Payout ratio
- Dividend growth rate

### Index Investing

Track market indices through index funds/ETFs.

**Advantages:**
- Diversification
- Low fees
- Passive management
- Market returns

**Popular Indices:**
- S&P 500
- NASDAQ-100
- Dow Jones Industrial Average
- Russell 2000

## Analysis Methods

### Fundamental Analysis

Evaluate intrinsic value through financial statements.

**Financial Statements:**
1. **Income Statement**: Revenue, expenses, profit
2. **Balance Sheet**: Assets, liabilities, equity
3. **Cash Flow Statement**: Operating, investing, financing cash flows

**Key Ratios:**
```python
# Profitability Ratios
def gross_margin(revenue, cogs):
    return ((revenue - cogs) / revenue) * 100

def net_profit_margin(net_income, revenue):
    return (net_income / revenue) * 100

def return_on_equity(net_income, shareholders_equity):
    return (net_income / shareholders_equity) * 100

# Liquidity Ratios
def current_ratio(current_assets, current_liabilities):
    return current_assets / current_liabilities

def quick_ratio(current_assets, inventory, current_liabilities):
    return (current_assets - inventory) / current_liabilities

# Leverage Ratios
def debt_to_equity(total_debt, total_equity):
    return total_debt / total_equity

def interest_coverage(ebit, interest_expense):
    return ebit / interest_expense

# Efficiency Ratios
def asset_turnover(revenue, total_assets):
    return revenue / total_assets

def inventory_turnover(cogs, average_inventory):
    return cogs / average_inventory
```

See: [Fundamental Analysis](fundamental_analysis.md)

### Technical Analysis

Analyze price patterns and trends using charts.

**Common Indicators:**
- **Moving Averages**: Simple (SMA), Exponential (EMA)
- **RSI**: Relative Strength Index (overbought/oversold)
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Volatility indicator
- **Volume**: Trading activity

```python
import pandas as pd
import numpy as np

def simple_moving_average(prices, period):
    """Calculate SMA"""
    return prices.rolling(window=period).mean()

def exponential_moving_average(prices, period):
    """Calculate EMA"""
    return prices.ewm(span=period, adjust=False).mean()

def relative_strength_index(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = simple_moving_average(prices, period)
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = exponential_moving_average(prices, fast)
    ema_slow = exponential_moving_average(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram
```

See: [Technical Analysis](technical_analysis.md)

## Risk Management

### Portfolio Diversification

Don't put all eggs in one basket.

**Diversification Strategies:**
- Across asset classes (stocks, bonds, real estate)
- Across sectors (tech, healthcare, finance)
- Across geographies (domestic, international)
- Across market caps (large, mid, small)

### Position Sizing

Determine how much to invest in each position.

```python
def position_size_fixed_dollar(account_balance, risk_per_trade):
    """Fixed dollar amount per trade"""
    return risk_per_trade

def position_size_percentage(account_balance, risk_percentage):
    """Percentage of account balance"""
    return account_balance * (risk_percentage / 100)

def position_size_volatility(account_balance, risk_percentage, entry_price, stop_loss):
    """Based on volatility and stop loss"""
    risk_per_share = abs(entry_price - stop_loss)
    total_risk = account_balance * (risk_percentage / 100)
    shares = total_risk / risk_per_share
    return int(shares)

# Example
account = 100000
risk_pct = 2  # 2% risk per trade
entry = 150
stop = 145

shares = position_size_volatility(account, risk_pct, entry, stop)
print(f"Buy {shares} shares at ${entry} with stop at ${stop}")
```

### Stop Loss Orders

Automatic sell orders to limit losses.

**Types:**
- **Fixed Stop**: Specific price level
- **Trailing Stop**: Adjusts with price movement
- **Percentage Stop**: Based on percentage decline

```python
def calculate_stop_loss(entry_price, stop_percentage, position_type='long'):
    """Calculate stop loss price"""
    if position_type == 'long':
        return entry_price * (1 - stop_percentage / 100)
    else:  # short
        return entry_price * (1 + stop_percentage / 100)

def calculate_take_profit(entry_price, risk_reward_ratio, stop_loss, position_type='long'):
    """Calculate take profit based on risk/reward ratio"""
    risk = abs(entry_price - stop_loss)
    reward = risk * risk_reward_ratio
    
    if position_type == 'long':
        return entry_price + reward
    else:  # short
        return entry_price - reward

# Example: 2:1 risk/reward ratio
entry = 100
stop_pct = 5
rr_ratio = 2

stop = calculate_stop_loss(entry, stop_pct, 'long')
target = calculate_take_profit(entry, rr_ratio, stop, 'long')

print(f"Entry: ${entry}")
print(f"Stop Loss: ${stop:.2f}")
print(f"Take Profit: ${target:.2f}")
print(f"Risk: ${entry - stop:.2f}")
print(f"Reward: ${target - entry:.2f}")
```

## Performance Metrics

### Returns

```python
def simple_return(start_value, end_value):
    """Simple return percentage"""
    return ((end_value - start_value) / start_value) * 100

def compound_annual_growth_rate(start_value, end_value, years):
    """CAGR"""
    return (((end_value / start_value) ** (1 / years)) - 1) * 100

def total_return(initial_investment, final_value, dividends):
    """Total return including dividends"""
    return ((final_value + dividends - initial_investment) / initial_investment) * 100
```

### Risk Metrics

```python
import numpy as np

def volatility(returns):
    """Standard deviation of returns (annualized)"""
    return np.std(returns) * np.sqrt(252)  # 252 trading days

def sharpe_ratio(returns, risk_free_rate=0.02):
    """Risk-adjusted return"""
    excess_returns = returns - risk_free_rate / 252
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

def maximum_drawdown(prices):
    """Maximum peak-to-trough decline"""
    cummax = np.maximum.accumulate(prices)
    drawdown = (prices - cummax) / cummax
    return np.min(drawdown) * 100

def beta(asset_returns, market_returns):
    """Measure of volatility relative to market"""
    covariance = np.cov(asset_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    return covariance / market_variance
```

## Trading Psychology

### Emotional Discipline

**Common Pitfalls:**
- **Fear of Missing Out (FOMO)**: Chasing rallies
- **Loss Aversion**: Holding losers too long
- **Overconfidence**: Taking excessive risk
- **Confirmation Bias**: Seeking supporting evidence only
- **Anchoring**: Fixating on specific price points

**Best Practices:**
- Follow your trading plan
- Keep emotions in check
- Accept losses gracefully
- Don't overtrade
- Take breaks when needed

### Trading Plan

Essential components:
1. **Entry Criteria**: When to buy
2. **Exit Criteria**: When to sell (profit and loss)
3. **Position Sizing**: How much to invest
4. **Risk Management**: Stop loss levels
5. **Record Keeping**: Track all trades

## Financial Calculations

### Time Value of Money

```python
def future_value(present_value, rate, periods):
    """FV = PV × (1 + r)^n"""
    return present_value * (1 + rate) ** periods

def present_value(future_value, rate, periods):
    """PV = FV / (1 + r)^n"""
    return future_value / (1 + rate) ** periods

def compound_interest(principal, rate, periods, compounds_per_period=1):
    """Compound interest formula"""
    return principal * (1 + rate / compounds_per_period) ** (periods * compounds_per_period)

# Example: $10,000 invested for 10 years at 7% annual return
principal = 10000
rate = 0.07
years = 10

fv = future_value(principal, rate, years)
print(f"${principal:,.2f} grows to ${fv:,.2f} in {years} years")
```

### Annuities

```python
def future_value_annuity(payment, rate, periods):
    """FV of regular payments"""
    return payment * (((1 + rate) ** periods - 1) / rate)

def present_value_annuity(payment, rate, periods):
    """PV of regular payments"""
    return payment * ((1 - (1 + rate) ** -periods) / rate)

def loan_payment(principal, rate, periods):
    """Calculate loan payment"""
    return principal * (rate * (1 + rate) ** periods) / ((1 + rate) ** periods - 1)

# Example: Mortgage calculation
loan_amount = 300000
annual_rate = 0.04
years = 30
monthly_rate = annual_rate / 12
months = years * 12

monthly_payment = loan_payment(loan_amount, monthly_rate, months)
total_paid = monthly_payment * months
total_interest = total_paid - loan_amount

print(f"Monthly Payment: ${monthly_payment:,.2f}")
print(f"Total Interest: ${total_interest:,.2f}")
```

## Investment Accounts

### Account Types

**Taxable Accounts:**
- Individual brokerage accounts
- Joint accounts
- Margin accounts

**Tax-Advantaged (US):**
- **401(k)**: Employer-sponsored retirement
- **IRA**: Individual Retirement Account
- **Roth IRA**: Tax-free growth and withdrawals
- **HSA**: Health Savings Account

### Fees and Costs

- **Expense Ratios**: Mutual fund/ETF annual fees
- **Trading Commissions**: Per-trade fees
- **Management Fees**: Advisory fees
- **Tax Implications**: Capital gains, dividends

## Market Orders

### Order Types

```python
# Common order types

class Order:
    """Order examples"""
    
    @staticmethod
    def market_order():
        """Execute at current market price"""
        return {
            "type": "market",
            "execution": "immediate",
            "price": "current market price"
        }
    
    @staticmethod
    def limit_order(limit_price):
        """Execute at specific price or better"""
        return {
            "type": "limit",
            "limit_price": limit_price,
            "execution": "when price reaches limit"
        }
    
    @staticmethod
    def stop_loss_order(stop_price):
        """Sell when price falls to stop level"""
        return {
            "type": "stop_loss",
            "stop_price": stop_price,
            "execution": "when price hits stop"
        }
    
    @staticmethod
    def stop_limit_order(stop_price, limit_price):
        """Combines stop and limit orders"""
        return {
            "type": "stop_limit",
            "stop_price": stop_price,
            "limit_price": limit_price,
            "execution": "limit order triggered at stop"
        }
```

### Order Duration

- **Day Order**: Expires at end of trading day
- **Good Till Canceled (GTC)**: Active until executed or canceled
- **Fill or Kill (FOK)**: Execute immediately in full or cancel
- **Immediate or Cancel (IOC)**: Execute immediately, cancel remainder

## Resources and Tools

### Financial Data Sources

- Yahoo Finance
- Bloomberg Terminal
- TradingView
- Alpha Vantage API
- IEX Cloud

### Analysis Tools

- Excel/Google Sheets
- Python (pandas, numpy, matplotlib)
- TradingView
- ThinkOrSwim
- MetaTrader

### Educational Resources

- Investopedia
- Khan Academy (Finance)
- CFA Institute
- Financial news (WSJ, FT, Bloomberg)

## Available Guides

Explore detailed guides for specific topics:

1. [General Finance](general.md) - Fundamental concepts and principles
2. [Stocks](stocks.md) - Equity investing and analysis
3. [Options](options.md) - Options trading strategies
4. [Futures](futures.md) - Futures contracts and trading
5. [Cryptocurrency](crypto.md) - Digital assets and blockchain
6. [Fundamental Analysis](fundamental_analysis.md) - Company valuation
7. [Technical Analysis](technical_analysis.md) - Chart patterns and indicators

## Important Disclaimers

1. **Not Financial Advice**: This is educational content only
2. **Do Your Own Research**: Always verify information
3. **Risk Warning**: Investing involves risk of loss
4. **Past Performance**: Does not guarantee future results
5. **Diversification**: Does not ensure profit or protect against loss
6. **Consult Professionals**: Consider seeking professional advice

## Key Principles

### Investment Principles

1. **Start Early**: Compound interest is powerful
2. **Diversify**: Spread risk across assets
3. **Invest Regularly**: Dollar-cost averaging
4. **Control Costs**: Minimize fees and taxes
5. **Stay Disciplined**: Stick to your plan
6. **Educate Yourself**: Continuous learning
7. **Manage Risk**: Protect your capital
8. **Think Long-Term**: Avoid emotional decisions

### Risk Management Rules

1. Never risk more than you can afford to lose
2. Use stop losses to limit downside
3. Diversify across multiple positions
4. Size positions appropriately
5. Have a clear exit strategy
6. Don't let winners become losers
7. Cut losses quickly
8. Let profits run (within reason)

## Next Steps

1. Learn the basics of [General Finance](general.md)
2. Study [Fundamental Analysis](fundamental_analysis.md)
3. Explore [Technical Analysis](technical_analysis.md)
4. Understand [Stocks](stocks.md) and equity markets
5. Learn about [Options](options.md) for hedging and income
6. Research [Cryptocurrency](crypto.md) opportunities
7. Practice with paper trading before using real money
8. Build a diversified portfolio aligned with your goals
9. Continuously educate yourself
10. Start investing with money you can afford to lose

Remember: Successful investing requires knowledge, discipline, and patience. Take time to learn, practice with small amounts, and gradually build your skills and portfolio.

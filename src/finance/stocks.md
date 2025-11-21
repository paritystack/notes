# Stocks

## What are Stocks?

Stocks (also called shares or equities) represent ownership in a corporation. When you buy a stock, you become a partial owner of the company and have a claim on its assets and earnings. Stocks are one of the most popular investment vehicles for building long-term wealth.

### Types of Stock

**Common Stock:**
- Voting rights in company decisions
- Dividends (if declared by board)
- Capital appreciation potential
- Last claim on assets in bankruptcy

**Preferred Stock:**
- Fixed dividend payments
- Priority over common stock for dividends and liquidation
- Limited or no voting rights
- Less price volatility

## How Stock Markets Work

### Primary Market
The primary market is where new securities are issued through:
- **Initial Public Offerings (IPOs)**: First public sale of company stock
- **Follow-on Offerings**: Additional shares issued after IPO
- **Rights Issues**: Existing shareholders given rights to buy new shares

### Secondary Market
The secondary market is where existing securities are traded between investors:
- **Stock Exchanges**: NYSE, NASDAQ, etc.
- **Over-the-Counter (OTC)**: Direct trading between parties
- Provides liquidity for investors to buy and sell

## Stock Market Classifications

### By Market Capitalization

```python
# Market Cap Calculation
def market_capitalization(share_price, shares_outstanding):
    """Calculate total market value of company"""
    return share_price * shares_outstanding

# Classification
def classify_by_market_cap(market_cap):
    """Classify stock by market capitalization"""
    if market_cap >= 200_000_000_000:
        return "Mega Cap"
    elif market_cap >= 10_000_000_000:
        return "Large Cap"
    elif market_cap >= 2_000_000_000:
        return "Mid Cap"
    elif market_cap >= 300_000_000:
        return "Small Cap"
    elif market_cap >= 50_000_000:
        return "Micro Cap"
    else:
        return "Nano Cap"

# Example
share_price = 150
shares_outstanding = 1_000_000_000  # 1 billion shares
market_cap = market_capitalization(share_price, shares_outstanding)
classification = classify_by_market_cap(market_cap)

print(f"Market Cap: ${market_cap:,.0f}")
print(f"Classification: {classification}")
```

**Market Cap Categories:**
- **Mega Cap**: $200B+ (e.g., Apple, Microsoft)
- **Large Cap**: $10B - $200B (established, stable companies)
- **Mid Cap**: $2B - $10B (growth potential with moderate risk)
- **Small Cap**: $300M - $2B (higher growth potential, higher risk)
- **Micro Cap**: $50M - $300M (high risk, high potential)
- **Nano Cap**: Under $50M (very high risk, speculative)

### By Sector

**11 GICS Sectors:**
1. **Information Technology**: Software, hardware, semiconductors
2. **Healthcare**: Pharmaceuticals, biotech, medical devices
3. **Financials**: Banks, insurance, investment firms
4. **Consumer Discretionary**: Retail, automotive, leisure
5. **Communication Services**: Telecom, media, entertainment
6. **Industrials**: Aerospace, machinery, construction
7. **Consumer Staples**: Food, beverages, household products
8. **Energy**: Oil, gas, renewable energy
9. **Utilities**: Electric, water, gas utilities
10. **Real Estate**: REITs, real estate management
11. **Materials**: Chemicals, metals, mining

### By Investment Style

**Growth Stocks:**
- High revenue/earnings growth rates
- High P/E ratios
- Reinvest profits rather than pay dividends
- Higher volatility
- Examples: Technology startups, biotech

**Value Stocks:**
- Trading below intrinsic value
- Low P/E, P/B ratios
- Established businesses
- Often pay dividends
- Examples: Mature industrial companies, utilities

**Dividend Stocks:**
- Regular dividend payments
- Stable cash flows
- Lower growth rates
- Income-focused investment
- Examples: Utilities, REITs, blue-chip companies

**Cyclical vs Defensive:**
- **Cyclical**: Performance tied to economic cycles (auto, construction, luxury)
- **Defensive**: Stable performance regardless of economy (utilities, healthcare, consumer staples)

## Key Financial Ratios and Metrics

### Valuation Ratios

```python
# Price to Earnings Ratio (P/E)
def pe_ratio(stock_price, earnings_per_share):
    """
    Measures how much investors pay per dollar of earnings
    Higher P/E = Higher growth expectations or overvaluation
    Lower P/E = Undervaluation or lower growth expectations
    """
    return stock_price / earnings_per_share

# Forward P/E
def forward_pe(stock_price, forward_eps):
    """P/E based on projected future earnings"""
    return stock_price / forward_eps

# Price to Book Ratio (P/B)
def pb_ratio(stock_price, book_value_per_share):
    """
    Compares market price to book value
    P/B < 1 may indicate undervaluation
    P/B > 1 suggests market values company above net assets
    """
    return stock_price / book_value_per_share

# Price to Sales Ratio (P/S)
def ps_ratio(market_cap, total_sales):
    """
    Useful for companies with no earnings yet
    Lower P/S generally indicates better value
    """
    return market_cap / total_sales

# PEG Ratio
def peg_ratio(pe, earnings_growth_rate):
    """
    P/E adjusted for growth rate
    PEG < 1 suggests stock may be undervalued given growth
    PEG > 1 suggests possible overvaluation
    """
    return pe / earnings_growth_rate

# Enterprise Value Ratios
def enterprise_value(market_cap, total_debt, cash):
    """Total value including debt, excluding cash"""
    return market_cap + total_debt - cash

def ev_to_ebitda(enterprise_value, ebitda):
    """Popular valuation metric independent of capital structure"""
    return enterprise_value / ebitda

def ev_to_sales(enterprise_value, revenue):
    """Alternative valuation multiple"""
    return enterprise_value / revenue

# Example Usage
stock_price = 50
eps = 3.50
forward_eps = 4.20
book_value_per_share = 25
market_cap = 50_000_000_000
revenue = 10_000_000_000
total_debt = 5_000_000_000
cash = 2_000_000_000
ebitda = 2_000_000_000
growth_rate = 15  # 15% annual growth

pe = pe_ratio(stock_price, eps)
fwd_pe = forward_pe(stock_price, forward_eps)
pb = pb_ratio(stock_price, book_value_per_share)
ps = ps_ratio(market_cap, revenue)
peg = peg_ratio(pe, growth_rate)
ev = enterprise_value(market_cap, total_debt, cash)
ev_ebitda = ev_to_ebitda(ev, ebitda)

print(f"P/E Ratio: {pe:.2f}")
print(f"Forward P/E: {fwd_pe:.2f}")
print(f"P/B Ratio: {pb:.2f}")
print(f"P/S Ratio: {ps:.2f}")
print(f"PEG Ratio: {peg:.2f}")
print(f"Enterprise Value: ${ev:,.0f}")
print(f"EV/EBITDA: {ev_ebitda:.2f}")
```

### Profitability Ratios

```python
# Earnings Per Share (EPS)
def eps(net_income, preferred_dividends, shares_outstanding):
    """Profit allocated to each share of common stock"""
    return (net_income - preferred_dividends) / shares_outstanding

# Return on Equity (ROE)
def roe(net_income, shareholders_equity):
    """
    Measures profitability relative to equity
    Higher ROE indicates efficient use of equity capital
    Target: Generally >15% is considered good
    """
    return (net_income / shareholders_equity) * 100

# Return on Assets (ROA)
def roa(net_income, total_assets):
    """
    Measures how efficiently assets generate profit
    Higher ROA indicates better asset utilization
    """
    return (net_income / total_assets) * 100

# Profit Margins
def gross_margin(gross_profit, revenue):
    """Profitability after cost of goods sold"""
    return (gross_profit / revenue) * 100

def operating_margin(operating_income, revenue):
    """Profitability after operating expenses"""
    return (operating_income / revenue) * 100

def net_margin(net_income, revenue):
    """Bottom-line profitability"""
    return (net_income / revenue) * 100

# Example
net_income = 1_000_000_000
preferred_divs = 0
shares_outstanding = 1_000_000_000
shareholders_equity = 8_000_000_000
total_assets = 15_000_000_000
revenue = 10_000_000_000
gross_profit = 6_000_000_000
operating_income = 2_000_000_000

earnings_per_share = eps(net_income, preferred_divs, shares_outstanding)
return_on_equity = roe(net_income, shareholders_equity)
return_on_assets = roa(net_income, total_assets)
gross_margin_pct = gross_margin(gross_profit, revenue)
operating_margin_pct = operating_margin(operating_income, revenue)
net_margin_pct = net_margin(net_income, revenue)

print(f"EPS: ${earnings_per_share:.2f}")
print(f"ROE: {return_on_equity:.2f}%")
print(f"ROA: {return_on_assets:.2f}%")
print(f"Gross Margin: {gross_margin_pct:.2f}%")
print(f"Operating Margin: {operating_margin_pct:.2f}%")
print(f"Net Margin: {net_margin_pct:.2f}%")
```

### Liquidity Ratios

```python
# Current Ratio
def current_ratio(current_assets, current_liabilities):
    """
    Measures ability to pay short-term obligations
    Target: Generally > 1.5 is healthy
    < 1.0 may indicate liquidity problems
    """
    return current_assets / current_liabilities

# Quick Ratio (Acid Test)
def quick_ratio(current_assets, inventory, current_liabilities):
    """
    More stringent test excluding inventory
    Target: Generally > 1.0 is healthy
    """
    return (current_assets - inventory) / current_liabilities

# Cash Ratio
def cash_ratio(cash, marketable_securities, current_liabilities):
    """Most conservative liquidity measure"""
    return (cash + marketable_securities) / current_liabilities

# Example
current_assets = 5_000_000_000
inventory = 1_500_000_000
cash = 2_000_000_000
marketable_securities = 500_000_000
current_liabilities = 3_000_000_000

curr_ratio = current_ratio(current_assets, current_liabilities)
quick = quick_ratio(current_assets, inventory, current_liabilities)
cash_r = cash_ratio(cash, marketable_securities, current_liabilities)

print(f"Current Ratio: {curr_ratio:.2f}")
print(f"Quick Ratio: {quick:.2f}")
print(f"Cash Ratio: {cash_r:.2f}")
```

### Leverage Ratios

```python
# Debt to Equity Ratio (D/E)
def debt_to_equity(total_debt, shareholders_equity):
    """
    Measures financial leverage
    Higher ratio = More debt financing (risky)
    Lower ratio = More equity financing (conservative)
    Target varies by industry
    """
    return total_debt / shareholders_equity

# Debt to Assets Ratio
def debt_to_assets(total_debt, total_assets):
    """Percentage of assets financed by debt"""
    return (total_debt / total_assets) * 100

# Interest Coverage Ratio
def interest_coverage(ebit, interest_expense):
    """
    Ability to pay interest on debt
    Higher is better; < 1.5 is concerning
    """
    return ebit / interest_expense

# Example
total_debt = 5_000_000_000
shareholders_equity = 8_000_000_000
total_assets = 15_000_000_000
ebit = 2_500_000_000
interest_expense = 200_000_000

de_ratio = debt_to_equity(total_debt, shareholders_equity)
da_ratio = debt_to_assets(total_debt, total_assets)
int_coverage = interest_coverage(ebit, interest_expense)

print(f"Debt-to-Equity: {de_ratio:.2f}")
print(f"Debt-to-Assets: {da_ratio:.2f}%")
print(f"Interest Coverage: {int_coverage:.2f}x")
```

### Dividend Metrics

```python
# Dividend Yield
def dividend_yield(annual_dividend_per_share, stock_price):
    """
    Annual dividend income as percentage of stock price
    Higher yield attracts income investors
    Very high yield may signal dividend cut risk
    """
    return (annual_dividend_per_share / stock_price) * 100

# Dividend Payout Ratio
def dividend_payout_ratio(dividends_per_share, earnings_per_share):
    """
    Percentage of earnings paid as dividends
    < 50%: Room to grow dividends
    > 80%: Less sustainable, less growth investment
    """
    return (dividends_per_share / earnings_per_share) * 100

# Dividend Coverage Ratio
def dividend_coverage(earnings_per_share, dividends_per_share):
    """How many times earnings cover dividend (inverse of payout ratio)"""
    return earnings_per_share / dividends_per_share

# Dividend Growth Rate
def dividend_growth_rate(current_dividend, previous_dividend):
    """Year-over-year dividend growth"""
    return ((current_dividend - previous_dividend) / previous_dividend) * 100

# Example
annual_dividend = 2.50
stock_price = 50
eps = 4.00
previous_year_dividend = 2.30

div_yield = dividend_yield(annual_dividend, stock_price)
payout_ratio = dividend_payout_ratio(annual_dividend, eps)
div_coverage = dividend_coverage(eps, annual_dividend)
div_growth = dividend_growth_rate(annual_dividend, previous_year_dividend)

print(f"Dividend Yield: {div_yield:.2f}%")
print(f"Payout Ratio: {payout_ratio:.2f}%")
print(f"Dividend Coverage: {div_coverage:.2f}x")
print(f"Dividend Growth: {div_growth:.2f}%")
```

## Stock Trading Strategies

### 1. Day Trading

**Characteristics:**
- Positions opened and closed within same day
- No overnight risk
- Requires significant time commitment
- High transaction costs

**Key Concepts:**
- Scalping: Very short-term trades for small profits
- Momentum trading: Riding intraday price movements
- Technical analysis intensive

**Tools:**
- Level 2 quotes
- Real-time charts
- Direct market access

### 2. Swing Trading

**Characteristics:**
- Hold positions for days to weeks
- Capture short to medium-term price swings
- Moderate time commitment
- Balance of technical and fundamental analysis

**Strategy:**
- Identify support/resistance levels
- Trade breakouts and reversals
- Use stop losses for risk management

**Example:**
```python
# Simple Swing Trading Position Calculator

def calculate_position_size(account_size, risk_percent, entry_price, stop_loss):
    """Calculate shares based on risk tolerance"""
    risk_amount = account_size * (risk_percent / 100)
    risk_per_share = entry_price - stop_loss
    shares = int(risk_amount / risk_per_share)
    return shares

def calculate_targets(entry_price, risk_per_share, reward_ratio):
    """Calculate profit targets based on risk-reward ratio"""
    target_price = entry_price + (risk_per_share * reward_ratio)
    return target_price

# Example
account_size = 50000
risk_percent = 2  # Risk 2% per trade
entry_price = 100
stop_loss = 95
reward_ratio = 3  # 3:1 reward to risk

shares = calculate_position_size(account_size, risk_percent, entry_price, stop_loss)
position_value = shares * entry_price
risk_per_share = entry_price - stop_loss
target_price = calculate_targets(entry_price, risk_per_share, reward_ratio)

print(f"Position Size: {shares} shares")
print(f"Position Value: ${position_value:,.0f}")
print(f"Entry Price: ${entry_price:.2f}")
print(f"Stop Loss: ${stop_loss:.2f}")
print(f"Target Price: ${target_price:.2f}")
print(f"Risk Amount: ${shares * risk_per_share:,.0f}")
print(f"Potential Profit: ${shares * (target_price - entry_price):,.0f}")
```

### 3. Position Trading (Long-Term Investing)

**Characteristics:**
- Hold positions for months to years
- Based on fundamental analysis
- Lower transaction costs
- Patience required

**Strategies:**
- **Buy and Hold**: Purchase quality stocks, hold long-term
- **Dollar Cost Averaging**: Regular fixed-amount investments
- **Dividend Growth Investing**: Focus on dividend growth stocks

### 4. Value Investing

**Philosophy:**
- Buy stocks trading below intrinsic value
- Margin of safety approach
- Focus on fundamentals

**Key Criteria:**
- Low P/E, P/B ratios
- Strong balance sheet
- Consistent earnings
- Competitive advantages (moats)

**Example Screening:**
```python
# Value Stock Screener

def value_score(pe, pb, de_ratio, roe, div_yield):
    """Simple value scoring system (lower is better value)"""
    score = 0

    # P/E scoring (lower is better)
    if pe < 15:
        score += 2
    elif pe < 20:
        score += 1

    # P/B scoring
    if pb < 1.5:
        score += 2
    elif pb < 3:
        score += 1

    # Debt level
    if de_ratio < 0.5:
        score += 2
    elif de_ratio < 1.0:
        score += 1

    # Profitability
    if roe > 15:
        score += 2
    elif roe > 10:
        score += 1

    # Dividend yield
    if div_yield > 3:
        score += 2
    elif div_yield > 2:
        score += 1

    return score

# Example stocks
stocks = {
    'Stock A': {'pe': 12, 'pb': 1.2, 'de': 0.4, 'roe': 18, 'div': 3.5},
    'Stock B': {'pe': 25, 'pb': 4.0, 'de': 1.5, 'roe': 22, 'div': 1.0},
    'Stock C': {'pe': 16, 'pb': 2.5, 'de': 0.7, 'roe': 14, 'div': 2.8},
}

print("Value Stock Screening Results:")
print("-" * 50)
for name, metrics in stocks.items():
    score = value_score(metrics['pe'], metrics['pb'], metrics['de'],
                       metrics['roe'], metrics['div'])
    rating = "Strong Buy" if score >= 8 else "Buy" if score >= 6 else "Hold" if score >= 4 else "Avoid"
    print(f"{name}: Score = {score}/10 - {rating}")
```

### 5. Growth Investing

**Philosophy:**
- Invest in companies with high growth potential
- Accept higher valuations for future growth
- Focus on innovation and market disruption

**Key Criteria:**
- High revenue/earnings growth
- Expanding margins
- Large addressable markets
- Innovative products/services
- Strong management

**Metrics to Focus On:**
- Revenue growth rate (>20% annually)
- EPS growth rate
- Market share gains
- PEG ratio (< 2.0)

### 6. Dividend Investing

**Philosophy:**
- Generate passive income through dividends
- Benefit from compounding
- Focus on stable, mature companies

**Dividend Aristocrats Criteria:**
- 25+ years of consecutive dividend increases
- S&P 500 member
- Strong financial health

**Strategy:**
```python
# Dividend Income Calculator

def annual_dividend_income(shares, dividend_per_share):
    """Calculate annual dividend income"""
    return shares * dividend_per_share

def dividend_reinvestment_growth(initial_investment, annual_dividend_yield,
                                 annual_dividend_growth, capital_appreciation,
                                 years):
    """
    Calculate portfolio value with dividend reinvestment
    and dividend growth over time
    """
    portfolio_value = initial_investment
    annual_income = initial_investment * (annual_dividend_yield / 100)

    results = []
    for year in range(1, years + 1):
        # Reinvest dividends
        portfolio_value += annual_income

        # Capital appreciation
        portfolio_value *= (1 + capital_appreciation / 100)

        # Dividend growth
        annual_income *= (1 + annual_dividend_growth / 100)

        results.append({
            'year': year,
            'portfolio_value': portfolio_value,
            'annual_income': annual_income
        })

    return results

# Example: Dividend Reinvestment Plan
initial_investment = 100000
annual_yield = 4.0  # 4% dividend yield
dividend_growth = 7.0  # 7% annual dividend growth
price_appreciation = 6.0  # 6% annual price growth
years = 20

results = dividend_reinvestment_growth(initial_investment, annual_yield,
                                       dividend_growth, price_appreciation, years)

print("Dividend Reinvestment Growth Projection")
print("=" * 60)
print(f"Initial Investment: ${initial_investment:,.0f}")
print(f"Dividend Yield: {annual_yield}%")
print(f"Dividend Growth: {dividend_growth}%")
print(f"Price Appreciation: {price_appreciation}%")
print("\nProjected Results:")
print("-" * 60)

# Show select years
for result in [results[4], results[9], results[14], results[19]]:
    print(f"Year {result['year']:2d}: Portfolio = ${result['portfolio_value']:,.0f}, "
          f"Annual Income = ${result['annual_income']:,.0f}")

final_value = results[-1]['portfolio_value']
final_income = results[-1]['annual_income']
total_return = ((final_value - initial_investment) / initial_investment) * 100

print("\n" + "=" * 60)
print(f"Final Portfolio Value: ${final_value:,.0f}")
print(f"Final Annual Income: ${final_income:,.0f}")
print(f"Total Return: {total_return:.1f}%")
print(f"Annualized Return: {(total_return / years):.1f}%")
```

### 7. Sector Rotation Strategy

**Concept:**
- Different sectors perform better at different economic stages
- Rotate investments based on economic cycle

**Economic Cycle Sectors:**
- **Early Recovery**: Financials, Technology, Industrials
- **Mid Expansion**: Consumer Discretionary, Industrials, Materials
- **Late Expansion**: Energy, Materials, Staples
- **Recession**: Utilities, Healthcare, Consumer Staples

## Corporate Actions

### Stock Splits

**Forward Split** (e.g., 2-for-1):
- Each share becomes 2 shares
- Price halves
- Market cap unchanged
- Makes stock more accessible

**Reverse Split** (e.g., 1-for-10):
- 10 shares become 1 share
- Price multiplies by 10
- Often signals financial distress

```python
# Stock Split Calculator

def forward_split(shares_owned, current_price, split_ratio):
    """Calculate post-split shares and price"""
    new_shares = shares_owned * split_ratio
    new_price = current_price / split_ratio
    return new_shares, new_price

def reverse_split(shares_owned, current_price, reverse_ratio):
    """Calculate post-reverse-split shares and price"""
    new_shares = shares_owned / reverse_ratio
    new_price = current_price * reverse_ratio
    return new_shares, new_price

# Example: 3-for-1 forward split
shares = 100
price = 300
split = 3

new_shares, new_price = forward_split(shares, price, split)
print(f"Before Split: {shares} shares @ ${price} = ${shares * price:,.0f}")
print(f"After {split}:1 Split: {int(new_shares)} shares @ ${new_price:.2f} = ${int(new_shares) * new_price:,.0f}")
```

### Dividends

**Types:**
- **Cash Dividends**: Direct cash payment to shareholders
- **Stock Dividends**: Additional shares instead of cash
- **Special Dividends**: One-time large payments

**Important Dates:**
- **Declaration Date**: Board announces dividend
- **Ex-Dividend Date**: Must own stock before this date
- **Record Date**: Shareholders on record receive dividend
- **Payment Date**: Dividend paid out

### Stock Buybacks

**Share Repurchase:**
- Company buys its own shares from market
- Reduces shares outstanding
- Increases EPS
- Returns capital to shareholders
- May indicate management believes stock is undervalued

## Risk Management for Stock Investing

### Position Sizing

```python
# Position Sizing Methods

def equal_weight_sizing(portfolio_value, num_positions):
    """Equal dollar amount in each position"""
    return portfolio_value / num_positions

def risk_based_sizing(portfolio_value, risk_per_trade, entry_price, stop_loss):
    """Size based on maximum loss tolerance"""
    risk_amount = portfolio_value * (risk_per_trade / 100)
    risk_per_share = abs(entry_price - stop_loss)
    shares = risk_amount / risk_per_share
    position_size = shares * entry_price
    return shares, position_size

def market_cap_weighting(portfolio_value, stock_market_cap, total_market_cap):
    """Weight by market capitalization"""
    weight = stock_market_cap / total_market_cap
    return portfolio_value * weight

# Example
portfolio = 100000
positions = 10
risk_pct = 2
entry = 50
stop = 45

equal_position = equal_weight_sizing(portfolio, positions)
risk_shares, risk_position = risk_based_sizing(portfolio, risk_pct, entry, stop)

print(f"Equal Weight Position: ${equal_position:,.0f}")
print(f"Risk-Based Position: {int(risk_shares)} shares = ${risk_position:,.0f}")
```

### Diversification

**Portfolio Diversification Strategies:**

1. **Across Sectors**: 8-11 different sectors
2. **Across Market Caps**: Mix of large, mid, small cap
3. **Across Geographies**: Domestic and international
4. **Across Styles**: Mix of growth, value, dividend stocks

```python
# Portfolio Diversification Checker

def analyze_diversification(portfolio):
    """Analyze portfolio diversification by sector"""
    sectors = {}
    total_value = sum(stock['value'] for stock in portfolio)

    # Calculate sector allocations
    for stock in portfolio:
        sector = stock['sector']
        if sector in sectors:
            sectors[sector] += stock['value']
        else:
            sectors[sector] = stock['value']

    # Calculate percentages and check concentration
    print("Portfolio Sector Allocation:")
    print("-" * 40)
    concentrated_sectors = []

    for sector, value in sorted(sectors.items(), key=lambda x: x[1], reverse=True):
        percentage = (value / total_value) * 100
        print(f"{sector}: {percentage:.1f}%")

        if percentage > 25:
            concentrated_sectors.append((sector, percentage))

    # Warnings
    if concentrated_sectors:
        print("\nWARNING: Concentrated Positions")
        for sector, pct in concentrated_sectors:
            print(f"  {sector}: {pct:.1f}% (>25% threshold)")

    return sectors

# Example Portfolio
portfolio = [
    {'name': 'Tech Stock A', 'sector': 'Technology', 'value': 15000},
    {'name': 'Tech Stock B', 'sector': 'Technology', 'value': 10000},
    {'name': 'Healthcare Stock', 'sector': 'Healthcare', 'value': 12000},
    {'name': 'Financial Stock', 'sector': 'Financials', 'value': 13000},
    {'name': 'Consumer Stock', 'sector': 'Consumer Discretionary', 'value': 10000},
    {'name': 'Industrial Stock', 'sector': 'Industrials', 'value': 8000},
    {'name': 'Energy Stock', 'sector': 'Energy', 'value': 7000},
    {'name': 'Utility Stock', 'sector': 'Utilities', 'value': 5000},
]

analyze_diversification(portfolio)
```

### Stop Loss Strategies

```python
# Stop Loss Calculators

def fixed_percentage_stop(entry_price, stop_percentage):
    """Stop loss at fixed percentage below entry"""
    return entry_price * (1 - stop_percentage / 100)

def atr_based_stop(entry_price, atr, multiplier=2):
    """Stop loss based on Average True Range (volatility)"""
    return entry_price - (atr * multiplier)

def trailing_stop(current_price, highest_price, trail_percentage):
    """Trailing stop that moves up with price"""
    return highest_price * (1 - trail_percentage / 100)

# Example
entry_price = 100
stop_pct = 8  # 8% stop loss
atr = 3  # Average True Range = $3
current_price = 120
highest_price = 125
trail_pct = 10

fixed_stop = fixed_percentage_stop(entry_price, stop_pct)
atr_stop = atr_based_stop(entry_price, atr, 2)
trailing = trailing_stop(current_price, highest_price, trail_pct)

print(f"Entry Price: ${entry_price:.2f}")
print(f"Fixed {stop_pct}% Stop: ${fixed_stop:.2f}")
print(f"ATR-Based Stop (2x ATR): ${atr_stop:.2f}")
print(f"\nCurrent Price: ${current_price:.2f}")
print(f"Highest Price: ${highest_price:.2f}")
print(f"Trailing Stop ({trail_pct}%): ${trailing:.2f}")
```

## Practical Example: Complete Stock Analysis

```python
# Complete Stock Analysis Example

class StockAnalysis:
    def __init__(self, name, price, shares_outstanding):
        self.name = name
        self.price = price
        self.shares_outstanding = shares_outstanding
        self.financials = {}

    def set_financials(self, revenue, net_income, total_assets, total_debt,
                      shareholders_equity, operating_cash_flow, capex,
                      annual_dividend=0):
        """Set company financial data"""
        self.financials = {
            'revenue': revenue,
            'net_income': net_income,
            'total_assets': total_assets,
            'total_debt': total_debt,
            'shareholders_equity': shareholders_equity,
            'operating_cash_flow': operating_cash_flow,
            'capex': capex,
            'annual_dividend': annual_dividend
        }

    def calculate_metrics(self):
        """Calculate all key metrics"""
        f = self.financials

        # Market metrics
        market_cap = self.price * self.shares_outstanding
        eps = f['net_income'] / self.shares_outstanding
        book_value_ps = f['shareholders_equity'] / self.shares_outstanding

        # Valuation
        pe = self.price / eps
        pb = self.price / book_value_ps
        ps = market_cap / f['revenue']

        # Profitability
        roe = (f['net_income'] / f['shareholders_equity']) * 100
        roa = (f['net_income'] / f['total_assets']) * 100
        net_margin = (f['net_income'] / f['revenue']) * 100

        # Financial Health
        de_ratio = f['total_debt'] / f['shareholders_equity']

        # Cash Flow
        fcf = f['operating_cash_flow'] - f['capex']
        fcf_per_share = fcf / self.shares_outstanding

        # Dividend
        if f['annual_dividend'] > 0:
            div_yield = (f['annual_dividend'] / self.price) * 100
            payout_ratio = (f['annual_dividend'] / eps) * 100
        else:
            div_yield = 0
            payout_ratio = 0

        return {
            'market_cap': market_cap,
            'eps': eps,
            'pe': pe,
            'pb': pb,
            'ps': ps,
            'roe': roe,
            'roa': roa,
            'net_margin': net_margin,
            'de_ratio': de_ratio,
            'fcf': fcf,
            'fcf_per_share': fcf_per_share,
            'div_yield': div_yield,
            'payout_ratio': payout_ratio
        }

    def generate_report(self):
        """Generate comprehensive analysis report"""
        metrics = self.calculate_metrics()

        print("=" * 60)
        print(f"STOCK ANALYSIS: {self.name}")
        print("=" * 60)
        print(f"\nCurrent Price: ${self.price:.2f}")
        print(f"Market Cap: ${metrics['market_cap']:,.0f}")

        print("\n1. VALUATION METRICS")
        print("-" * 40)
        print(f"P/E Ratio: {metrics['pe']:.2f}")
        print(f"P/B Ratio: {metrics['pb']:.2f}")
        print(f"P/S Ratio: {metrics['ps']:.2f}")

        print("\n2. PROFITABILITY")
        print("-" * 40)
        print(f"EPS: ${metrics['eps']:.2f}")
        print(f"ROE: {metrics['roe']:.1f}%")
        print(f"ROA: {metrics['roa']:.1f}%")
        print(f"Net Margin: {metrics['net_margin']:.1f}%")

        print("\n3. FINANCIAL HEALTH")
        print("-" * 40)
        print(f"Debt-to-Equity: {metrics['de_ratio']:.2f}")

        print("\n4. CASH FLOW")
        print("-" * 40)
        print(f"Free Cash Flow: ${metrics['fcf']:,.0f}")
        print(f"FCF per Share: ${metrics['fcf_per_share']:.2f}")

        if metrics['div_yield'] > 0:
            print("\n5. DIVIDEND")
            print("-" * 40)
            print(f"Dividend Yield: {metrics['div_yield']:.2f}%")
            print(f"Payout Ratio: {metrics['payout_ratio']:.1f}%")

        print("\n6. INVESTMENT RATING")
        print("-" * 40)

        # Simple rating logic
        score = 0
        if metrics['pe'] < 20: score += 1
        if metrics['roe'] > 15: score += 1
        if metrics['de_ratio'] < 1.0: score += 1
        if metrics['fcf'] > 0: score += 1
        if metrics['net_margin'] > 10: score += 1

        rating = ["Avoid", "Hold", "Accumulate", "Buy", "Strong Buy"][score]
        print(f"Score: {score}/5")
        print(f"Rating: {rating}")
        print("=" * 60)

# Example Analysis
stock = StockAnalysis("TechGrowth Inc.", 75, 1_000_000_000)
stock.set_financials(
    revenue=15_000_000_000,
    net_income=2_000_000_000,
    total_assets=25_000_000_000,
    total_debt=5_000_000_000,
    shareholders_equity=18_000_000_000,
    operating_cash_flow=3_000_000_000,
    capex=800_000_000,
    annual_dividend=0.50
)

stock.generate_report()
```

## Key Takeaways

### Stock Selection Criteria:
1. **Strong Financials**: Healthy balance sheet, positive cash flow
2. **Competitive Advantages**: Moats that protect market position
3. **Growth Potential**: Expanding markets, innovation
4. **Reasonable Valuation**: Not overpaying for growth
5. **Quality Management**: Track record of execution

### Risk Management Essentials:
1. **Diversification**: Don't put all eggs in one basket
2. **Position Sizing**: Limit single position risk to 2-5% of portfolio
3. **Stop Losses**: Protect capital from large losses
4. **Regular Review**: Monitor holdings and rebalance

### Common Mistakes to Avoid:
1. Chasing hot stocks without research
2. Over-concentration in single sector
3. Ignoring valuation
4. Emotional decision-making
5. Lack of exit strategy
6. Trading too frequently (tax implications)

## Additional Resources

**Related Guides:**
- [Fundamental Analysis](./fundamental_analysis.md) - Detailed valuation methods and financial analysis
- [Technical Analysis](./technical_analysis.md) - Chart patterns and indicators
- [Options Trading](./options.md) - Derivatives strategies
- [Main Finance Guide](./README.md) - Comprehensive overview

**Stock Research Tools:**
- Yahoo Finance, Google Finance - Free stock data
- Seeking Alpha - Analysis and research
- Finviz - Stock screener
- TradingView - Charts and technical analysis
- SEC EDGAR - Official company filings

**Best Brokers for Stock Trading:**
- **Interactive Brokers**: Professional tools, low costs
- **Fidelity**: Research, retirement accounts
- **Charles Schwab**: Full service, good support
- **TD Ameritrade**: thinkorswim platform
- **Robinhood**: Simple interface, fractional shares
- **Vanguard**: Low-cost index funds

---

*Note: This guide is for educational purposes only and does not constitute financial advice. Stock investing involves risk, including potential loss of principal. Always conduct thorough research and consider consulting with financial professionals before making investment decisions.*

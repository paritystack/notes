# Fundamental Analysis

## What is Fundamental Analysis?

Fundamental analysis is a method of evaluating the intrinsic value of an asset, such as a stock, by examining related economic, financial, and other qualitative and quantitative factors. The goal of fundamental analysis is to determine whether an asset is overvalued or undervalued by the market, and to make investment decisions based on this assessment.

Unlike technical analysis which focuses on price patterns and trends, fundamental analysis looks at the underlying business fundamentals, financial health, competitive position, and growth prospects of a company.

## Key Components of Fundamental Analysis

### 1. Economic Analysis

Economic analysis involves examining the overall macroeconomic environment and its impact on businesses:

**Key Economic Indicators:**
- **GDP Growth**: Indicates overall economic health and potential for business expansion
- **Inflation Rates**: Affects purchasing power, interest rates, and input costs
- **Interest Rates**: Impacts borrowing costs, consumer spending, and discount rates for valuations
- **Unemployment Levels**: Reflects labor market health and consumer demand
- **Consumer Confidence**: Indicates future spending patterns
- **Trade Balance**: Affects currency values and export-oriented businesses

**Economic Cycles:**
- **Expansion**: Rising GDP, low unemployment, increasing business activity
- **Peak**: Maximum economic output, potential overheating
- **Contraction**: Declining GDP, rising unemployment, decreasing business activity
- **Trough**: Lowest point, potential for recovery

### 2. Industry Analysis

Industry analysis examines the specific sector in which a company operates:

**Porter's Five Forces Framework:**
1. **Threat of New Entrants**: Barriers to entry, capital requirements, economies of scale
2. **Bargaining Power of Suppliers**: Supplier concentration, switching costs, input availability
3. **Bargaining Power of Buyers**: Customer concentration, price sensitivity, switching costs
4. **Threat of Substitutes**: Alternative products/services, price-performance trade-offs
5. **Industry Rivalry**: Number of competitors, market growth rate, product differentiation

**Industry Life Cycle:**
- **Introduction**: High growth potential, high risk, low profitability
- **Growth**: Rapid expansion, increasing competition, improving margins
- **Maturity**: Stable growth, intense competition, market consolidation
- **Decline**: Declining demand, price competition, industry contraction

### 3. Company Analysis

Detailed examination of individual company fundamentals through financial statements, management quality, and competitive advantages.

## Financial Statements Analysis

### Income Statement

The income statement shows a company's revenues, expenses, and profits over a specific period.

**Key Metrics:**

```python
# Income Statement Analysis

# Revenue Growth Rate
def revenue_growth_rate(current_revenue, previous_revenue):
    """Calculate year-over-year revenue growth"""
    return ((current_revenue - previous_revenue) / previous_revenue) * 100

# Gross Profit Margin
def gross_profit_margin(gross_profit, revenue):
    """Measures efficiency of production"""
    return (gross_profit / revenue) * 100

# Operating Profit Margin
def operating_profit_margin(operating_income, revenue):
    """Measures operational efficiency"""
    return (operating_income / revenue) * 100

# Net Profit Margin
def net_profit_margin(net_income, revenue):
    """Measures overall profitability"""
    return (net_income / revenue) * 100

# EBITDA
def calculate_ebitda(net_income, interest, taxes, depreciation, amortization):
    """Earnings Before Interest, Taxes, Depreciation, and Amortization"""
    return net_income + interest + taxes + depreciation + amortization

# Example
revenue = 1000000
cogs = 600000
operating_expenses = 200000
interest = 20000
taxes = 36000
depreciation = 30000

gross_profit = revenue - cogs
operating_income = gross_profit - operating_expenses
ebit = operating_income
ebitda = ebit + depreciation
net_income = ebit - interest - taxes

print(f"Gross Profit Margin: {gross_profit_margin(gross_profit, revenue):.2f}%")
print(f"Operating Margin: {operating_profit_margin(operating_income, revenue):.2f}%")
print(f"Net Profit Margin: {net_profit_margin(net_income, revenue):.2f}%")
print(f"EBITDA: ${ebitda:,.0f}")
```

### Balance Sheet

The balance sheet provides a snapshot of a company's assets, liabilities, and shareholders' equity at a specific point in time.

**Key Metrics:**

```python
# Balance Sheet Analysis

# Current Ratio
def current_ratio(current_assets, current_liabilities):
    """Measures short-term liquidity"""
    return current_assets / current_liabilities

# Quick Ratio (Acid Test)
def quick_ratio(current_assets, inventory, current_liabilities):
    """Measures immediate liquidity without inventory"""
    return (current_assets - inventory) / current_liabilities

# Cash Ratio
def cash_ratio(cash, marketable_securities, current_liabilities):
    """Measures most liquid assets coverage"""
    return (cash + marketable_securities) / current_liabilities

# Debt-to-Equity Ratio
def debt_to_equity(total_debt, total_equity):
    """Measures financial leverage"""
    return total_debt / total_equity

# Debt-to-Assets Ratio
def debt_to_assets(total_debt, total_assets):
    """Measures proportion of assets financed by debt"""
    return total_debt / total_assets

# Working Capital
def working_capital(current_assets, current_liabilities):
    """Available capital for operations"""
    return current_assets - current_liabilities

# Book Value Per Share
def book_value_per_share(total_equity, shares_outstanding):
    """Equity value per share"""
    return total_equity / shares_outstanding

# Example
current_assets = 500000
inventory = 150000
cash = 100000
marketable_securities = 50000
current_liabilities = 250000
total_assets = 2000000
total_debt = 800000
total_equity = 1200000
shares_outstanding = 100000

print(f"Current Ratio: {current_ratio(current_assets, current_liabilities):.2f}")
print(f"Quick Ratio: {quick_ratio(current_assets, inventory, current_liabilities):.2f}")
print(f"Cash Ratio: {cash_ratio(cash, marketable_securities, current_liabilities):.2f}")
print(f"Debt-to-Equity: {debt_to_equity(total_debt, total_equity):.2f}")
print(f"Working Capital: ${working_capital(current_assets, current_liabilities):,.0f}")
print(f"Book Value/Share: ${book_value_per_share(total_equity, shares_outstanding):.2f}")
```

### Cash Flow Statement

The cash flow statement shows cash inflows and outflows from operating, investing, and financing activities.

**Key Metrics:**

```python
# Cash Flow Analysis

# Operating Cash Flow Ratio
def operating_cash_flow_ratio(operating_cash_flow, current_liabilities):
    """Measures ability to pay short-term obligations"""
    return operating_cash_flow / current_liabilities

# Free Cash Flow
def free_cash_flow(operating_cash_flow, capital_expenditures):
    """Cash available after maintaining/expanding asset base"""
    return operating_cash_flow - capital_expenditures

# Free Cash Flow to Equity
def fcf_to_equity(fcf, net_borrowing):
    """Cash available to equity holders"""
    return fcf + net_borrowing

# Cash Flow Margin
def cash_flow_margin(operating_cash_flow, revenue):
    """Percentage of revenue converted to cash"""
    return (operating_cash_flow / revenue) * 100

# Cash Return on Assets
def cash_return_on_assets(operating_cash_flow, total_assets):
    """Cash generation efficiency"""
    return (operating_cash_flow / total_assets) * 100

# Example
operating_cash_flow = 180000
investing_cash_flow = -60000  # negative = cash outflow
financing_cash_flow = -40000
capital_expenditures = 50000
revenue = 1000000
total_assets = 2000000

fcf = free_cash_flow(operating_cash_flow, capital_expenditures)

print(f"Free Cash Flow: ${fcf:,.0f}")
print(f"Cash Flow Margin: {cash_flow_margin(operating_cash_flow, revenue):.2f}%")
print(f"Cash Return on Assets: {cash_return_on_assets(operating_cash_flow, total_assets):.2f}%")
```

## Comprehensive Financial Ratios

### Profitability Ratios

```python
# Return on Equity (ROE)
def roe(net_income, shareholders_equity):
    """Measures return generated on equity investment"""
    return (net_income / shareholders_equity) * 100

# Return on Assets (ROA)
def roa(net_income, total_assets):
    """Measures efficiency of asset utilization"""
    return (net_income / total_assets) * 100

# Return on Invested Capital (ROIC)
def roic(nopat, invested_capital):
    """Measures return on total capital invested
    NOPAT = Net Operating Profit After Tax
    Invested Capital = Total Debt + Total Equity - Cash
    """
    return (nopat / invested_capital) * 100

# DuPont Analysis (ROE decomposition)
def dupont_analysis(net_margin, asset_turnover, equity_multiplier):
    """ROE = Net Margin × Asset Turnover × Equity Multiplier"""
    return net_margin * asset_turnover * equity_multiplier

# Example
net_income = 144000
shareholders_equity = 1200000
total_assets = 2000000
revenue = 1000000
nopat = 120000
invested_capital = 1600000

print(f"ROE: {roe(net_income, shareholders_equity):.2f}%")
print(f"ROA: {roa(net_income, total_assets):.2f}%")
print(f"ROIC: {roic(nopat, invested_capital):.2f}%")

# DuPont
net_margin = net_income / revenue
asset_turnover = revenue / total_assets
equity_multiplier = total_assets / shareholders_equity
print(f"DuPont ROE: {dupont_analysis(net_margin, asset_turnover, equity_multiplier)*100:.2f}%")
```

### Efficiency Ratios

```python
# Asset Turnover Ratio
def asset_turnover(revenue, average_total_assets):
    """Measures efficiency of asset utilization"""
    return revenue / average_total_assets

# Inventory Turnover
def inventory_turnover(cogs, average_inventory):
    """How many times inventory is sold and replaced"""
    return cogs / average_inventory

# Days Inventory Outstanding (DIO)
def days_inventory_outstanding(average_inventory, cogs):
    """Average days to sell inventory"""
    return (average_inventory / cogs) * 365

# Receivables Turnover
def receivables_turnover(revenue, average_accounts_receivable):
    """How many times receivables are collected"""
    return revenue / average_accounts_receivable

# Days Sales Outstanding (DSO)
def days_sales_outstanding(average_accounts_receivable, revenue):
    """Average days to collect payment"""
    return (average_accounts_receivable / revenue) * 365

# Payables Turnover
def payables_turnover(cogs, average_accounts_payable):
    """How many times payables are paid"""
    return cogs / average_accounts_payable

# Days Payable Outstanding (DPO)
def days_payable_outstanding(average_accounts_payable, cogs):
    """Average days to pay suppliers"""
    return (average_accounts_payable / cogs) * 365

# Cash Conversion Cycle
def cash_conversion_cycle(dio, dso, dpo):
    """Days to convert investments back to cash"""
    return dio + dso - dpo

# Example
revenue = 1000000
cogs = 600000
average_inventory = 120000
average_receivables = 80000
average_payables = 60000
average_assets = 1900000

dio = days_inventory_outstanding(average_inventory, cogs)
dso = days_sales_outstanding(average_receivables, revenue)
dpo = days_payable_outstanding(average_payables, cogs)

print(f"Asset Turnover: {asset_turnover(revenue, average_assets):.2f}x")
print(f"Inventory Turnover: {inventory_turnover(cogs, average_inventory):.2f}x")
print(f"Days Inventory Outstanding: {dio:.1f} days")
print(f"Days Sales Outstanding: {dso:.1f} days")
print(f"Days Payable Outstanding: {dpo:.1f} days")
print(f"Cash Conversion Cycle: {cash_conversion_cycle(dio, dso, dpo):.1f} days")
```

### Leverage Ratios

```python
# Interest Coverage Ratio
def interest_coverage(ebit, interest_expense):
    """Ability to pay interest on debt"""
    return ebit / interest_expense

# Debt Service Coverage Ratio
def debt_service_coverage(operating_income, total_debt_service):
    """Ability to pay all debt obligations"""
    return operating_income / total_debt_service

# Equity Multiplier
def equity_multiplier(total_assets, total_equity):
    """Financial leverage measure"""
    return total_assets / total_equity

# Example
ebit = 180000
interest_expense = 20000
operating_income = 200000
debt_service = 80000

print(f"Interest Coverage: {interest_coverage(ebit, interest_expense):.2f}x")
print(f"Debt Service Coverage: {debt_service_coverage(operating_income, debt_service):.2f}x")
```

## Valuation Methods

### 1. Discounted Cash Flow (DCF) Analysis

DCF is the most comprehensive valuation method, estimating intrinsic value based on projected future cash flows.

```python
# DCF Valuation

def calculate_wacc(equity_value, debt_value, cost_of_equity, cost_of_debt, tax_rate):
    """Weighted Average Cost of Capital"""
    total_value = equity_value + debt_value
    equity_weight = equity_value / total_value
    debt_weight = debt_value / total_value
    return (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - tax_rate))

def discount_cash_flows(cash_flows, discount_rate):
    """Calculate present value of cash flows"""
    pv = 0
    for year, cf in enumerate(cash_flows, start=1):
        pv += cf / ((1 + discount_rate) ** year)
    return pv

def terminal_value_gordon_growth(final_year_fcf, discount_rate, perpetual_growth_rate):
    """Terminal value using Gordon Growth Model"""
    return (final_year_fcf * (1 + perpetual_growth_rate)) / (discount_rate - perpetual_growth_rate)

def terminal_value_exit_multiple(final_year_ebitda, exit_multiple):
    """Terminal value using exit multiple"""
    return final_year_ebitda * exit_multiple

def dcf_valuation(fcf_projections, terminal_value, discount_rate, net_debt, shares_outstanding):
    """Complete DCF valuation"""
    # Present value of projected cash flows
    pv_fcf = discount_cash_flows(fcf_projections, discount_rate)

    # Present value of terminal value
    years = len(fcf_projections)
    pv_terminal = terminal_value / ((1 + discount_rate) ** years)

    # Enterprise value
    enterprise_value = pv_fcf + pv_terminal

    # Equity value
    equity_value = enterprise_value - net_debt

    # Price per share
    price_per_share = equity_value / shares_outstanding

    return {
        'pv_fcf': pv_fcf,
        'pv_terminal': pv_terminal,
        'enterprise_value': enterprise_value,
        'equity_value': equity_value,
        'price_per_share': price_per_share
    }

# Example: DCF Valuation
# Company Assumptions
fcf_projections = [100000, 110000, 121000, 133100, 146410]  # 10% annual growth
final_year_fcf = fcf_projections[-1]
perpetual_growth = 0.03  # 3% perpetual growth
equity_value = 2000000
debt_value = 500000
cost_of_equity = 0.12  # 12%
cost_of_debt = 0.05  # 5%
tax_rate = 0.25
net_debt = 300000  # Total debt - cash
shares_outstanding = 100000

# Calculate WACC
wacc = calculate_wacc(equity_value, debt_value, cost_of_equity, cost_of_debt, tax_rate)
print(f"WACC: {wacc*100:.2f}%")

# Calculate terminal value
terminal_value = terminal_value_gordon_growth(final_year_fcf, wacc, perpetual_growth)
print(f"Terminal Value: ${terminal_value:,.0f}")

# DCF Valuation
valuation = dcf_valuation(fcf_projections, terminal_value, wacc, net_debt, shares_outstanding)
print(f"\nDCF Valuation Results:")
print(f"PV of Projected FCF: ${valuation['pv_fcf']:,.0f}")
print(f"PV of Terminal Value: ${valuation['pv_terminal']:,.0f}")
print(f"Enterprise Value: ${valuation['enterprise_value']:,.0f}")
print(f"Equity Value: ${valuation['equity_value']:,.0f}")
print(f"Intrinsic Value per Share: ${valuation['price_per_share']:.2f}")
```

### 2. Relative Valuation (Multiples)

```python
# Valuation Multiples

def pe_ratio(stock_price, earnings_per_share):
    """Price-to-Earnings Ratio"""
    return stock_price / earnings_per_share

def forward_pe(stock_price, forward_eps):
    """Forward P/E using projected earnings"""
    return stock_price / forward_eps

def pb_ratio(stock_price, book_value_per_share):
    """Price-to-Book Ratio"""
    return stock_price / book_value_per_share

def ps_ratio(market_cap, revenue):
    """Price-to-Sales Ratio"""
    return market_cap / revenue

def peg_ratio(pe_ratio, earnings_growth_rate):
    """P/E to Growth - accounts for growth"""
    return pe_ratio / earnings_growth_rate

def ev_ebitda(enterprise_value, ebitda):
    """Enterprise Value to EBITDA"""
    return enterprise_value / ebitda

def ev_sales(enterprise_value, revenue):
    """Enterprise Value to Sales"""
    return enterprise_value / revenue

# Comparable Company Analysis
def comparable_valuation(target_metric, comparable_multiples):
    """Value based on comparable company multiples"""
    avg_multiple = sum(comparable_multiples) / len(comparable_multiples)
    return target_metric * avg_multiple

# Example: Valuation Multiples
stock_price = 50
eps = 3.5
forward_eps = 4.0
book_value_ps = 25
market_cap = 5000000
revenue = 1000000
earnings_growth = 15  # 15% growth
enterprise_value = 5500000
ebitda = 200000

print(f"P/E Ratio: {pe_ratio(stock_price, eps):.2f}")
print(f"Forward P/E: {forward_pe(stock_price, forward_eps):.2f}")
print(f"P/B Ratio: {pb_ratio(stock_price, book_value_ps):.2f}")
print(f"P/S Ratio: {ps_ratio(market_cap, revenue):.2f}")
print(f"PEG Ratio: {peg_ratio(pe_ratio(stock_price, eps), earnings_growth):.2f}")
print(f"EV/EBITDA: {ev_ebitda(enterprise_value, ebitda):.2f}")

# Comparable Company Valuation
target_ebitda = 200000
comparable_ev_ebitda = [12.5, 14.0, 13.5, 15.0, 13.0]
implied_ev = comparable_valuation(target_ebitda, comparable_ev_ebitda)
print(f"\nImplied Enterprise Value (based on comps): ${implied_ev:,.0f}")
```

### 3. Dividend Discount Model (DDM)

```python
# Dividend Discount Models

def gordon_growth_model(dividend, growth_rate, required_return):
    """Constant growth DDM"""
    return dividend / (required_return - growth_rate)

def two_stage_ddm(initial_dividend, high_growth_rate, high_growth_years,
                  stable_growth_rate, required_return):
    """Two-stage growth model"""
    # High growth period
    pv_high_growth = 0
    for year in range(1, high_growth_years + 1):
        dividend = initial_dividend * ((1 + high_growth_rate) ** year)
        pv_high_growth += dividend / ((1 + required_return) ** year)

    # Terminal value at start of stable growth
    terminal_dividend = initial_dividend * ((1 + high_growth_rate) ** high_growth_years)
    terminal_value = gordon_growth_model(terminal_dividend * (1 + stable_growth_rate),
                                        stable_growth_rate, required_return)
    pv_terminal = terminal_value / ((1 + required_return) ** high_growth_years)

    return pv_high_growth + pv_terminal

# Example: DDM
current_dividend = 2.0
dividend_growth = 0.05  # 5% growth
required_return = 0.10  # 10% required return

intrinsic_value = gordon_growth_model(current_dividend * (1 + dividend_growth),
                                      dividend_growth, required_return)
print(f"Gordon Growth Model Value: ${intrinsic_value:.2f}")

# Two-stage model
initial_div = 2.0
high_growth = 0.15  # 15% for 5 years
high_years = 5
stable_growth = 0.04  # 4% thereafter
required_ret = 0.11

two_stage_value = two_stage_ddm(initial_div, high_growth, high_years,
                                stable_growth, required_ret)
print(f"Two-Stage DDM Value: ${two_stage_value:.2f}")
```

## Real-World Case Study: Complete Company Analysis

Let's perform a comprehensive fundamental analysis of a hypothetical company:

```python
# Case Study: TechCorp Inc.

# Company Overview
company_name = "TechCorp Inc."
sector = "Technology - Software"
market_cap = 10_000_000_000  # $10 billion

# Income Statement (in millions)
revenue = 2000
cogs = 600
rd_expense = 300
sales_marketing = 400
general_admin = 150
depreciation = 50
interest_expense = 30
tax_rate = 0.21

gross_profit = revenue - cogs
operating_expenses = rd_expense + sales_marketing + general_admin + depreciation
operating_income = gross_profit - operating_expenses
ebit = operating_income
ebitda = ebit + depreciation
ebt = ebit - interest_expense
taxes = ebt * tax_rate
net_income = ebt - taxes

# Balance Sheet (in millions)
cash = 500
receivables = 300
inventory = 100
current_assets_total = 1000
ppe = 800
intangibles = 1200
total_assets = 3000

accounts_payable = 200
short_term_debt = 150
current_liabilities_total = 400
long_term_debt = 600
total_liabilities = 1000
shareholders_equity = 2000

# Cash Flow (in millions)
operating_cash_flow = 550
capex = 100
fcf = operating_cash_flow - capex

# Share Information
shares_outstanding = 200  # million
current_stock_price = 50

# Market Data
risk_free_rate = 0.04
market_return = 0.10
beta = 1.2

print("=" * 60)
print(f"FUNDAMENTAL ANALYSIS: {company_name}")
print("=" * 60)

# 1. Profitability Analysis
print("\n1. PROFITABILITY METRICS")
print("-" * 40)
print(f"Gross Margin: {(gross_profit/revenue)*100:.1f}%")
print(f"Operating Margin: {(operating_income/revenue)*100:.1f}%")
print(f"Net Margin: {(net_income/revenue)*100:.1f}%")
print(f"ROE: {(net_income/shareholders_equity)*100:.1f}%")
print(f"ROA: {(net_income/total_assets)*100:.1f}%")

# 2. Liquidity Analysis
print("\n2. LIQUIDITY METRICS")
print("-" * 40)
print(f"Current Ratio: {current_assets_total/current_liabilities_total:.2f}")
print(f"Quick Ratio: {(current_assets_total-inventory)/current_liabilities_total:.2f}")
print(f"Cash Ratio: {cash/current_liabilities_total:.2f}")

# 3. Leverage Analysis
print("\n3. LEVERAGE METRICS")
print("-" * 40)
total_debt = short_term_debt + long_term_debt
print(f"Debt-to-Equity: {total_debt/shareholders_equity:.2f}")
print(f"Debt-to-Assets: {total_debt/total_assets:.2f}")
print(f"Interest Coverage: {ebit/interest_expense:.2f}x")

# 4. Valuation Metrics
print("\n4. VALUATION METRICS")
print("-" * 40)
eps = net_income / shares_outstanding
book_value_per_share = shareholders_equity / shares_outstanding
print(f"EPS: ${eps:.2f}")
print(f"P/E Ratio: {current_stock_price/eps:.2f}")
print(f"P/B Ratio: {current_stock_price/book_value_per_share:.2f}")
print(f"P/S Ratio: {market_cap/revenue:.2f}")
print(f"EV/EBITDA: {(market_cap + total_debt - cash)/ebitda:.2f}")

# 5. DCF Valuation
print("\n5. DCF VALUATION")
print("-" * 40)

# Calculate cost of equity using CAPM
cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)
cost_of_debt = interest_expense / total_debt
wacc = ((shareholders_equity / (shareholders_equity + total_debt)) * cost_of_equity +
        (total_debt / (shareholders_equity + total_debt)) * cost_of_debt * (1 - tax_rate))

# Project FCF (assuming 10% growth for 5 years, then 3% perpetual)
fcf_growth_rate = 0.10
projection_years = 5
fcf_projections = [fcf * ((1 + fcf_growth_rate) ** year) for year in range(1, projection_years + 1)]

# Terminal value
perpetual_growth = 0.03
terminal_fcf = fcf_projections[-1]
terminal_value = terminal_fcf * (1 + perpetual_growth) / (wacc - perpetual_growth)

# Calculate present values
pv_fcf = sum([cf / ((1 + wacc) ** year) for year, cf in enumerate(fcf_projections, 1)])
pv_terminal = terminal_value / ((1 + wacc) ** projection_years)

enterprise_value = pv_fcf + pv_terminal
equity_value = enterprise_value - (total_debt - cash)
intrinsic_value_per_share = equity_value / shares_outstanding

print(f"WACC: {wacc*100:.2f}%")
print(f"Enterprise Value: ${enterprise_value:,.0f}M")
print(f"Equity Value: ${equity_value:,.0f}M")
print(f"Intrinsic Value per Share: ${intrinsic_value_per_share:.2f}")
print(f"Current Stock Price: ${current_stock_price:.2f}")
print(f"Upside/Downside: {((intrinsic_value_per_share/current_stock_price)-1)*100:+.1f}%")

# 6. Investment Recommendation
print("\n6. INVESTMENT RECOMMENDATION")
print("-" * 40)
margin_of_safety = ((intrinsic_value_per_share - current_stock_price) / intrinsic_value_per_share) * 100

if margin_of_safety > 20:
    recommendation = "STRONG BUY"
elif margin_of_safety > 10:
    recommendation = "BUY"
elif margin_of_safety > -10:
    recommendation = "HOLD"
elif margin_of_safety > -20:
    recommendation = "SELL"
else:
    recommendation = "STRONG SELL"

print(f"Margin of Safety: {margin_of_safety:.1f}%")
print(f"Recommendation: {recommendation}")

print("\n" + "=" * 60)
```

## Key Takeaways

### Strengths of Fundamental Analysis:
- **Long-term Focus**: Identifies companies with sustainable competitive advantages
- **Value Discovery**: Helps find undervalued or overvalued securities
- **Comprehensive View**: Considers multiple factors affecting intrinsic value
- **Risk Assessment**: Evaluates financial health and stability

### Limitations:
- **Time-Intensive**: Requires extensive research and analysis
- **Assumption-Dependent**: DCF and other models rely on assumptions about future performance
- **Market Timing**: Fundamental value may take time to be recognized by the market
- **Qualitative Factors**: Difficult to quantify management quality, brand value, etc.

### Best Practices:
1. **Use Multiple Valuation Methods**: Don't rely on a single metric or model
2. **Compare to Peers**: Benchmark against industry averages and competitors
3. **Consider Economic Context**: Factor in macroeconomic conditions and industry trends
4. **Margin of Safety**: Buy with a buffer between price and intrinsic value
5. **Monitor Continuously**: Update analysis as new information becomes available
6. **Quality Over Quantity**: Focus on companies with strong fundamentals
7. **Understand the Business**: Invest in businesses you understand

## Additional Resources

**For deeper analysis, see also:**
- [Stock Market Fundamentals](./stocks.md) - Basic stock metrics and ratios
- [Technical Analysis](./technical_analysis.md) - Price patterns and indicators
- [Main Finance Guide](./README.md) - Comprehensive finance overview

**Financial Statement Sources:**
- Company 10-K and 10-Q filings (SEC EDGAR)
- Annual reports and earnings presentations
- Financial data platforms (Bloomberg, FactSet, Yahoo Finance)

**Valuation Databases:**
- Morningstar for valuation metrics
- Seeking Alpha for analysis and research
- GuruFocus for value investing metrics

---

*Note: This guide is for educational purposes only and does not constitute financial advice. Always conduct thorough research and consider consulting with financial professionals before making investment decisions.*

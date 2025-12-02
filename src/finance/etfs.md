# Exchange-Traded Funds (ETFs)

## Overview

Exchange-Traded Funds (ETFs) are investment funds that trade on stock exchanges like individual stocks. They offer diversification, low costs, transparency, and flexibility, making them popular investment vehicles for both individual and institutional investors.

## What is an ETF?

An ETF is a basket of securities that tracks an index, sector, commodity, or other assets. Unlike mutual funds, ETFs trade throughout the day at market prices.

**Key Characteristics:**
- Trade on exchanges like stocks
- Priced and traded throughout trading day
- Can be bought/sold at market prices
- Typically lower expense ratios than mutual funds
- Tax efficient structure
- Transparent holdings (usually daily disclosure)

## ETF vs. Mutual Fund

```python
def compare_etf_mutual_fund():
    """Compare key differences"""
    comparison = {
        'Trading': {
            'ETF': 'Trade all day at market prices',
            'Mutual Fund': 'Trade once per day at NAV'
        },
        'Minimum Investment': {
            'ETF': 'Price of 1 share',
            'Mutual Fund': 'Often $1,000-$3,000 minimum'
        },
        'Expense Ratio': {
            'ETF': 'Typically 0.05%-0.75%',
            'Mutual Fund': 'Typically 0.50%-2.00%'
        },
        'Tax Efficiency': {
            'ETF': 'Generally more tax efficient',
            'Mutual Fund': 'Can generate capital gains distributions'
        },
        'Trading Costs': {
            'ETF': 'Brokerage commission (often free)',
            'Mutual Fund': 'May have load fees'
        },
        'Transparency': {
            'ETF': 'Daily holdings disclosure',
            'Mutual Fund': 'Quarterly holdings disclosure'
        }
    }
    return comparison

comparison = compare_etf_mutual_fund()
for feature, details in comparison.items():
    print(f"\n{feature}:")
    for product, characteristic in details.items():
        print(f"  {product}: {characteristic}")
```

## Types of ETFs

### 1. Index ETFs

Track market indices like S&P 500, NASDAQ-100.

**Examples:**
- **SPY**: SPDR S&P 500 ETF Trust
- **QQQ**: Invesco QQQ (NASDAQ-100)
- **VTI**: Vanguard Total Stock Market ETF
- **IWM**: iShares Russell 2000 ETF

```python
def index_etf_characteristics():
    """Key characteristics of index ETFs"""
    return {
        'objective': 'Match index performance',
        'management': 'Passive',
        'expense_ratio': '0.03% - 0.20%',
        'tracking_error': 'Very low',
        'holdings': 'Matches index constituents',
        'best_for': 'Core portfolio holdings'
    }
```

### 2. Sector ETFs

Focus on specific industry sectors.

**Examples:**
- **XLK**: Technology Select Sector SPDR
- **XLV**: Health Care Select Sector SPDR
- **XLF**: Financial Select Sector SPDR
- **XLE**: Energy Select Sector SPDR

```python
def major_sector_etfs():
    """Common sector ETF categories"""
    sectors = {
        'Technology': ['XLK', 'VGT', 'FTEC'],
        'Healthcare': ['XLV', 'VHT', 'IYH'],
        'Financials': ['XLF', 'VFH', 'IYF'],
        'Energy': ['XLE', 'VDE', 'IYE'],
        'Consumer Discretionary': ['XLY', 'VCR', 'IYC'],
        'Consumer Staples': ['XLP', 'VDC', 'IYK'],
        'Industrials': ['XLI', 'VIS', 'IYJ'],
        'Materials': ['XLB', 'VAW', 'IYM'],
        'Utilities': ['XLU', 'VPU', 'IDU'],
        'Real Estate': ['XLRE', 'VNQ', 'IYR'],
        'Communication': ['XLC', 'VOX', 'IYZ']
    }
    return sectors

sectors = major_sector_etfs()
print("Major Sector ETFs:")
for sector, tickers in sectors.items():
    print(f"  {sector}: {', '.join(tickers)}")
```

### 3. Bond ETFs

Provide fixed income exposure.

**Types:**
- **Government**: AGG, BND, GOVT
- **Corporate**: LQD, VCIT, USIG
- **High Yield**: HYG, JNK
- **Municipal**: MUB, VTEB
- **International**: BNDX, IAGG
- **Treasury**: IEF (7-10yr), TLT (20+yr), SHY (1-3yr)

```python
def bond_etf_ladder():
    """Create bond ladder with ETFs"""
    ladder = {
        'Short-Term (1-3yr)': ['SHY', 'VGSH'],
        'Intermediate (3-10yr)': ['IEI', 'VGIT'],
        'Long-Term (10-30yr)': ['TLT', 'VGLT'],
        'Total Bond Market': ['AGG', 'BND'],
        'Corporate': ['LQD', 'VCIT'],
        'High Yield': ['HYG', 'JNK']
    }
    return ladder
```

### 4. International ETFs

Exposure to foreign markets.

**Developed Markets:**
- **VXUS**: Vanguard Total International Stock
- **VEA**: Vanguard FTSE Developed Markets
- **EFA**: iShares MSCI EAFE
- **VGK**: Vanguard FTSE Europe

**Emerging Markets:**
- **VWO**: Vanguard FTSE Emerging Markets
- **IEMG**: iShares Core MSCI Emerging Markets
- **EEM**: iShares MSCI Emerging Markets

**Country-Specific:**
- **EWJ**: iShares MSCI Japan
- **EWG**: iShares MSCI Germany
- **EWC**: iShares MSCI Canada
- **INDA**: iShares MSCI India

### 5. Commodity ETFs

Track commodities or commodity indices.

**Types:**
- **Gold**: GLD, IAU, GLDM
- **Silver**: SLV
- **Oil**: USO (crude), UNG (natural gas)
- **Broad Commodities**: DBC, GSG

```python
def commodity_etf_types():
    """Different commodity ETF structures"""
    return {
        'Physical Backed': {
            'description': 'Holds actual commodity',
            'examples': 'GLD (gold), SLV (silver)',
            'pros': 'Direct exposure',
            'cons': 'Storage costs'
        },
        'Futures Based': {
            'description': 'Holds futures contracts',
            'examples': 'USO (oil), UNG (gas)',
            'pros': 'Liquid, easy to trade',
            'cons': 'Contango/backwardation effects'
        },
        'Equity Based': {
            'description': 'Holds commodity company stocks',
            'examples': 'GDX (gold miners), XLE (energy)',
            'pros': 'Dividend income',
            'cons': 'Not pure commodity play'
        }
    }
```

### 6. Smart Beta/Factor ETFs

Weight holdings based on factors beyond market cap.

**Factors:**
- **Value**: RPV, VTV, IVE
- **Growth**: RPG, VUG, IVW
- **Dividend**: VYM, SCHD, VIG
- **Momentum**: MTUM, PDP
- **Quality**: QUAL, JQUA
- **Low Volatility**: USMV, SPLV

```python
def factor_investing_etfs():
    """Common investment factors"""
    factors = {
        'Value': {
            'metric': 'Low P/E, P/B ratios',
            'rationale': 'Value stocks outperform over time',
            'etfs': ['VTV', 'IVE', 'RPV']
        },
        'Momentum': {
            'metric': 'Recent price performance',
            'rationale': 'Winners keep winning (short-term)',
            'etfs': ['MTUM', 'PDP', 'QMOM']
        },
        'Quality': {
            'metric': 'ROE, earnings stability, low debt',
            'rationale': 'Quality companies outperform',
            'etfs': ['QUAL', 'JQUA', 'SPHQ']
        },
        'Size': {
            'metric': 'Market capitalization',
            'rationale': 'Small cap premium',
            'etfs': ['VB', 'IWM', 'SCHA']
        },
        'Low Volatility': {
            'metric': 'Lower price volatility',
            'rationale': 'Defensive, lower drawdowns',
            'etfs': ['USMV', 'SPLV', 'EEMV']
        },
        'Dividend': {
            'metric': 'Dividend yield, growth',
            'rationale': 'Income + growth',
            'etfs': ['VYM', 'SCHD', 'VIG']
        }
    }
    return factors
```

### 7. Leveraged and Inverse ETFs

Amplify or inverse market movements.

**Warning: High risk, short-term trading tools only!**

```python
def leveraged_inverse_etfs():
    """
    WARNING: These are trading tools, not buy-and-hold investments
    Daily rebalancing causes decay in sideways markets
    """
    examples = {
        'Leveraged Long (2x)': {
            'S&P 500': 'SSO (2x)',
            'NASDAQ': 'QLD (2x)',
            'Risk': 'Amplified losses, volatility decay'
        },
        'Leveraged Long (3x)': {
            'S&P 500': 'UPRO (3x)',
            'NASDAQ': 'TQQQ (3x)',
            'Risk': 'Extreme risk, can lose >90% in crash'
        },
        'Inverse (Short)': {
            'S&P 500': 'SH (-1x), SDS (-2x)',
            'NASDAQ': 'PSQ (-1x), QID (-2x)',
            'Risk': 'Loses money in rising markets'
        }
    }

    warnings = [
        'NOT for buy-and-hold investing',
        'Daily rebalancing causes tracking error',
        'Volatility decay in sideways markets',
        'Can lose more than 100% of investment (3x)',
        'High expense ratios (0.75% - 1.00%)',
        'Best for day trading only'
    ]

    return {'examples': examples, 'warnings': warnings}

lev_etfs = leveraged_inverse_etfs()
print("LEVERAGED/INVERSE ETF WARNINGS:")
for warning in lev_etfs['warnings']:
    print(f"  ⚠ {warning}")
```

### 8. Thematic ETFs

Focus on specific themes or trends.

**Examples:**
- **ARKK**: ARK Innovation ETF
- **BOTZ**: Global Robotics & AI
- **ICLN**: Clean Energy
- **HACK**: Cybersecurity
- **MOON**: Space Exploration
- **TAN**: Solar Energy
- **BLOK**: Blockchain

### 9. ESG ETFs

Environmental, Social, Governance focus.

**Examples:**
- **ESGU**: iShares ESG Aware MSCI USA
- **VSGX**: Vanguard ESG International Stock
- **SUSL**: iShares ESG Aware MSCI USA
- **ESGV**: Vanguard ESG U.S. Stock

## ETF Costs and Expenses

### Expense Ratios

```python
def etf_cost_comparison():
    """Compare typical expense ratios"""
    costs = {
        'Broad Market Index': {
            'range': '0.03% - 0.10%',
            'example': 'VTI: 0.03%',
            'annual_cost_on_10k': 3
        },
        'Sector ETF': {
            'range': '0.10% - 0.40%',
            'example': 'XLK: 0.10%',
            'annual_cost_on_10k': 10
        },
        'International': {
            'range': '0.05% - 0.30%',
            'example': 'VXUS: 0.07%',
            'annual_cost_on_10k': 7
        },
        'Smart Beta': {
            'range': '0.15% - 0.60%',
            'example': 'QUAL: 0.15%',
            'annual_cost_on_10k': 15
        },
        'Leveraged': {
            'range': '0.75% - 1.00%',
            'example': 'TQQQ: 0.95%',
            'annual_cost_on_10k': 95
        },
        'Thematic': {
            'range': '0.40% - 0.75%',
            'example': 'ARKK: 0.75%',
            'annual_cost_on_10k': 75
        }
    }
    return costs

def calculate_expense_impact(investment, expense_ratio, years, annual_return=0.08):
    """
    Calculate long-term impact of expense ratio

    Args:
        investment: Initial investment
        expense_ratio: Annual expense ratio (as decimal)
        years: Investment period
        annual_return: Expected gross return
    """
    # With expenses
    net_return = annual_return - expense_ratio
    final_with_expenses = investment * (1 + net_return) ** years

    # Without expenses
    final_without_expenses = investment * (1 + annual_return) ** years

    cost_of_expenses = final_without_expenses - final_with_expenses

    return {
        'final_value_with_expenses': final_with_expenses,
        'final_value_no_expenses': final_without_expenses,
        'total_expense_cost': cost_of_expenses,
        'expense_drag_pct': (cost_of_expenses / final_without_expenses) * 100
    }

# Example: $10,000 over 30 years
low_cost = calculate_expense_impact(10000, 0.0003, 30, 0.08)   # 0.03% ER
high_cost = calculate_expense_impact(10000, 0.0075, 30, 0.08)  # 0.75% ER

print("Impact of Expense Ratios over 30 years:")
print(f"\nLow Cost ETF (0.03%):")
print(f"  Final Value: ${low_cost['final_value_with_expenses']:,.0f}")
print(f"  Expense Cost: ${low_cost['total_expense_cost']:,.0f}")

print(f"\nHigh Cost ETF (0.75%):")
print(f"  Final Value: ${high_cost['final_value_with_expenses']:,.0f}")
print(f"  Expense Cost: ${high_cost['total_expense_cost']:,.0f}")

difference = high_cost['total_expense_cost'] - low_cost['total_expense_cost']
print(f"\nExtra cost of high fees: ${difference:,.0f}")
```

### Trading Costs

```python
def etf_trading_costs(share_price, shares, commission=0, bid_ask_spread=0.01):
    """
    Calculate total trading costs

    Args:
        share_price: ETF price
        shares: Number of shares
        commission: Commission per trade (many brokers now $0)
        bid_ask_spread: Bid-ask spread per share
    """
    trade_value = share_price * shares
    spread_cost = bid_ask_spread * shares
    total_cost = commission + spread_cost

    cost_percentage = (total_cost / trade_value) * 100

    return {
        'trade_value': trade_value,
        'commission': commission,
        'spread_cost': spread_cost,
        'total_cost': total_cost,
        'cost_percentage': cost_percentage
    }

# Example
trade = etf_trading_costs(share_price=100, shares=100, commission=0, bid_ask_spread=0.05)
print(f"Trade Value: ${trade['trade_value']:,.2f}")
print(f"Spread Cost: ${trade['spread_cost']:.2f}")
print(f"Total Cost: ${trade['total_cost']:.2f} ({trade['cost_percentage']:.3f}%)")
```

## ETF Selection Criteria

### Key Metrics to Evaluate

```python
def evaluate_etf(ticker, expense_ratio, aum, avg_volume, tracking_error, spread):
    """
    Evaluate ETF quality

    Args:
        ticker: ETF ticker symbol
        expense_ratio: Annual expense ratio (%)
        aum: Assets under management (millions)
        avg_volume: Average daily trading volume
        tracking_error: Annual tracking error (%)
        spread: Typical bid-ask spread (%)
    """
    score = 0
    notes = []

    # Expense Ratio (lower is better)
    if expense_ratio < 0.10:
        score += 2
        notes.append("✓ Very low expense ratio")
    elif expense_ratio < 0.25:
        score += 1
        notes.append("✓ Low expense ratio")
    else:
        notes.append("⚠ Higher expense ratio")

    # AUM (bigger is generally better)
    if aum > 1000:
        score += 2
        notes.append("✓ Large AUM (good liquidity)")
    elif aum > 100:
        score += 1
        notes.append("✓ Adequate AUM")
    else:
        notes.append("⚠ Small AUM (liquidity risk)")

    # Trading Volume
    if avg_volume > 1000000:
        score += 2
        notes.append("✓ High trading volume")
    elif avg_volume > 100000:
        score += 1
        notes.append("✓ Adequate volume")
    else:
        notes.append("⚠ Low volume (wider spreads)")

    # Tracking Error (lower is better for index ETFs)
    if tracking_error < 0.20:
        score += 2
        notes.append("✓ Excellent tracking")
    elif tracking_error < 0.50:
        score += 1
        notes.append("✓ Good tracking")
    else:
        notes.append("⚠ Higher tracking error")

    # Bid-Ask Spread (lower is better)
    if spread < 0.05:
        score += 2
        notes.append("✓ Tight spread")
    elif spread < 0.10:
        score += 1
        notes.append("✓ Reasonable spread")
    else:
        notes.append("⚠ Wide spread (higher trading costs)")

    rating = "Excellent" if score >= 8 else "Good" if score >= 6 else "Fair" if score >= 4 else "Poor"

    return {
        'ticker': ticker,
        'score': score,
        'rating': rating,
        'notes': notes
    }

# Example
etf_eval = evaluate_etf(
    ticker='VTI',
    expense_ratio=0.03,
    aum=300000,  # $300B
    avg_volume=4000000,
    tracking_error=0.05,
    spread=0.01
)

print(f"\nETF Evaluation: {etf_eval['ticker']}")
print(f"Score: {etf_eval['score']}/10")
print(f"Rating: {etf_eval['rating']}")
print("\nDetails:")
for note in etf_eval['notes']:
    print(f"  {note}")
```

## ETF Investment Strategies

### Core-Satellite Strategy

```python
def core_satellite_etf_portfolio(portfolio_value):
    """
    Core-satellite strategy using ETFs

    Core (70-80%): Broad market index ETFs
    Satellite (20-30%): Tactical/thematic/sector ETFs
    """
    core_allocation = 0.75
    satellite_allocation = 0.25

    portfolio = {
        'Core (75%)': {
            'value': portfolio_value * core_allocation,
            'holdings': {
                'VTI (Total US Market)': 0.50,    # 37.5% of portfolio
                'VXUS (Total International)': 0.30,  # 22.5% of portfolio
                'BND (Total Bond)': 0.20           # 15% of portfolio
            }
        },
        'Satellite (25%)': {
            'value': portfolio_value * satellite_allocation,
            'holdings': {
                'QQQ (Tech Growth)': 0.30,        # 7.5% of portfolio
                'VYM (Dividend)': 0.25,           # 6.25% of portfolio
                'GLD (Gold)': 0.20,               # 5% of portfolio
                'ARKK (Innovation)': 0.15,        # 3.75% of portfolio
                'ICLN (Clean Energy)': 0.10       # 2.5% of portfolio
            }
        }
    }

    return portfolio

portfolio = core_satellite_etf_portfolio(100000)
print("Core-Satellite ETF Portfolio ($100,000):\n")
for section, details in portfolio.items():
    print(f"{section}: ${details['value']:,.0f}")
    for etf, weight in details['holdings'].items():
        dollar_amount = details['value'] * weight
        print(f"  {etf}: ${dollar_amount:,.0f}")
    print()
```

### Age-Based Allocation

```python
def age_based_etf_allocation(age, risk_tolerance='moderate'):
    """
    Age-based ETF allocation

    Rule of thumb: Bond % = Age (or 120 - Age for stocks)
    """
    if risk_tolerance == 'aggressive':
        stock_pct = min(100, 120 - age)
    elif risk_tolerance == 'moderate':
        stock_pct = 100 - age
    else:  # conservative
        stock_pct = max(30, 80 - age)

    bond_pct = 100 - stock_pct

    allocation = {
        'Stocks': {
            'percentage': stock_pct,
            'etfs': {
                'VTI (US Total Market)': 0.60,
                'VXUS (International)': 0.30,
                'VWO (Emerging Markets)': 0.10
            }
        },
        'Bonds': {
            'percentage': bond_pct,
            'etfs': {
                'BND (Total Bond)': 0.70,
                'VGIT (Intermediate Treasury)': 0.20,
                'TIP (TIPS)': 0.10
            }
        }
    }

    return allocation

# Example: 35-year-old, moderate risk
allocation = age_based_etf_allocation(35, 'moderate')
print(f"Age-Based Allocation (35 years old, moderate):\n")
for asset_class, details in allocation.items():
    print(f"{asset_class}: {details['percentage']}%")
    for etf, weight in details['etfs'].items():
        etf_pct = details['percentage'] * weight
        print(f"  {etf}: {etf_pct:.1f}%")
    print()
```

### Dividend Income Strategy

```python
def dividend_etf_portfolio():
    """
    High dividend yield ETF portfolio for income
    """
    portfolio = {
        'High Dividend Yield': {
            'VYM': {
                'name': 'Vanguard High Dividend Yield',
                'yield': 3.0,
                'allocation': 30
            },
            'SCHD': {
                'name': 'Schwab US Dividend Equity',
                'yield': 3.5,
                'allocation': 25
            }
        },
        'Dividend Growth': {
            'VIG': {
                'name': 'Vanguard Dividend Appreciation',
                'yield': 2.0,
                'allocation': 20
            },
            'DGRO': {
                'name': 'iShares Core Dividend Growth',
                'yield': 2.5,
                'allocation': 15
            }
        },
        'REITs': {
            'VNQ': {
                'name': 'Vanguard Real Estate',
                'yield': 4.0,
                'allocation': 10
            }
        }
    }

    def calculate_portfolio_yield(portfolio, investment=100000):
        total_yield = 0
        for category, etfs in portfolio.items():
            for ticker, details in etfs.items():
                weight = details['allocation'] / 100
                contribution = details['yield'] * weight
                total_yield += contribution

                investment_amount = investment * weight
                annual_income = investment_amount * (details['yield'] / 100)
                print(f"{ticker}: ${investment_amount:,.0f} @ {details['yield']:.1f}% = ${annual_income:,.0f}/year")

        total_annual_income = investment * (total_yield / 100)
        print(f"\nTotal Portfolio Yield: {total_yield:.2f}%")
        print(f"Annual Income on ${investment:,}: ${total_annual_income:,.0f}")

    return portfolio

portfolio = dividend_etf_portfolio()
print("Dividend Income ETF Portfolio:\n")
calculate_portfolio_yield(portfolio, 100000)
```

## Tax Considerations

### Tax Efficiency of ETFs

```python
def etf_tax_efficiency():
    """
    Why ETFs are more tax-efficient than mutual funds
    """
    return {
        'In-Kind Creation/Redemption': {
            'mechanism': 'Authorized Participants exchange securities, not cash',
            'benefit': 'Avoids triggering capital gains',
            'result': 'Fewer taxable distributions'
        },
        'Low Turnover': {
            'mechanism': 'Index ETFs rarely trade holdings',
            'benefit': 'Minimal capital gains',
            'result': 'Tax-deferred growth'
        },
        'Investor Control': {
            'mechanism': 'You control when to sell',
            'benefit': 'Tax-loss harvesting opportunities',
            'result': 'Optimize your tax situation'
        }
    }

def tax_loss_harvest_example():
    """Example of tax-loss harvesting with ETFs"""
    scenario = {
        'Position': 'VTI (Total Market)',
        'Cost Basis': 10000,
        'Current Value': 8500,
        'Loss': -1500,
        'Action': 'Sell VTI, immediately buy ITOT (similar but not identical)',
        'Tax Benefit': 1500 * 0.24,  # Assuming 24% tax bracket
        'Result': 'Keep market exposure, harvest tax loss'
    }
    return scenario

example = tax_loss_harvest_example()
print("Tax-Loss Harvesting Example:")
for key, value in example.items():
    if key == 'Tax Benefit':
        print(f"  {key}: ${value:.0f}")
    else:
        print(f"  {key}: {value}")
```

## Common ETF Mistakes

```python
def common_etf_mistakes():
    """Mistakes to avoid when investing in ETFs"""
    return {
        '1. Chasing Performance': {
            'mistake': 'Buying last year\'s top performers',
            'why_bad': 'Past performance doesn\'t predict future',
            'solution': 'Focus on asset allocation, not recent returns'
        },
        '2. Ignoring Expense Ratios': {
            'mistake': 'Not comparing costs between similar ETFs',
            'why_bad': 'High fees compound over time',
            'solution': 'Choose low-cost options for core holdings'
        },
        '3. Trading Too Much': {
            'mistake': 'Frequent buying/selling',
            'why_bad': 'Bid-ask spreads and taxes add up',
            'solution': 'Buy and hold, minimize trading'
        },
        '4. Leveraged ETFs Long-Term': {
            'mistake': 'Holding 2x/3x ETFs for months/years',
            'why_bad': 'Volatility decay destroys returns',
            'solution': 'Use only for day trading, if at all'
        },
        '5. Overlapping Holdings': {
            'mistake': 'Buying multiple similar ETFs',
            'why_bad': 'False diversification, extra fees',
            'solution': 'Check holdings overlap before buying'
        },
        '6. Ignoring Liquidity': {
            'mistake': 'Buying low-volume ETFs',
            'why_bad': 'Wide bid-ask spreads increase costs',
            'solution': 'Stick to ETFs with >100k daily volume'
        },
        '7. Thematic Overconcentration': {
            'mistake': 'Too much in trendy thematic ETFs',
            'why_bad': 'High risk, often poor timing',
            'solution': 'Limit thematic to <10% of portfolio'
        }
    }

mistakes = common_etf_mistakes()
print("Common ETF Investment Mistakes:\n")
for number, details in mistakes.items():
    print(f"{number}")
    for key, value in details.items():
        print(f"  {key}: {value}")
    print()
```

## Best Practices

### ETF Portfolio Guidelines

1. **Core holdings should be low-cost, broad market ETFs**
2. **Total expense ratio <0.20% for core holdings**
3. **AUM >$500M for liquidity**
4. **Daily volume >100,000 shares**
5. **Rebalance quarterly or when drift >5%**
6. **Tax-loss harvest annually**
7. **Avoid overlapping holdings**
8. **Limit thematic/sector ETFs to 20% of portfolio**

### Simple ETF Portfolios

```python
def simple_etf_portfolios():
    """
    Simple, low-cost ETF portfolio examples
    """
    portfolios = {
        'Three-Fund Portfolio': {
            'description': 'Classic Bogleheads approach',
            'allocation': {
                'VTI (US Total Stock Market)': 60,
                'VXUS (International Stocks)': 30,
                'BND (Total Bond Market)': 10
            },
            'expense_ratio': 0.05,
            'complexity': 'Very Simple',
            'rebalancing': 'Annual'
        },
        'Two-Fund Portfolio': {
            'description': 'Even simpler',
            'allocation': {
                'VT (Total World Stock)': 80,
                'BND (Total Bond Market)': 20
            },
            'expense_ratio': 0.06,
            'complexity': 'Simplest',
            'rebalancing': 'Annual'
        },
        'All-Weather Portfolio': {
            'description': 'Ray Dalio inspired',
            'allocation': {
                'VTI (US Stocks)': 30,
                'TLT (Long-Term Treasuries)': 40,
                'IEI (Intermediate Treasuries)': 15,
                'GLD (Gold)': 7.5,
                'DBC (Commodities)': 7.5
            },
            'expense_ratio': 0.15,
            'complexity': 'Moderate',
            'rebalancing': 'Quarterly'
        }
    }
    return portfolios

portfolios = simple_etf_portfolios()
print("Simple ETF Portfolio Examples:\n")
for name, details in portfolios.items():
    print(f"{name}")
    print(f"  {details['description']}")
    print(f"  Allocation:")
    for etf, pct in details['allocation'].items():
        print(f"    {etf}: {pct}%")
    print(f"  Avg Expense Ratio: {details['expense_ratio']}%")
    print()
```

## Resources

- **ETF.com**: Comprehensive ETF data and research
- **Morningstar**: ETF analysis and ratings
- **ETFdb.com**: ETF database and screener
- **Vanguard/iShares/Schwab**: Major ETF providers
- **"The Bogleheads' Guide to Investing"**

## Key Takeaways

1. **ETFs offer low-cost diversification** - Often cheaper than mutual funds
2. **Index ETFs for core holdings** - Passive, low-cost, tax-efficient
3. **Keep it simple** - 3-5 ETFs can build complete portfolio
4. **Watch expense ratios** - They compound over time
5. **Consider tax efficiency** - ETFs are generally tax-advantaged
6. **Avoid leveraged ETFs** - Unless day trading
7. **Liquidity matters** - Stick to high-volume ETFs
8. **Rebalance systematically** - Maintain target allocation

## Next Steps

1. Define your asset allocation strategy
2. Select low-cost ETFs for each asset class
3. Compare expense ratios and liquidity
4. Open brokerage account (ideally with $0 commissions)
5. Buy ETFs and set up automatic investments if available
6. Rebalance quarterly or annually
7. Tax-loss harvest annually
8. Stay disciplined and avoid frequent trading

Remember: ETFs are tools. Focus on your overall asset allocation strategy, not on finding the "best" ETF. Low costs, broad diversification, and consistent execution matter more than picking the perfect funds.

# Commodities Trading

## Overview

Commodities are basic goods used in commerce that are interchangeable with other goods of the same type. They include precious metals, energy, agriculture, and livestock. Commodities provide diversification and inflation hedging in investment portfolios.

## Types of Commodities

### 1. Precious Metals
- **Gold (XAU)**: Store of value, safe haven
- **Silver (XAG)**: Industrial and investment demand
- **Platinum (XPT)**: Industrial metal
- **Palladium (XPD)**: Auto catalysts

### 2. Energy
- **Crude Oil (CL, WTI, Brent)**: Most traded commodity
- **Natural Gas (NG)**: Heating and power generation
- **Heating Oil**: Refined petroleum product
- **Gasoline (RBOB)**: Transportation fuel

### 3. Agriculture
- **Grains**: Corn, Wheat, Soybeans
- **Soft Commodities**: Coffee, Cotton, Sugar, Cocoa
- **Livestock**: Live Cattle, Lean Hogs, Feeder Cattle

### 4. Industrial Metals
- **Copper**: Economic indicator ("Dr. Copper")
- **Aluminum**: Construction, transportation
- **Zinc**: Galvanization
- **Nickel**: Stainless steel production

## Why Invest in Commodities?

```python
def commodity_investment_benefits():
    """Benefits and risks of commodity investing"""
    return {
        'Benefits': {
            'Inflation Hedge': 'Commodities typically rise with inflation',
            'Diversification': 'Low correlation with stocks/bonds',
            'Real Assets': 'Tangible goods with intrinsic value',
            'Supply/Demand': 'Clear fundamental drivers',
            'Global Growth Play': 'Benefit from emerging market growth'
        },
        'Risks': {
            'High Volatility': 'Price swings can be extreme',
            'No Income': 'No dividends or interest (except storage costs)',
            'Storage Costs': 'Physical commodities require storage',
            'Contango/Backwardation': 'Futures curve affects returns',
            'Geopolitical': 'Sensitive to global events',
            'Seasonality': 'Weather and seasonal patterns'
        }
    }

benefits = commodity_investment_benefits()
for category, items in benefits.items():
    print(f"\n{category}:")
    for item, description in items.items():
        print(f"  {item}: {description}")
```

## Ways to Invest in Commodities

### 1. Physical Ownership

```python
def physical_commodity_investment():
    """Pros and cons of physical ownership"""
    return {
        'Gold/Silver': {
            'methods': ['Bullion bars', 'Coins', 'Jewelry'],
            'pros': 'Direct ownership, no counterparty risk',
            'cons': 'Storage, insurance, liquidity, premiums over spot',
            'best_for': 'Long-term holders, crisis hedge'
        },
        'Agricultural': {
            'methods': 'Not practical for most investors',
            'pros': 'None for individuals',
            'cons': 'Storage, spoilage, logistics',
            'best_for': 'Commercial users only'
        },
        'Energy': {
            'methods': 'Not practical for individuals',
            'pros': 'None',
            'cons': 'Dangerous, expensive storage',
            'best_for': 'Not recommended'
        }
    }
```

### 2. Futures Contracts

```python
def commodity_futures_basics():
    """Understanding commodity futures"""
    return {
        'Definition': 'Contract to buy/sell commodity at future date',
        'Standardized': 'Exchange-traded, fixed quantity and quality',
        'Leverage': 'Control large amount with small margin',
        'Settlement': 'Cash or physical delivery',
        'Expiration': 'Contracts expire monthly/quarterly',

        'Example - Gold Futures': {
            'contract_size': '100 troy ounces',
            'tick_size': '$0.10 per ounce = $10 per contract',
            'margin': '$5,000 - $10,000 initial margin',
            'notional_value': '~$180,000 at $1,800/oz'
        },

        'Key Terms': {
            'Contango': 'Future price > Spot price (normal)',
            'Backwardation': 'Future price < Spot price (shortage)',
            'Roll Yield': 'Profit/loss from rolling contracts'
        }
    }

def futures_profit_loss(contracts, entry_price, exit_price,
                       contract_size, tick_size, tick_value):
    """
    Calculate futures P&L

    Args:
        contracts: Number of contracts
        entry_price: Entry price per unit
        exit_price: Exit price per unit
        contract_size: Units per contract
        tick_size: Minimum price movement
        tick_value: Dollar value of tick
    """
    price_change = exit_price - entry_price
    ticks = price_change / tick_size
    profit = ticks * tick_value * contracts

    return {
        'price_change': price_change,
        'ticks': ticks,
        'profit_loss': profit,
        'profit_per_contract': profit / contracts
    }

# Example: Gold futures (GC)
pnl = futures_profit_loss(
    contracts=1,
    entry_price=1800.0,
    exit_price=1850.0,
    contract_size=100,  # 100 oz
    tick_size=0.10,     # $0.10/oz
    tick_value=10       # $10 per tick
)

print("Gold Futures Trade:")
print(f"  Price change: ${pnl['price_change']:.2f}/oz")
print(f"  Ticks: {pnl['ticks']:.0f}")
print(f"  Profit: ${pnl['profit_loss']:,.0f}")
```

### 3. ETFs and ETNs

```python
def commodity_etf_types():
    """Types of commodity ETFs"""
    return {
        'Physical-Backed ETFs': {
            'examples': 'GLD (gold), SLV (silver), PALL (palladium)',
            'mechanism': 'Hold actual metal in vaults',
            'pros': 'Direct exposure, no roll yield',
            'cons': 'Storage fees (expense ratio)',
            'best_for': 'Precious metals investors'
        },
        'Futures-Based ETFs': {
            'examples': 'USO (oil), UNG (natural gas), DBA (agriculture)',
            'mechanism': 'Hold futures contracts, roll monthly',
            'pros': 'Liquid, easy to trade',
            'cons': 'Contango drag, tracking error',
            'best_for': 'Short-term tactical trades'
        },
        'Equity-Based ETFs': {
            'examples': 'GDX (gold miners), XLE (energy stocks)',
            'mechanism': 'Hold commodity company stocks',
            'pros': 'Dividends, leveraged to commodity prices',
            'cons': 'Company-specific risks, not pure commodity play',
            'best_for': 'Long-term commodity exposure'
        },
        'Broad Commodity ETFs': {
            'examples': 'DBC, GSG, PDBC',
            'mechanism': 'Basket of commodity futures',
            'pros': 'Diversified commodity exposure',
            'cons': 'Futures roll costs',
            'best_for': 'Portfolio diversification'
        }
    }

def contango_impact_example(spot_price, futures_price, months, initial_investment):
    """
    Demonstrate contango drag on futures-based ETF

    Contango: futures price > spot price
    As futures approach expiration, they converge to spot (roll down)
    """
    # Monthly roll cost
    monthly_roll_cost_pct = ((futures_price - spot_price) / spot_price) / months

    # Value after rolling for specified months
    final_value = initial_investment
    for month in range(months):
        # Lose roll cost each month
        final_value *= (1 - monthly_roll_cost_pct)

    total_loss = initial_investment - final_value
    loss_percentage = (total_loss / initial_investment) * 100

    return {
        'initial_investment': initial_investment,
        'spot_price': spot_price,
        'futures_price': futures_price,
        'contango_pct': ((futures_price - spot_price) / spot_price) * 100,
        'months': months,
        'monthly_drag': monthly_roll_cost_pct * 100,
        'final_value': final_value,
        'total_loss': total_loss,
        'loss_pct': loss_percentage
    }

# Example: Oil ETF in contango
contango = contango_impact_example(
    spot_price=70,      # Spot oil $70
    futures_price=77,   # 1-year futures $77 (10% contango)
    months=12,
    initial_investment=10000
)

print("Contango Drag Example (Oil ETF):")
print(f"  Spot Price: ${contango['spot_price']}")
print(f"  Futures Price: ${contango['futures_price']}")
print(f"  Contango: {contango['contango_pct']:.1f}%")
print(f"  Monthly Drag: {contango['monthly_drag']:.2f}%")
print(f"  Initial Investment: ${contango['initial_investment']:,}")
print(f"  Value After {contango['months']} Months: ${contango['final_value']:,.0f}")
print(f"  Loss from Rolling: ${contango['total_loss']:,.0f} ({contango['loss_pct']:.1f}%)")
print("\nNote: This assumes flat spot prices - additional losses if spot declines!")
```

### 4. Commodity Stocks

```python
def commodity_stocks_vs_commodities():
    """Compare investing in commodity stocks vs. commodities"""
    return {
        'Gold Miners vs. Gold': {
            'leverage': 'Miners are levered to gold price (2-3x)',
            'dividends': 'Miners pay dividends, gold doesn\'t',
            'risks': 'Operational, management, country risk',
            'example': 'Gold +10% → Miners +20-30% (or more)'
        },
        'Oil Companies vs. Oil': {
            'leverage': 'Moderate leverage to oil prices',
            'dividends': 'Many pay high dividends',
            'risks': 'Operational, regulatory, ESG concerns',
            'diversification': 'Integrated companies less volatile'
        },
        'Agricultural Companies': {
            'examples': 'Deere (equipment), Nutrien (fertilizer), ADM (processing)',
            'exposure': 'Indirect, diversified business models',
            'stability': 'More stable than raw commodities'
        }
    }
```

## Commodity Fundamentals

### Supply and Demand Analysis

```python
def commodity_supply_demand_factors():
    """Key factors affecting commodity prices"""
    return {
        'Supply Factors': {
            'Production': 'Mine output, oil wells, crop yields',
            'Weather': 'Droughts, floods affect agriculture',
            'Technology': 'Fracking revolution increased oil supply',
            'OPEC/Cartels': 'Coordinated production cuts/increases',
            'Geopolitics': 'Wars, sanctions disrupt supply',
            'Strikes': 'Labor disputes at mines',
            'Capex Cycles': 'Years to bring new supply online'
        },
        'Demand Factors': {
            'Economic Growth': 'China/India industrialization',
            'Seasonality': 'Heating oil in winter, gasoline in summer',
            'Currency': 'Weak USD → higher commodity prices',
            'Substitution': 'Electric vehicles reduce oil demand',
            'Technology': 'Solar panels use silver',
            'Speculation': 'Hedge fund positioning'
        }
    }

def commodity_seasonality():
    """Common seasonal patterns"""
    return {
        'Natural Gas': {
            'peak_demand': 'Winter (heating) and Summer (A/C)',
            'low_demand': 'Spring and Fall (shoulder seasons)',
            'storage': 'Build in summer, draw in winter'
        },
        'Crude Oil': {
            'peak_demand': 'Summer (driving season)',
            'low_demand': 'Winter',
            'refineries': 'Maintenance in spring/fall'
        },
        'Agriculture': {
            'planting_season': 'Spring uncertainty = higher prices',
            'harvest': 'Fall harvest = lower prices',
            'weather': 'June/July critical for US corn/soybeans'
        },
        'Gold': {
            'jewelry_demand': 'Peak in Q4 (holidays), Q1 (Chinese New Year)',
            'investment': 'Crisis periods, low real rates'
        }
    }
```

### Key Economic Indicators

```python
def commodity_economic_indicators():
    """Important reports for commodity traders"""
    return {
        'Energy': {
            'EIA Weekly Report': 'Wed 10:30am ET - Oil/gas inventories',
            'OPEC Meetings': 'Production quotas',
            'Baker Hughes Rig Count': 'Friday - Active drilling rigs'
        },
        'Agriculture': {
            'USDA Reports': 'Monthly crop reports, WASDE',
            'Planting/Harvest Progress': 'Weekly during season',
            'Export Sales': 'Thursday mornings'
        },
        'Metals': {
            'China PMI': 'Manufacturing activity (copper demand)',
            'Auto Sales': 'Platinum/palladium demand',
            'Jewelry Demand': 'Gold/silver seasonal patterns'
        },
        'General': {
            'USD Index': 'Inverse relationship with commodities',
            'Real Interest Rates': 'Higher rates = lower gold',
            'Global GDP': 'Economic growth = commodity demand'
        }
    }
```

## Commodity Trading Strategies

### Trend Following

```python
def commodity_trend_strategy():
    """Trend following in commodities"""
    return {
        'Rationale': 'Commodities trend strongly due to supply/demand imbalances',
        'Indicators': {
            'Moving Averages': '50/200 day crossovers',
            'Breakouts': 'Multi-year highs/lows',
            'Momentum': 'Price momentum indicators'
        },
        'Rules': {
            'Entry': 'Buy breakout above resistance + trend confirmation',
            'Stop Loss': 'Below recent swing low or MA',
            'Position Size': '1-2% risk per trade',
            'Exit': 'Trailing stop or trend reversal'
        },
        'Best Markets': 'Crude oil, gold, grains during trends',
        'Timeframe': 'Weekly/monthly charts for major trends'
    }
```

### Seasonal Trading

```python
def seasonal_commodity_trades():
    """Classic seasonal trades"""
    return {
        'Natural Gas - Summer Storage Build': {
            'trade': 'Short natural gas March-June',
            'rationale': 'Shoulder season, builds inventory',
            'historical_win_rate': '70%',
            'risks': 'Early heat wave'
        },
        'Crude Oil - Summer Driving': {
            'trade': 'Long crude/gasoline Feb-May',
            'rationale': 'Driving season demand',
            'historical_win_rate': '65%',
            'risks': 'Economic slowdown'
        },
        'Corn/Soybeans - Weather Premium': {
            'trade': 'Long May-July, sell at harvest',
            'rationale': 'Weather uncertainty premium',
            'historical_win_rate': '60%',
            'risks': 'Perfect weather = price crash'
        },
        'Gold - Q4 Jewelry Demand': {
            'trade': 'Long Aug-Nov',
            'rationale': 'Holiday season, Indian festivals',
            'historical_win_rate': '55-60%',
            'risks': 'Strong USD, higher rates'
        }
    }
```

### Spread Trading

```python
def commodity_spread_trades():
    """Spread trading strategies"""
    return {
        'Calendar Spreads': {
            'definition': 'Long one month, short another month',
            'example': 'Long Dec Corn, Short July Corn',
            'rationale': 'Storage costs, seasonal patterns',
            'advantage': 'Lower risk than outright position'
        },
        'Inter-Commodity Spreads': {
            'definition': 'Related commodities',
            'examples': [
                'Crack Spread: Crude vs. Gasoline+Heating Oil (refining margin)',
                'Crush Spread: Soybeans vs. Soy Oil + Soy Meal (processing margin)',
                'Gold/Silver Ratio: Gold price / Silver price'
            ],
            'rationale': 'Trade relationships, not direction'
        },
        'Location Spreads': {
            'definition': 'Same commodity, different locations',
            'example': 'WTI vs. Brent crude oil',
            'rationale': 'Transportation costs, local supply/demand'
        }
    }

def gold_silver_ratio_trade(gold_price, silver_price, historical_avg=70):
    """
    Gold/Silver ratio trading strategy

    When ratio high: Silver relatively cheap → Buy silver
    When ratio low: Gold relatively cheap → Buy gold
    """
    ratio = gold_price / silver_price

    if ratio > historical_avg * 1.2:  # 20% above average
        signal = 'BUY SILVER (or long silver, short gold)'
        rationale = f'Ratio {ratio:.1f} is high vs. avg {historical_avg}'
    elif ratio < historical_avg * 0.8:  # 20% below average
        signal = 'BUY GOLD (or long gold, short silver)'
        rationale = f'Ratio {ratio:.1f} is low vs. avg {historical_avg}'
    else:
        signal = 'NEUTRAL'
        rationale = f'Ratio {ratio:.1f} near historical avg {historical_avg}'

    return {
        'gold_price': gold_price,
        'silver_price': silver_price,
        'ratio': ratio,
        'historical_avg': historical_avg,
        'signal': signal,
        'rationale': rationale
    }

# Example
gs_ratio = gold_silver_ratio_trade(gold_price=1800, silver_price=22)
print("Gold/Silver Ratio Analysis:")
for key, value in gs_ratio.items():
    print(f"  {key}: {value}")
```

## Risk Management

### Position Sizing for Commodities

```python
def commodity_position_sizing(account_size, risk_per_trade, entry_price,
                              stop_loss_price, contract_size, tick_value):
    """
    Position sizing for commodity futures

    Args:
        account_size: Account balance
        risk_per_trade: Percentage to risk (e.g., 0.02 for 2%)
        entry_price: Entry price
        stop_loss_price: Stop loss price
        contract_size: Units per contract
        tick_value: Dollar value per tick
    """
    # Dollar risk
    dollar_risk = account_size * risk_per_trade

    # Risk per contract
    price_risk = abs(entry_price - stop_loss_price)
    contract_risk = price_risk * contract_size

    # Number of contracts
    num_contracts = dollar_risk / contract_risk

    return {
        'account_size': account_size,
        'dollar_risk': dollar_risk,
        'contracts': int(num_contracts),
        'actual_risk': int(num_contracts) * contract_risk,
        'entry_price': entry_price,
        'stop_loss': stop_loss_price,
        'risk_per_contract': contract_risk
    }

# Example: Gold futures
position = commodity_position_sizing(
    account_size=50000,
    risk_per_trade=0.02,  # 2%
    entry_price=1800,
    stop_loss_price=1780,
    contract_size=100,     # 100 oz
    tick_value=10
)

print("Gold Futures Position Sizing:")
print(f"  Account: ${position['account_size']:,}")
print(f"  Risk: ${position['dollar_risk']:,}")
print(f"  Contracts: {position['contracts']}")
print(f"  Entry: ${position['entry_price']}")
print(f"  Stop: ${position['stop_loss']}")
print(f"  Actual Risk: ${position['actual_risk']:,}")
```

## Commodity-Specific Considerations

### Gold Trading

```python
def gold_trading_guide():
    """Gold-specific trading considerations"""
    return {
        'Price Drivers': {
            'Real Interest Rates': 'Negative real rates = bullish gold',
            'USD Strength': 'Weak USD = higher gold',
            'Inflation': 'Inflation hedge',
            'Geopolitical Risk': 'Safe haven demand',
            'Central Bank Buying': 'Demand from central banks',
            'Jewelry Demand': 'India, China seasonal'
        },
        'Trading Vehicles': {
            'Futures': 'GC (100 oz), MGC (10 oz micro)',
            'Physical': 'Coins, bars (premiums over spot)',
            'ETFs': 'GLD, IAU, GLDM',
            'Miners': 'GDX (miners), GDXJ (junior miners)'
        },
        'Key Levels': {
            'Support/Resistance': '$1,700, $1,800, $1,900, $2,000',
            'Psychological': 'Round numbers matter'
        },
        'Correlations': {
            'Inverse USD': 'Typically -0.7 to -0.8',
            'Inverse Real Rates': 'Strong inverse correlation',
            'Silver': 'Positive but volatile'
        }
    }
```

### Crude Oil Trading

```python
def crude_oil_trading_guide():
    """Crude oil specific considerations"""
    return {
        'Price Drivers': {
            'OPEC': 'Production cuts/increases',
            'US Shale': 'Supply response to prices',
            'Demand': 'Economic growth, travel',
            'Inventories': 'EIA weekly reports',
            'Geopolitics': 'Middle East tensions',
            'Refining': 'Crack spreads, refinery maintenance'
        },
        'Benchmarks': {
            'WTI': 'US crude, landlocked (Cushing, OK)',
            'Brent': 'International benchmark, seaborne',
            'Spread': 'Brent typically $2-5 premium to WTI'
        },
        'Trading Vehicles': {
            'Futures': 'CL (1,000 bbls), MCL (100 bbls micro)',
            'ETFs': 'USO (futures-based, contango issues)',
            'Stocks': 'XLE (energy sector), XOP (E&P)'
        },
        'Volatility': 'Very high, $5-10 moves in a day',
        'Key Reports': 'EIA Wednesday 10:30am, Baker Hughes Friday'
    }
```

## Common Mistakes

1. **Ignoring Contango**: Futures ETFs lose money in contango
2. **Over-Leveraging**: Futures require small margins but are risky
3. **Missing Seasonality**: Fighting seasonal patterns
4. **Ignoring Fundamentals**: Supply/demand matter more than in stocks
5. **Poor Timing**: Catching falling knives in bear markets
6. **Neglecting Storage Costs**: Physical commodities have holding costs
7. **Correlation Surprises**: Correlations break down in crises

## Best Practices

1. **Understand the Commodity**: Know what drives it
2. **Watch the Dollar**: USD inverse correlation
3. **Follow the Data**: EIA, USDA, etc.
4. **Respect Trends**: Commodities can trend for years
5. **Use Stops**: Volatility requires risk management
6. **Start Small**: Commodities are volatile
7. **Avoid Physical Unless Expert**: Stick to paper/ETFs
8. **Consider Tax Implications**: 60/40 treatment for futures

## Resources

- **EIA.gov**: Energy data
- **USDA.gov**: Agricultural reports
- **Kitco.com**: Precious metals prices and news
- **CME Group**: Commodity futures specifications
- **Bloomberg/Reuters**: Commodity market data

## Key Takeaways

1. **Commodities provide diversification** and inflation hedge
2. **High volatility** requires careful risk management
3. **Fundamentals matter** - supply/demand drive prices
4. **Multiple ways to invest** - futures, ETFs, stocks, physical
5. **Contango is expensive** - avoid long-term futures ETFs
6. **Seasonality is real** - use to your advantage
7. **Start with ETFs** unless experienced in futures
8. **Gold and oil** are most liquid commodities

## Next Steps

1. Decide commodity allocation (5-15% of portfolio)
2. Choose investment vehicle (ETFs for simplicity)
3. Understand fundamentals of chosen commodities
4. Follow key economic reports
5. Start with small positions
6. Use proper risk management
7. Consider seasonal patterns
8. Rebalance regularly

Remember: Commodities are more volatile than stocks. They're best used as portfolio diversifiers, not core holdings. Most investors should limit commodity exposure to 5-15% of their portfolio.

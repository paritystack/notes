# Risk Management

## Overview

Risk management is the process of identifying, assessing, and controlling threats to an investment portfolio. Effective risk management protects capital while allowing for growth, balancing potential returns against acceptable levels of risk.

## What is Risk?

In finance, risk is the possibility of losing money or not achieving expected returns.

**Types of Risk:**
- **Market Risk**: Overall market movements
- **Credit Risk**: Issuer default
- **Liquidity Risk**: Inability to sell quickly
- **Operational Risk**: System/process failures
- **Concentration Risk**: Over-exposure to single asset
- **Currency Risk**: Foreign exchange fluctuations
- **Interest Rate Risk**: Changes in interest rates
- **Inflation Risk**: Purchasing power erosion

## Risk Measurement

### Volatility (Standard Deviation)

```python
import numpy as np

def calculate_volatility(returns, annualize=True, periods_per_year=252):
    """
    Calculate volatility (standard deviation of returns)

    Args:
        returns: Array of period returns
        annualize: Whether to annualize the result
        periods_per_year: Trading periods per year (252 for daily, 12 for monthly)
    """
    volatility = np.std(returns, ddof=1)

    if annualize:
        volatility *= np.sqrt(periods_per_year)

    return volatility

# Example: Daily returns
daily_returns = np.array([0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.008])
annual_vol = calculate_volatility(daily_returns, annualize=True, periods_per_year=252)
print(f"Annualized Volatility: {annual_vol*100:.2f}%")
```

### Value at Risk (VaR)

```python
def value_at_risk(returns, confidence_level=0.95, portfolio_value=100000):
    """
    Calculate Value at Risk (VaR)
    Maximum loss expected at given confidence level

    Args:
        returns: Array of historical returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        portfolio_value: Current portfolio value
    """
    # Historical VaR (percentile method)
    var_percentile = np.percentile(returns, (1 - confidence_level) * 100)
    var_dollar = portfolio_value * abs(var_percentile)

    return {
        'var_percentage': var_percentile * 100,
        'var_dollar': var_dollar,
        'confidence_level': confidence_level * 100
    }

# Example
returns = np.random.normal(0.0005, 0.02, 1000)  # Simulated returns
var = value_at_risk(returns, 0.95, 100000)

print(f"95% VaR: ${var['var_dollar']:,.2f}")
print(f"Expected maximum loss on 95% of days: ${var['var_dollar']:,.2f}")
```

### Conditional Value at Risk (CVaR)

```python
def conditional_var(returns, confidence_level=0.95, portfolio_value=100000):
    """
    Calculate Conditional VaR (Expected Shortfall)
    Average loss when VaR is exceeded

    More informative than VaR as it accounts for tail risk
    """
    var_percentile = np.percentile(returns, (1 - confidence_level) * 100)

    # Average of returns below VaR threshold
    tail_returns = returns[returns <= var_percentile]
    cvar_percentage = np.mean(tail_returns)
    cvar_dollar = portfolio_value * abs(cvar_percentage)

    return {
        'cvar_percentage': cvar_percentage * 100,
        'cvar_dollar': cvar_dollar,
        'confidence_level': confidence_level * 100
    }

# Example
cvar = conditional_var(returns, 0.95, 100000)
print(f"95% CVaR: ${cvar['cvar_dollar']:,.2f}")
print(f"Average loss when VaR is breached: ${cvar['cvar_dollar']:,.2f}")
```

### Beta

```python
def calculate_beta(asset_returns, market_returns):
    """
    Calculate Beta (systematic risk)
    Beta = Covariance(Asset, Market) / Variance(Market)

    Beta = 1: Moves with market
    Beta > 1: More volatile than market
    Beta < 1: Less volatile than market
    Beta < 0: Moves opposite to market
    """
    covariance = np.cov(asset_returns, market_returns)[0][1]
    market_variance = np.var(market_returns, ddof=1)
    beta = covariance / market_variance

    return beta

# Example
asset_returns = np.array([0.02, -0.01, 0.03, -0.02, 0.04])
market_returns = np.array([0.015, -0.008, 0.02, -0.015, 0.025])

beta = calculate_beta(asset_returns, market_returns)
print(f"Beta: {beta:.2f}")

if beta > 1:
    print(f"Asset is {beta:.2f}x as volatile as the market")
elif beta < 1:
    print(f"Asset is {beta:.2f}x as volatile as the market (less risky)")
```

### Maximum Drawdown

```python
def maximum_drawdown(prices):
    """
    Calculate maximum drawdown
    Largest peak-to-trough decline
    """
    prices = np.array(prices)
    running_max = np.maximum.accumulate(prices)
    drawdowns = (prices - running_max) / running_max

    max_dd = np.min(drawdowns)
    max_dd_idx = np.argmin(drawdowns)

    # Find peak before the trough
    peak_idx = np.argmax(running_max[:max_dd_idx+1] == running_max[max_dd_idx])

    return {
        'max_drawdown_pct': max_dd * 100,
        'peak_value': running_max[max_dd_idx],
        'trough_value': prices[max_dd_idx],
        'peak_date_idx': peak_idx,
        'trough_date_idx': max_dd_idx,
        'drawdown_length': max_dd_idx - peak_idx
    }

# Example
portfolio_values = [100000, 105000, 110000, 108000, 103000, 98000, 101000, 106000]
mdd = maximum_drawdown(portfolio_values)

print(f"Maximum Drawdown: {mdd['max_drawdown_pct']:.2f}%")
print(f"Peak to Trough: ${mdd['peak_value']:,.0f} → ${mdd['trough_value']:,.0f}")
print(f"Drawdown Duration: {mdd['drawdown_length']} periods")
```

## Position Sizing

### Fixed Dollar Amount

```python
def fixed_dollar_sizing(risk_per_trade=1000):
    """
    Simplest method: Risk same dollar amount per trade
    """
    return risk_per_trade

position = fixed_dollar_sizing(1000)
print(f"Risk per trade: ${position}")
```

### Fixed Percentage Risk

```python
def percent_risk_sizing(account_balance, risk_percentage, entry_price, stop_loss):
    """
    Risk fixed percentage of account per trade

    Args:
        account_balance: Total account value
        risk_percentage: Percentage to risk (e.g., 1 for 1%)
        entry_price: Entry price per share
        stop_loss: Stop loss price per share
    """
    dollar_risk = account_balance * (risk_percentage / 100)
    risk_per_share = abs(entry_price - stop_loss)
    shares = int(dollar_risk / risk_per_share)
    position_value = shares * entry_price

    return {
        'shares': shares,
        'position_value': position_value,
        'dollar_risk': dollar_risk,
        'risk_per_share': risk_per_share,
        'position_risk_pct': (dollar_risk / account_balance) * 100
    }

# Example: 2% risk rule
account = 100000
position = percent_risk_sizing(account, 2, entry_price=50, stop_loss=48)

print(f"Account: ${account:,}")
print(f"Position Size: {position['shares']} shares")
print(f"Position Value: ${position['position_value']:,}")
print(f"Risk Amount: ${position['dollar_risk']:,} ({position['position_risk_pct']:.1f}%)")
```

### Kelly Criterion

```python
def kelly_criterion(win_probability, win_loss_ratio):
    """
    Kelly Criterion for optimal position sizing
    Kelly % = (Win Probability × Win/Loss Ratio - (1 - Win Probability)) / Win/Loss Ratio

    Args:
        win_probability: Probability of winning (0 to 1)
        win_loss_ratio: Average win / Average loss

    Returns optimal fraction of capital to risk
    """
    kelly_pct = ((win_probability * win_loss_ratio) - (1 - win_probability)) / win_loss_ratio

    # Many traders use fraction of Kelly (e.g., Half-Kelly)
    full_kelly = max(0, kelly_pct)  # Don't go negative
    half_kelly = full_kelly / 2

    return {
        'full_kelly': full_kelly * 100,
        'half_kelly': half_kelly * 100
    }

# Example: 55% win rate, avg win 2x avg loss
kelly = kelly_criterion(0.55, 2.0)

print(f"Full Kelly: {kelly['full_kelly']:.1f}% of capital")
print(f"Half Kelly (recommended): {kelly['half_kelly']:.1f}% of capital")
```

### Volatility-Based Sizing

```python
def volatility_based_sizing(account_balance, target_volatility, asset_volatility,
                            asset_price, position_type='long'):
    """
    Size position based on volatility
    Equalizes risk across positions with different volatilities

    Args:
        account_balance: Total account value
        target_volatility: Target portfolio volatility
        asset_volatility: Asset's volatility
        asset_price: Current asset price
    """
    # Volatility scaling factor
    scale = target_volatility / asset_volatility

    # Position value
    position_value = account_balance * scale

    # Shares to buy
    shares = int(position_value / asset_price)

    return {
        'shares': shares,
        'position_value': shares * asset_price,
        'position_pct': (shares * asset_price / account_balance) * 100,
        'volatility_scale': scale
    }

# Example
position = volatility_based_sizing(
    account_balance=100000,
    target_volatility=0.15,  # 15% target vol
    asset_volatility=0.25,    # Asset has 25% vol
    asset_price=50
)

print(f"Shares: {position['shares']}")
print(f"Position Value: ${position['position_value']:,}")
print(f"Position %: {position['position_pct']:.1f}%")
```

## Stop Loss Strategies

### Fixed Percentage Stop

```python
def fixed_percentage_stop(entry_price, stop_percentage, position_type='long'):
    """
    Stop loss at fixed percentage from entry

    Args:
        entry_price: Entry price
        stop_percentage: Stop loss percentage (e.g., 5 for 5%)
        position_type: 'long' or 'short'
    """
    if position_type == 'long':
        stop_price = entry_price * (1 - stop_percentage / 100)
        risk_per_share = entry_price - stop_price
    else:  # short
        stop_price = entry_price * (1 + stop_percentage / 100)
        risk_per_share = stop_price - entry_price

    return {
        'stop_price': stop_price,
        'risk_per_share': risk_per_share,
        'stop_percentage': stop_percentage
    }

# Example: 5% stop loss on long position
stop = fixed_percentage_stop(100, 5, 'long')
print(f"Entry: $100")
print(f"Stop Loss: ${stop['stop_price']:.2f}")
print(f"Risk per share: ${stop['risk_per_share']:.2f}")
```

### ATR-Based Stop

```python
def atr_stop_loss(entry_price, atr, atr_multiplier=2, position_type='long'):
    """
    Stop loss based on Average True Range (volatility)
    More dynamic than fixed percentage

    Args:
        entry_price: Entry price
        atr: Average True Range value
        atr_multiplier: Number of ATRs for stop (typically 2-3)
        position_type: 'long' or 'short'
    """
    stop_distance = atr * atr_multiplier

    if position_type == 'long':
        stop_price = entry_price - stop_distance
    else:  # short
        stop_price = entry_price + stop_distance

    return {
        'stop_price': stop_price,
        'stop_distance': stop_distance,
        'atr': atr,
        'atr_multiplier': atr_multiplier
    }

# Example
stop = atr_stop_loss(entry_price=50, atr=2.5, atr_multiplier=2, position_type='long')
print(f"Entry: $50")
print(f"ATR: ${stop['atr']:.2f}")
print(f"Stop Loss: ${stop['stop_price']:.2f} ({stop['atr_multiplier']}x ATR)")
```

### Trailing Stop

```python
class TrailingStop:
    """
    Trailing stop that adjusts with favorable price movement
    """
    def __init__(self, entry_price, trail_percentage, position_type='long'):
        self.entry_price = entry_price
        self.trail_percentage = trail_percentage
        self.position_type = position_type

        if position_type == 'long':
            self.stop_price = entry_price * (1 - trail_percentage / 100)
            self.highest_price = entry_price
        else:
            self.stop_price = entry_price * (1 + trail_percentage / 100)
            self.lowest_price = entry_price

    def update(self, current_price):
        """Update trailing stop based on current price"""
        if self.position_type == 'long':
            # Update highest price seen
            if current_price > self.highest_price:
                self.highest_price = current_price
                # Adjust stop up
                new_stop = current_price * (1 - self.trail_percentage / 100)
                self.stop_price = max(self.stop_price, new_stop)

            # Check if stopped out
            if current_price <= self.stop_price:
                return {'status': 'stopped_out', 'exit_price': self.stop_price}

        else:  # short position
            if current_price < self.lowest_price:
                self.lowest_price = current_price
                new_stop = current_price * (1 + self.trail_percentage / 100)
                self.stop_price = min(self.stop_price, new_stop)

            if current_price >= self.stop_price:
                return {'status': 'stopped_out', 'exit_price': self.stop_price}

        return {
            'status': 'active',
            'stop_price': self.stop_price,
            'highest': self.highest_price if self.position_type == 'long' else self.lowest_price
        }

# Example
trailing = TrailingStop(entry_price=100, trail_percentage=5, position_type='long')
print(f"Initial Stop: ${trailing.stop_price:.2f}")

# Simulate price movements
prices = [102, 105, 103, 107, 104]
for price in prices:
    result = trailing.update(price)
    print(f"Price: ${price}, Stop: ${result['stop_price']:.2f}, Status: {result['status']}")
```

## Diversification

### Correlation Analysis

```python
def correlation_diversification(returns_matrix):
    """
    Analyze correlation between assets
    Lower correlation = better diversification

    Args:
        returns_matrix: DataFrame with returns for each asset
    """
    correlation_matrix = np.corrcoef(returns_matrix.T)

    # Average correlation
    n = correlation_matrix.shape[0]
    sum_corr = np.sum(correlation_matrix) - n  # Exclude diagonal
    avg_correlation = sum_corr / (n * (n - 1))

    return {
        'correlation_matrix': correlation_matrix,
        'average_correlation': avg_correlation
    }

# Example
# Simulated returns for 3 assets
asset1 = np.random.normal(0.001, 0.02, 100)
asset2 = np.random.normal(0.0008, 0.015, 100)
asset3 = asset1 * 0.7 + np.random.normal(0, 0.01, 100)  # Correlated with asset1

returns_matrix = np.vstack([asset1, asset2, asset3])
corr = correlation_diversification(returns_matrix)

print(f"Average Correlation: {corr['average_correlation']:.3f}")
print("\nCorrelation Matrix:")
print(corr['correlation_matrix'])
```

### Sector Diversification

```python
def sector_allocation_limits(portfolio_value, max_sector_pct=25):
    """
    Limit concentration in any single sector

    Args:
        portfolio_value: Total portfolio value
        max_sector_pct: Maximum percentage in any sector
    """
    max_sector_value = portfolio_value * (max_sector_pct / 100)

    sectors = [
        'Technology', 'Healthcare', 'Financial', 'Consumer',
        'Industrial', 'Energy', 'Utilities', 'Real Estate',
        'Materials', 'Communication'
    ]

    return {
        'max_per_sector': max_sector_value,
        'max_percentage': max_sector_pct,
        'num_sectors': len(sectors),
        'sectors': sectors
    }

limits = sector_allocation_limits(100000, 25)
print(f"Maximum per sector: ${limits['max_per_sector']:,} ({limits['max_percentage']}%)")
print(f"Recommended sectors: {limits['num_sectors']}")
```

### Geographic Diversification

```python
def geographic_allocation(portfolio_value, risk_tolerance='moderate'):
    """
    Diversify across geographic regions

    Args:
        portfolio_value: Total portfolio value
        risk_tolerance: 'conservative', 'moderate', 'aggressive'
    """
    allocations = {
        'conservative': {
            'US': 70,
            'Developed_International': 25,
            'Emerging_Markets': 5
        },
        'moderate': {
            'US': 60,
            'Developed_International': 30,
            'Emerging_Markets': 10
        },
        'aggressive': {
            'US': 50,
            'Developed_International': 30,
            'Emerging_Markets': 20
        }
    }

    allocation = allocations[risk_tolerance]
    dollar_allocation = {
        region: portfolio_value * (pct / 100)
        for region, pct in allocation.items()
    }

    return {
        'percentages': allocation,
        'dollar_amounts': dollar_allocation
    }

geo_alloc = geographic_allocation(100000, 'moderate')
print("Geographic Allocation (Moderate):")
for region, amount in geo_alloc['dollar_amounts'].items():
    pct = geo_alloc['percentages'][region]
    print(f"  {region}: ${amount:,.0f} ({pct}%)")
```

## Hedging Strategies

### Portfolio Hedging with Options

```python
def protective_put_hedge(portfolio_value, current_price, put_strike,
                        put_premium, shares_to_hedge):
    """
    Protective Put: Buy puts to hedge downside risk

    Args:
        portfolio_value: Total portfolio value
        current_price: Current stock price
        put_strike: Put option strike price
        put_premium: Cost per put option
        shares_to_hedge: Number of shares to protect
    """
    # Number of put contracts needed (1 contract = 100 shares)
    contracts = shares_to_hedge / 100
    total_cost = contracts * 100 * put_premium

    # Maximum loss calculation
    stock_position_value = shares_to_hedge * current_price
    max_loss_per_share = current_price - put_strike
    gross_loss = max_loss_per_share * shares_to_hedge
    net_loss = gross_loss + total_cost  # Include put cost

    return {
        'put_contracts': contracts,
        'hedge_cost': total_cost,
        'hedge_cost_pct': (total_cost / portfolio_value) * 100,
        'max_loss': net_loss,
        'protected_at': put_strike,
        'breakeven': current_price + put_premium
    }

# Example: Hedge 1000 shares
hedge = protective_put_hedge(
    portfolio_value=100000,
    current_price=50,
    put_strike=48,
    put_premium=1.50,
    shares_to_hedge=1000
)

print(f"Protective Put Hedge:")
print(f"  Put Contracts: {hedge['put_contracts']:.0f}")
print(f"  Hedge Cost: ${hedge['hedge_cost']:,.2f} ({hedge['hedge_cost_pct']:.2f}%)")
print(f"  Protected Below: ${hedge['protected_at']}")
print(f"  Max Loss: ${hedge['max_loss']:,.2f}")
```

### Collar Strategy

```python
def collar_strategy(shares, current_price, put_strike, put_premium,
                   call_strike, call_premium):
    """
    Collar: Buy protective put, sell covered call
    Reduces hedge cost but caps upside

    Args:
        shares: Number of shares to hedge
        current_price: Current stock price
        put_strike: Put strike (downside protection)
        put_premium: Put cost per share
        call_strike: Call strike (upside cap)
        call_premium: Call premium received per share
    """
    position_value = shares * current_price

    # Net cost (put cost - call premium)
    net_cost = (put_premium - call_premium) * shares

    # Profit/Loss scenarios
    max_loss = (current_price - put_strike) * shares + net_cost
    max_gain = (call_strike - current_price) * shares - net_cost

    return {
        'position_value': position_value,
        'net_cost': net_cost,
        'downside_protected_at': put_strike,
        'upside_capped_at': call_strike,
        'max_loss': max_loss,
        'max_gain': max_gain,
        'cost_pct': (net_cost / position_value) * 100
    }

# Example: Zero-cost collar
collar = collar_strategy(
    shares=1000,
    current_price=50,
    put_strike=47,
    put_premium=1.50,
    call_strike=53,
    call_premium=1.50
)

print(f"Collar Strategy:")
print(f"  Net Cost: ${collar['net_cost']:,.2f} ({collar['cost_pct']:.2f}%)")
print(f"  Protected below: ${collar['downside_protected_at']}")
print(f"  Capped above: ${collar['upside_capped_at']}")
print(f"  Max Loss: ${collar['max_loss']:,.2f}")
print(f"  Max Gain: ${collar['max_gain']:,.2f}")
```

### Beta Hedging

```python
def beta_hedge(portfolio_value, portfolio_beta, hedge_instrument_beta=1.0):
    """
    Hedge using index futures/ETFs
    Reduces market risk (beta) while maintaining individual stock exposure

    Args:
        portfolio_value: Portfolio value to hedge
        portfolio_beta: Portfolio's beta relative to market
        hedge_instrument_beta: Beta of hedging instrument (usually 1.0 for index)
    """
    # Amount to hedge to achieve beta-neutral
    hedge_ratio = portfolio_beta / hedge_instrument_beta
    hedge_notional = portfolio_value * hedge_ratio

    return {
        'portfolio_beta': portfolio_beta,
        'hedge_ratio': hedge_ratio,
        'hedge_notional': hedge_notional,
        'resulting_beta': portfolio_beta - (hedge_ratio * hedge_instrument_beta)
    }

# Example: Hedge portfolio with beta of 1.3
hedge = beta_hedge(portfolio_value=100000, portfolio_beta=1.3)

print(f"Portfolio Beta: {hedge['portfolio_beta']}")
print(f"Hedge Ratio: {hedge['hedge_ratio']:.2f}")
print(f"Short Index: ${hedge['hedge_notional']:,.0f}")
print(f"Resulting Beta: {hedge['resulting_beta']:.2f}")
```

## Risk Budgeting

### Risk Parity

```python
def risk_parity_weights(asset_volatilities):
    """
    Risk Parity: Allocate capital so each asset contributes equal risk

    Args:
        asset_volatilities: Array of asset volatilities
    """
    # Inverse volatility weighting
    inv_vols = 1 / np.array(asset_volatilities)
    weights = inv_vols / np.sum(inv_vols)

    # Risk contribution
    risk_contributions = weights * np.array(asset_volatilities)
    risk_contributions = risk_contributions / np.sum(risk_contributions)

    return {
        'weights': weights * 100,
        'risk_contributions': risk_contributions * 100
    }

# Example
volatilities = [0.20, 0.10, 0.15, 0.25]  # Stocks, Bonds, REITs, Commodities
rp = risk_parity_weights(volatilities)

assets = ['Stocks', 'Bonds', 'REITs', 'Commodities']
print("Risk Parity Allocation:")
for asset, weight, risk in zip(assets, rp['weights'], rp['risk_contributions']):
    print(f"  {asset}: {weight:.1f}% (risk contribution: {risk:.1f}%)")
```

### Risk Budget Allocation

```python
def allocate_risk_budget(total_budget, asset_risk_budgets, asset_volatilities):
    """
    Allocate based on risk budget for each asset

    Args:
        total_budget: Total risk budget (e.g., 10% portfolio volatility)
        asset_risk_budgets: Dict of risk allocations (must sum to 1)
        asset_volatilities: Dict of asset volatilities
    """
    weights = {}

    for asset, risk_allocation in asset_risk_budgets.items():
        # Weight = Risk Budget / Asset Volatility
        asset_risk = total_budget * risk_allocation
        weight = asset_risk / asset_volatilities[asset]
        weights[asset] = weight

    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    normalized_weights = {k: v/total_weight for k, v in weights.items()}

    return normalized_weights

# Example
risk_budgets = {'stocks': 0.50, 'bonds': 0.30, 'alternatives': 0.20}
volatilities = {'stocks': 0.18, 'bonds': 0.06, 'alternatives': 0.25}

weights = allocate_risk_budget(0.10, risk_budgets, volatilities)
print("Risk Budget Allocation:")
for asset, weight in weights.items():
    print(f"  {asset}: {weight*100:.1f}%")
```

## Scenario Analysis

### Stress Testing

```python
def stress_test_portfolio(portfolio_positions, stress_scenarios):
    """
    Test portfolio under extreme scenarios

    Args:
        portfolio_positions: Dict of {asset: {'value': X, 'beta': Y}}
        stress_scenarios: Dict of market scenarios
    """
    results = {}

    for scenario_name, market_move in stress_scenarios.items():
        scenario_loss = 0

        for asset, position in portfolio_positions.items():
            # Approximate loss using beta
            expected_move = market_move * position['beta']
            asset_loss = position['value'] * expected_move
            scenario_loss += asset_loss

        results[scenario_name] = {
            'market_move': market_move * 100,
            'portfolio_impact': scenario_loss,
            'portfolio_impact_pct': (scenario_loss / sum(p['value'] for p in portfolio_positions.values())) * 100
        }

    return results

# Example
portfolio = {
    'tech_stocks': {'value': 40000, 'beta': 1.4},
    'value_stocks': {'value': 30000, 'beta': 0.9},
    'bonds': {'value': 25000, 'beta': 0.2},
    'gold': {'value': 5000, 'beta': -0.1}
}

scenarios = {
    '2008_Crisis': -0.40,      # -40% market drop
    'Flash_Crash': -0.10,      # -10% sudden drop
    'Mild_Correction': -0.05,  # -5% correction
    'Bear_Market': -0.20       # -20% bear market
}

stress_results = stress_test_portfolio(portfolio, scenarios)

print("Stress Test Results:")
for scenario, result in stress_results.items():
    print(f"\n{scenario} ({result['market_move']:.1f}% market):")
    print(f"  Portfolio Impact: ${result['portfolio_impact']:,.0f}")
    print(f"  Portfolio Impact: {result['portfolio_impact_pct']:.2f}%")
```

### Monte Carlo Simulation

```python
def monte_carlo_risk_simulation(initial_value, expected_return, volatility,
                                years, simulations=1000):
    """
    Monte Carlo simulation for portfolio value

    Args:
        initial_value: Starting portfolio value
        expected_return: Expected annual return
        volatility: Annual volatility
        years: Investment horizon
        simulations: Number of simulations to run
    """
    results = []

    for _ in range(simulations):
        value = initial_value
        annual_returns = []

        for year in range(years):
            # Generate random return
            annual_return = np.random.normal(expected_return, volatility)
            value *= (1 + annual_return)
            annual_returns.append(annual_return)

        results.append(value)

    results = np.array(results)

    return {
        'mean': np.mean(results),
        'median': np.median(results),
        'percentile_5': np.percentile(results, 5),
        'percentile_25': np.percentile(results, 25),
        'percentile_75': np.percentile(results, 75),
        'percentile_95': np.percentile(results, 95),
        'probability_of_loss': np.sum(results < initial_value) / simulations * 100
    }

# Example: $100k invested for 10 years
simulation = monte_carlo_risk_simulation(
    initial_value=100000,
    expected_return=0.08,
    volatility=0.15,
    years=10,
    simulations=10000
)

print("10-Year Monte Carlo Simulation:")
print(f"  Mean outcome: ${simulation['mean']:,.0f}")
print(f"  Median outcome: ${simulation['median']:,.0f}")
print(f"  5th percentile: ${simulation['percentile_5']:,.0f}")
print(f"  95th percentile: ${simulation['percentile_95']:,.0f}")
print(f"  Probability of loss: {simulation['probability_of_loss']:.1f}%")
```

## Best Practices

### Risk Management Rules

1. **Never risk more than 1-2% per trade**
2. **Always use stop losses**
3. **Diversify across assets, sectors, and strategies**
4. **Size positions based on volatility**
5. **Have maximum portfolio drawdown limit**
6. **Monitor correlation between positions**
7. **Hedge tail risk in large portfolios**
8. **Regular stress testing and scenario analysis**

### Risk Management Checklist

```python
def risk_management_checklist():
    """Daily/weekly risk management checklist"""
    return {
        'Daily': [
            'Review all open positions',
            'Check stop losses are in place',
            'Monitor position sizes',
            'Review risk per position',
            'Check total portfolio risk'
        ],
        'Weekly': [
            'Calculate portfolio VaR',
            'Review correlation matrix',
            'Assess sector concentration',
            'Check maximum drawdown',
            'Review risk-adjusted returns',
            'Rebalance if needed'
        ],
        'Monthly': [
            'Full portfolio risk analysis',
            'Stress test scenarios',
            'Review risk limits and rules',
            'Analyze risk attribution',
            'Adjust hedges if needed'
        ],
        'Quarterly': [
            'Comprehensive risk review',
            'Update risk models',
            'Review and update IPS',
            'Assess risk-return profile',
            'Strategic allocation review'
        ]
    }

checklist = risk_management_checklist()
for timeframe, items in checklist.items():
    print(f"\n{timeframe}:")
    for item in items:
        print(f"  □ {item}")
```

## Common Mistakes

1. **Over-leveraging**: Using too much margin/leverage
2. **No stop losses**: Letting losses run
3. **Over-concentration**: Too much in one position
4. **Ignoring correlation**: Thinking you're diversified when positions are correlated
5. **Position sizing errors**: Risking too much on single trades
6. **Revenge trading**: Taking excess risk to recover losses
7. **Ignoring tail risk**: Not preparing for extreme events
8. **Static hedging**: Not adjusting hedges as conditions change

## Key Takeaways

1. **Risk management is more important than return generation**
2. **Preservation of capital is the first priority**
3. **Position sizing is critical** - right size prevents catastrophic losses
4. **Diversification works** - but understand correlation
5. **Stop losses protect capital** - use them religiously
6. **Measure risk properly** - volatility, VaR, drawdown, beta
7. **Stress test regularly** - know your worst-case scenarios
8. **Have a risk management plan** - and follow it

## Resources

- "Risk Management in Trading" by Davis Edwards
- "The Intelligent Investor" by Benjamin Graham
- "Against the Gods" by Peter Bernstein
- "When Genius Failed" (LTCM case study)

## Next Steps

1. Calculate your risk tolerance
2. Determine position sizing rules
3. Implement stop loss discipline
4. Diversify properly across uncorrelated assets
5. Set up risk monitoring systems
6. Stress test your portfolio
7. Create and follow risk management plan
8. Review and adjust regularly

Remember: In investing, it's not about how much you make, but how much you don't lose. Effective risk management allows you to stay in the game long enough to benefit from compounding returns.

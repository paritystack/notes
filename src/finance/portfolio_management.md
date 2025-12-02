# Portfolio Management

## Overview

Portfolio management is the art and science of making decisions about investment mix and policy to match investments to objectives, asset allocation, and balancing risk against performance. This guide covers modern portfolio theory, asset allocation strategies, and portfolio optimization techniques.

## What is Portfolio Management?

Portfolio management involves:
- **Asset Allocation**: Distributing investments across asset classes
- **Security Selection**: Choosing specific investments
- **Risk Management**: Controlling portfolio risk
- **Performance Monitoring**: Tracking and evaluating results
- **Rebalancing**: Maintaining target allocations

**Goals:**
- Maximize returns for given risk level
- Minimize risk for desired return level
- Achieve specific investment objectives

## Modern Portfolio Theory (MPT)

### Core Concepts

Developed by Harry Markowitz (1952), MPT demonstrates that diversification can reduce portfolio risk.

**Key Principles:**
1. **Expected Return**: Weighted average of asset returns
2. **Portfolio Risk**: Not just average of individual risks
3. **Correlation**: How assets move together
4. **Efficient Frontier**: Optimal risk-return combinations
5. **Diversification**: Reduces unsystematic risk

### Portfolio Return

```python
import numpy as np

def portfolio_return(weights, expected_returns):
    """
    Calculate expected portfolio return

    Args:
        weights: Array of asset weights (must sum to 1)
        expected_returns: Array of expected returns for each asset
    """
    return np.dot(weights, expected_returns)

# Example: 3-asset portfolio
weights = np.array([0.40, 0.35, 0.25])  # 40% stocks, 35% bonds, 25% REITs
returns = np.array([0.10, 0.04, 0.08])  # Expected returns: 10%, 4%, 8%

port_return = portfolio_return(weights, returns)
print(f"Expected Portfolio Return: {port_return*100:.2f}%")
```

### Portfolio Volatility

```python
def portfolio_volatility(weights, cov_matrix):
    """
    Calculate portfolio standard deviation (volatility)

    Args:
        weights: Array of asset weights
        cov_matrix: Covariance matrix of asset returns
    """
    variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return np.sqrt(variance)

# Example: Covariance matrix for 3 assets
cov_matrix = np.array([
    [0.04, 0.006, 0.02],   # Stock variance and covariances
    [0.006, 0.01, 0.005],  # Bond variance and covariances
    [0.02, 0.005, 0.03]    # REIT variance and covariances
])

weights = np.array([0.40, 0.35, 0.25])
port_vol = portfolio_volatility(weights, cov_matrix)
print(f"Portfolio Volatility: {port_vol*100:.2f}%")
```

### Correlation and Diversification

```python
def correlation_matrix(returns_data):
    """
    Calculate correlation matrix from returns

    Args:
        returns_data: DataFrame with returns for each asset
    """
    return returns_data.corr()

def diversification_ratio(weights, individual_volatilities, portfolio_volatility):
    """
    Measure of diversification benefit
    Higher ratio = better diversification
    """
    weighted_vol = np.dot(weights, individual_volatilities)
    return weighted_vol / portfolio_volatility

# Example
individual_vols = np.array([0.20, 0.10, 0.17])  # Individual volatilities
weights = np.array([0.40, 0.35, 0.25])
portfolio_vol = 0.12

div_ratio = diversification_ratio(weights, individual_vols, portfolio_vol)
print(f"Diversification Ratio: {div_ratio:.2f}")
```

## Asset Allocation

### Strategic Asset Allocation

Long-term target allocation based on goals and risk tolerance.

```python
def strategic_allocation(age, risk_tolerance='moderate'):
    """
    Rule-of-thumb asset allocation

    Common rules:
    - 100 - age = stock allocation
    - 120 - age = stock allocation (more aggressive)
    - Adjust based on risk tolerance
    """
    base_stock = 100 - age

    if risk_tolerance == 'aggressive':
        stock_pct = min(base_stock + 20, 90)
    elif risk_tolerance == 'moderate':
        stock_pct = base_stock
    else:  # conservative
        stock_pct = max(base_stock - 20, 20)

    bond_pct = 100 - stock_pct

    return {
        'stocks': stock_pct,
        'bonds': bond_pct
    }

# Example: 35-year-old, moderate risk
allocation = strategic_allocation(35, 'moderate')
print(f"Strategic Allocation:")
print(f"  Stocks: {allocation['stocks']}%")
print(f"  Bonds: {allocation['bonds']}%")
```

### Tactical Asset Allocation

Short-term adjustments based on market conditions.

```python
def tactical_adjustment(base_allocation, market_view):
    """
    Adjust allocation based on market outlook

    Args:
        base_allocation: Dict with base percentages
        market_view: Dict with adjustments
    """
    adjusted = base_allocation.copy()

    for asset, adjustment in market_view.items():
        if asset in adjusted:
            adjusted[asset] += adjustment

    # Normalize to 100%
    total = sum(adjusted.values())
    adjusted = {k: (v/total)*100 for k, v in adjusted.items()}

    return adjusted

# Example: Bullish on stocks
base = {'stocks': 60, 'bonds': 30, 'cash': 10}
view = {'stocks': 5, 'bonds': -3, 'cash': -2}  # Increase stocks 5%

tactical = tactical_adjustment(base, view)
print("Tactical Allocation:")
for asset, pct in tactical.items():
    print(f"  {asset}: {pct:.1f}%")
```

### Dynamic Asset Allocation

Continuously adjusts based on changing conditions.

```python
def risk_parity_allocation(volatilities, target_risk=0.10):
    """
    Risk Parity: Equal risk contribution from each asset

    Args:
        volatilities: Array of asset volatilities
        target_risk: Target portfolio risk
    """
    # Inverse volatility weighting
    inv_vols = 1 / np.array(volatilities)
    weights = inv_vols / np.sum(inv_vols)

    return weights

# Example
vols = np.array([0.15, 0.08, 0.12])  # Stocks, Bonds, Commodities
weights = risk_parity_allocation(vols)

print("Risk Parity Allocation:")
assets = ['Stocks', 'Bonds', 'Commodities']
for asset, weight in zip(assets, weights):
    print(f"  {asset}: {weight*100:.1f}%")
```

## Portfolio Optimization

### Mean-Variance Optimization

Find portfolio with best return for given risk.

```python
from scipy.optimize import minimize

def optimize_portfolio(expected_returns, cov_matrix, target_return=None):
    """
    Find optimal portfolio weights

    Args:
        expected_returns: Array of expected returns
        cov_matrix: Covariance matrix
        target_return: Target return (if None, finds minimum variance)
    """
    n_assets = len(expected_returns)

    def portfolio_stats(weights):
        returns = np.dot(weights, expected_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return returns, vol

    def min_volatility(weights):
        return portfolio_stats(weights)[1]

    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # weights sum to 1

    if target_return is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda x: portfolio_stats(x)[0] - target_return
        })

    # Bounds: 0 <= weight <= 1
    bounds = tuple((0, 1) for _ in range(n_assets))

    # Initial guess: equal weights
    init_guess = n_assets * [1. / n_assets]

    # Optimize
    result = minimize(min_volatility, init_guess,
                     method='SLSQP', bounds=bounds,
                     constraints=constraints)

    return result.x

# Example
returns = np.array([0.10, 0.06, 0.08])
cov_matrix = np.array([
    [0.04, 0.006, 0.02],
    [0.006, 0.01, 0.005],
    [0.02, 0.005, 0.03]
])

# Minimum variance portfolio
optimal_weights = optimize_portfolio(returns, cov_matrix)
print("Optimal Weights (Minimum Variance):")
for i, weight in enumerate(optimal_weights):
    print(f"  Asset {i+1}: {weight*100:.1f}%")
```

### Efficient Frontier

```python
def efficient_frontier(expected_returns, cov_matrix, n_portfolios=100):
    """
    Generate efficient frontier portfolios

    Args:
        expected_returns: Array of expected returns
        cov_matrix: Covariance matrix
        n_portfolios: Number of portfolios to generate
    """
    min_return = np.min(expected_returns)
    max_return = np.max(expected_returns)

    target_returns = np.linspace(min_return, max_return, n_portfolios)
    efficient_portfolios = []

    for target in target_returns:
        try:
            weights = optimize_portfolio(expected_returns, cov_matrix, target)
            ret = np.dot(weights, expected_returns)
            vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            efficient_portfolios.append({
                'return': ret,
                'volatility': vol,
                'weights': weights
            })
        except:
            continue

    return efficient_portfolios

# Generate frontier
# frontier = efficient_frontier(returns, cov_matrix, 50)
```

### Maximum Sharpe Ratio Portfolio

```python
def max_sharpe_portfolio(expected_returns, cov_matrix, risk_free_rate=0.02):
    """
    Find portfolio with maximum Sharpe ratio

    Sharpe Ratio = (Return - Risk-Free Rate) / Volatility
    """
    n_assets = len(expected_returns)

    def neg_sharpe(weights):
        ret = np.dot(weights, expected_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (ret - risk_free_rate) / vol
        return -sharpe  # Negative because we minimize

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple((0, 1) for _ in range(n_assets))
    init_guess = n_assets * [1. / n_assets]

    result = minimize(neg_sharpe, init_guess,
                     method='SLSQP', bounds=bounds,
                     constraints=constraints)

    return result.x

# Example
sharpe_weights = max_sharpe_portfolio(returns, cov_matrix, 0.02)
print("\nMaximum Sharpe Ratio Portfolio:")
for i, weight in enumerate(sharpe_weights):
    print(f"  Asset {i+1}: {weight*100:.1f}%")
```

## Rebalancing

### When to Rebalance

```python
def check_rebalancing_needed(current_weights, target_weights, threshold=0.05):
    """
    Check if rebalancing is needed

    Args:
        current_weights: Current portfolio weights
        target_weights: Target weights
        threshold: Tolerance threshold (e.g., 0.05 = 5%)
    """
    drift = np.abs(current_weights - target_weights)
    needs_rebalancing = np.any(drift > threshold)

    rebalance_info = []
    for i, (curr, targ, d) in enumerate(zip(current_weights, target_weights, drift)):
        rebalance_info.append({
            'asset': i+1,
            'current': curr * 100,
            'target': targ * 100,
            'drift': d * 100,
            'action': 'Rebalance' if d > threshold else 'OK'
        })

    return needs_rebalancing, rebalance_info

# Example
current = np.array([0.70, 0.25, 0.05])  # Drifted allocation
target = np.array([0.60, 0.30, 0.10])   # Target allocation

needs_rebal, info = check_rebalancing_needed(current, target, 0.05)
print(f"Rebalancing Needed: {needs_rebal}\n")
for item in info:
    print(f"Asset {item['asset']}: {item['current']:.1f}% -> {item['target']:.1f}% "
          f"(drift: {item['drift']:.1f}%) - {item['action']}")
```

### Rebalancing Methods

```python
def calculate_rebalancing_trades(current_value, current_weights,
                                target_weights, trade_cost=0.001):
    """
    Calculate trades needed to rebalance

    Args:
        current_value: Total portfolio value
        current_weights: Current weights
        target_weights: Target weights
        trade_cost: Transaction cost as percentage
    """
    current_dollars = current_value * current_weights
    target_dollars = current_value * target_weights

    trades = target_dollars - current_dollars
    trade_costs = np.abs(trades) * trade_cost

    trades_info = []
    for i, (trade, cost) in enumerate(zip(trades, trade_costs)):
        if abs(trade) > 1:  # Only if trade > $1
            trades_info.append({
                'asset': i+1,
                'action': 'Buy' if trade > 0 else 'Sell',
                'amount': abs(trade),
                'cost': cost
            })

    total_cost = np.sum(trade_costs)

    return trades_info, total_cost

# Example
portfolio_value = 100000
current = np.array([0.70, 0.25, 0.05])
target = np.array([0.60, 0.30, 0.10])

trades, total_cost = calculate_rebalancing_trades(
    portfolio_value, current, target, 0.001
)

print("Rebalancing Trades:")
for trade in trades:
    print(f"  Asset {trade['asset']}: {trade['action']} "
          f"${trade['amount']:.2f} (cost: ${trade['cost']:.2f})")
print(f"\nTotal Transaction Cost: ${total_cost:.2f}")
```

### Rebalancing Strategies

```python
class RebalancingStrategy:
    """Different rebalancing approaches"""

    @staticmethod
    def calendar_rebalancing(months=12):
        """Rebalance at fixed intervals"""
        return f"Rebalance every {months} months"

    @staticmethod
    def threshold_rebalancing(threshold=5):
        """Rebalance when drift exceeds threshold"""
        return f"Rebalance when any asset drifts >{threshold}%"

    @staticmethod
    def tolerance_band(target_weight, band_width=5):
        """Create tolerance bands around targets"""
        lower = max(0, target_weight - band_width)
        upper = min(100, target_weight + band_width)
        return {
            'target': target_weight,
            'lower_band': lower,
            'upper_band': upper
        }

# Example: 60/40 portfolio with 5% bands
stock_band = RebalancingStrategy.tolerance_band(60, 5)
bond_band = RebalancingStrategy.tolerance_band(40, 5)

print("Tolerance Bands:")
print(f"  Stocks: {stock_band['lower_band']}% - {stock_band['upper_band']}%")
print(f"  Bonds: {bond_band['lower_band']}% - {bond_band['upper_band']}%")
```

## Performance Measurement

### Return Metrics

```python
def time_weighted_return(values):
    """
    Time-weighted return (TWR)
    Removes effect of cash flows
    """
    returns = []
    for i in range(1, len(values)):
        ret = (values[i] - values[i-1]) / values[i-1]
        returns.append(ret)

    # Geometric mean
    twr = 1
    for r in returns:
        twr *= (1 + r)
    twr = (twr - 1) * 100

    return twr

def money_weighted_return(cash_flows, values, periods):
    """
    Money-weighted return (MWR) / Internal Rate of Return (IRR)
    Accounts for timing and size of cash flows
    """
    from numpy import irr  # Note: numpy.irr is deprecated, use numpy_financial
    # This is a simplified example
    return "Use numpy_financial.irr(cash_flows)"

# Example: Portfolio values over time
portfolio_values = [100000, 105000, 103000, 110000, 115000]
twr = time_weighted_return(portfolio_values)
print(f"Time-Weighted Return: {twr:.2f}%")
```

### Risk-Adjusted Returns

```python
def sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Sharpe Ratio = (Return - Risk-Free Rate) / Standard Deviation
    Higher is better
    """
    excess_return = np.mean(returns) - risk_free_rate
    std_dev = np.std(returns, ddof=1)
    return excess_return / std_dev

def sortino_ratio(returns, risk_free_rate=0.02):
    """
    Sortino Ratio = (Return - Risk-Free Rate) / Downside Deviation
    Only penalizes downside volatility
    """
    excess_return = np.mean(returns) - risk_free_rate
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 0 else 0
    return excess_return / downside_std if downside_std > 0 else 0

def information_ratio(portfolio_returns, benchmark_returns):
    """
    Information Ratio = (Portfolio Return - Benchmark Return) / Tracking Error
    Measures active return per unit of active risk
    """
    active_return = np.mean(portfolio_returns - benchmark_returns)
    tracking_error = np.std(portfolio_returns - benchmark_returns, ddof=1)
    return active_return / tracking_error if tracking_error > 0 else 0

# Example
returns = np.array([0.08, 0.12, -0.05, 0.15, 0.10])
benchmark = np.array([0.07, 0.09, -0.03, 0.12, 0.08])

sharpe = sharpe_ratio(returns, 0.02)
sortino = sortino_ratio(returns, 0.02)
info_ratio = information_ratio(returns, benchmark)

print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Sortino Ratio: {sortino:.2f}")
print(f"Information Ratio: {info_ratio:.2f}")
```

### Drawdown Analysis

```python
def calculate_drawdowns(portfolio_values):
    """
    Calculate drawdown series

    Drawdown = (Current Value - Peak Value) / Peak Value
    """
    portfolio_values = np.array(portfolio_values)
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - running_max) / running_max

    max_drawdown = np.min(drawdowns)
    max_dd_index = np.argmin(drawdowns)

    return {
        'drawdowns': drawdowns,
        'max_drawdown': max_drawdown * 100,
        'max_dd_index': max_dd_index,
        'peak_value': running_max[max_dd_index],
        'trough_value': portfolio_values[max_dd_index]
    }

# Example
values = [100000, 105000, 110000, 102000, 98000, 103000, 108000]
dd_stats = calculate_drawdowns(values)

print(f"Maximum Drawdown: {dd_stats['max_drawdown']:.2f}%")
print(f"Peak Value: ${dd_stats['peak_value']:,.0f}")
print(f"Trough Value: ${dd_stats['trough_value']:,.0f}")
```

## Portfolio Construction Strategies

### Core-Satellite Strategy

```python
def core_satellite_portfolio(total_value, core_pct=70):
    """
    Core-Satellite Approach

    Core: Passive index funds (70-80%)
    Satellite: Active strategies (20-30%)
    """
    core_value = total_value * (core_pct / 100)
    satellite_value = total_value * ((100 - core_pct) / 100)

    portfolio = {
        'core': {
            'value': core_value,
            'allocation': {
                'total_market_index': 60,
                'bond_index': 30,
                'international_index': 10
            }
        },
        'satellite': {
            'value': satellite_value,
            'allocation': {
                'sector_funds': 40,
                'individual_stocks': 30,
                'alternatives': 30
            }
        }
    }

    return portfolio

# Example
portfolio = core_satellite_portfolio(100000, 70)
print(f"Core: ${portfolio['core']['value']:,.0f}")
print(f"Satellite: ${portfolio['satellite']['value']:,.0f}")
```

### Dollar-Cost Averaging

```python
def dollar_cost_averaging(monthly_investment, prices, months):
    """
    Simulate dollar-cost averaging strategy

    Args:
        monthly_investment: Fixed dollar amount per month
        prices: Array of monthly prices
        months: Number of months to invest
    """
    shares_purchased = []
    total_shares = 0
    total_invested = 0

    for i in range(min(months, len(prices))):
        shares = monthly_investment / prices[i]
        shares_purchased.append(shares)
        total_shares += shares
        total_invested += monthly_investment

    avg_price = total_invested / total_shares
    final_value = total_shares * prices[months-1]
    total_return = ((final_value - total_invested) / total_invested) * 100

    return {
        'total_shares': total_shares,
        'total_invested': total_invested,
        'average_price': avg_price,
        'final_value': final_value,
        'return': total_return
    }

# Example: $1000/month for 12 months
prices = [50, 48, 52, 45, 47, 49, 51, 53, 50, 48, 52, 54]
result = dollar_cost_averaging(1000, prices, 12)

print(f"Total Invested: ${result['total_invested']:,.0f}")
print(f"Total Shares: {result['total_shares']:.2f}")
print(f"Average Price: ${result['average_price']:.2f}")
print(f"Final Value: ${result['final_value']:,.0f}")
print(f"Return: {result['return']:.2f}%")
```

## Tax-Efficient Portfolio Management

### Tax Loss Harvesting

```python
def tax_loss_harvesting(positions, current_prices, tax_rate=0.24):
    """
    Identify tax loss harvesting opportunities

    Args:
        positions: List of dicts with 'cost_basis', 'shares'
        current_prices: Current prices for each position
        tax_rate: Capital gains tax rate
    """
    opportunities = []

    for i, (pos, price) in enumerate(zip(positions, current_prices)):
        current_value = pos['shares'] * price
        cost_basis = pos['cost_basis'] * pos['shares']
        gain_loss = current_value - cost_basis

        if gain_loss < 0:  # Loss
            tax_benefit = abs(gain_loss) * tax_rate
            opportunities.append({
                'position': i+1,
                'loss': gain_loss,
                'tax_benefit': tax_benefit,
                'cost_basis': cost_basis,
                'current_value': current_value
            })

    return opportunities

# Example
positions = [
    {'cost_basis': 50, 'shares': 100},
    {'cost_basis': 75, 'shares': 50},
    {'cost_basis': 40, 'shares': 150},
]
current_prices = [45, 80, 35]

tlh_opps = tax_loss_harvesting(positions, current_prices, 0.24)
print("Tax Loss Harvesting Opportunities:")
for opp in tlh_opps:
    print(f"  Position {opp['position']}: "
          f"Loss ${opp['loss']:.2f}, "
          f"Tax Benefit ${opp['tax_benefit']:.2f}")
```

### Asset Location

```python
def optimize_asset_location(assets, taxable_space, tax_advantaged_space):
    """
    Optimal asset placement across account types

    General rules:
    - Tax-inefficient assets → Tax-advantaged accounts
    - Tax-efficient assets → Taxable accounts
    """
    location_strategy = {
        'tax_advantaged': {
            'bonds': 'High interest income',
            'reits': 'High dividends',
            'actively_managed_funds': 'High turnover/distributions',
            'high_yield_bonds': 'Ordinary income'
        },
        'taxable': {
            'index_funds': 'Low turnover',
            'growth_stocks': 'Deferred capital gains',
            'municipal_bonds': 'Tax-exempt interest',
            'tax_managed_funds': 'Tax-efficient'
        }
    }

    return location_strategy

strategy = optimize_asset_location({}, 100000, 200000)
print("Asset Location Strategy:")
print("\nTax-Advantaged Accounts:")
for asset, reason in strategy['tax_advantaged'].items():
    print(f"  {asset}: {reason}")
print("\nTaxable Accounts:")
for asset, reason in strategy['taxable'].items():
    print(f"  {asset}: {reason}")
```

## Best Practices

### Portfolio Management Principles

1. **Define Clear Objectives**: Know your goals, timeline, and risk tolerance
2. **Asset Allocation First**: Most important decision (90% of return variance)
3. **Diversify Broadly**: Across assets, sectors, geographies
4. **Keep Costs Low**: Minimize fees and taxes
5. **Rebalance Systematically**: Maintain target allocation
6. **Stay Disciplined**: Avoid emotional decisions
7. **Monitor Regularly**: Review performance and adjust as needed
8. **Think Long-Term**: Don't react to short-term noise

### Common Mistakes to Avoid

1. **Over-concentration**: Too much in single stock/sector
2. **Chasing Performance**: Buying last year's winners
3. **Ignoring Costs**: High fees erode returns
4. **Market Timing**: Trying to predict market movements
5. **Emotional Decisions**: Panic selling or greed buying
6. **Neglecting Rebalancing**: Letting portfolio drift
7. **Ignoring Taxes**: Not considering tax efficiency
8. **Over-diversification**: Too many holdings to manage

### Portfolio Review Checklist

```python
def portfolio_review_checklist():
    """Quarterly portfolio review items"""
    checklist = {
        'performance': [
            'Compare returns to benchmarks',
            'Check risk-adjusted metrics (Sharpe, Sortino)',
            'Review drawdown periods',
            'Analyze attribution (what drove returns)'
        ],
        'allocation': [
            'Current vs. target allocation',
            'Drift from targets',
            'Rebalancing needed?',
            'Changes to strategic allocation?'
        ],
        'risk': [
            'Portfolio volatility acceptable?',
            'Concentration risk?',
            'Correlation changes?',
            'Stress test scenarios'
        ],
        'costs': [
            'Expense ratios',
            'Trading costs',
            'Tax efficiency',
            'Opportunities for improvement'
        ],
        'goals': [
            'On track for objectives?',
            'Life changes affecting goals?',
            'Timeline changes?',
            'Risk tolerance changes?'
        ]
    }
    return checklist

checklist = portfolio_review_checklist()
print("Portfolio Review Checklist:")
for category, items in checklist.items():
    print(f"\n{category.upper()}:")
    for item in items:
        print(f"  □ {item}")
```

## Tools and Resources

### Portfolio Management Software

- **Morningstar Direct**: Professional portfolio analysis
- **Bloomberg Terminal**: Institutional-grade tools
- **Personal Capital**: Free portfolio tracking
- **Vanguard Portfolio Watch**: Fund analysis
- **Python Libraries**: pandas, numpy, scipy, PyPortfolioOpt

### Educational Resources

- CFA Institute - Portfolio Management
- Modern Portfolio Theory (Markowitz, 1952)
- "A Random Walk Down Wall Street" (Malkiel)
- "The Intelligent Asset Allocator" (Bernstein)

## Key Takeaways

1. **Asset allocation** is the primary driver of portfolio returns
2. **Diversification** reduces risk without sacrificing expected return
3. **Rebalancing** maintains your target risk/return profile
4. **Costs matter** - every dollar in fees is a dollar not compounding
5. **Risk-adjusted returns** matter more than absolute returns
6. **Tax efficiency** can significantly improve after-tax returns
7. **Discipline** beats market timing
8. **Regular reviews** keep portfolio aligned with goals

## Next Steps

1. Define your investment objectives and constraints
2. Determine your risk tolerance
3. Choose your strategic asset allocation
4. Select specific investments (index funds, ETFs, etc.)
5. Implement your portfolio
6. Set rebalancing rules
7. Monitor performance regularly
8. Adjust as life circumstances change

Remember: Portfolio management is a continuous process, not a one-time event. Regular monitoring and disciplined rebalancing are keys to long-term success.

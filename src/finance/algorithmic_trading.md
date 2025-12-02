# Algorithmic Trading

## Overview

Algorithmic trading uses computer programs to execute trades automatically based on predefined rules. It ranges from simple rule-based systems to complex machine learning models.

## Types of Algorithmic Trading

### 1. Trend Following
- Moving average crossovers
- Breakout systems
- Channel trading
- No prediction needed, just follow price

### 2. Mean Reversion
- Price returns to average
- Bollinger Bands
- RSI extremes
- Pairs trading

### 3. Arbitrage
- Statistical arbitrage
- Cross-exchange arbitrage
- Index arbitrage
- Latency arbitrage (HFT)

### 4. Market Making
- Provide liquidity
- Capture bid-ask spread
- High frequency, low margin
- Requires sophisticated infrastructure

### 5. Sentiment Analysis
- News sentiment
- Social media analysis
- Alternative data
- Natural language processing

## Components of Trading System

### 1. Data Collection
- Historical data
- Real-time market data
- Alternative data sources
- Clean and normalize data

### 2. Strategy Development
- Define entry/exit rules
- Position sizing
- Risk management
- Backtesting

### 3. Backtesting Engine
- Historical simulation
- Walk-forward analysis
- Out-of-sample testing
- Avoid overfitting

### 4. Execution System
- Order routing
- Slippage management
- Position tracking
- Real-time monitoring

### 5. Risk Management
- Position limits
- Stop losses
- Portfolio heat
- Circuit breakers

## Simple Trading Algorithm Example

```python
"""
Simple Moving Average Crossover Strategy
"""

def moving_average_strategy(prices, short_window=50, long_window=200):
    """
    Buy when short MA crosses above long MA (Golden Cross)
    Sell when short MA crosses below long MA (Death Cross)
    """
    import pandas as pd
    import numpy as np
    
    # Calculate moving averages
    df = pd.DataFrame({'price': prices})
    df['short_ma'] = df['price'].rolling(window=short_window).mean()
    df['long_ma'] = df['price'].rolling(window=long_window).mean()
    
    # Generate signals
    df['signal'] = 0
    df['signal'][short_window:] = np.where(
        df['short_ma'][short_window:] > df['long_ma'][short_window:], 1, 0
    )
    
    # Generate trading orders
    df['position'] = df['signal'].diff()
    
    # 1 = Buy, -1 = Sell, 0 = Hold
    return df

# Backtest function
def backtest_strategy(df, initial_capital=100000):
    """Calculate returns from strategy"""
    positions = pd.DataFrame(index=df.index).fillna(0.0)
    
    # Buy 100 shares when signal = 1
    positions['stock'] = 100 * df['signal']
    
    # Calculate portfolio value
    portfolio = positions.multiply(df['price'], axis=0)
    
    # Calculate returns
    pos_diff = positions.diff()
    portfolio['holdings'] = (positions.multiply(df['price'], axis=0)).sum(axis=1)
    portfolio['cash'] = initial_capital - (pos_diff.multiply(df['price'], axis=0)).sum(axis=1).cumsum()
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    portfolio['returns'] = portfolio['total'].pct_change()
    
    return portfolio
```

## Key Performance Metrics

### Return Metrics
- **Total Return**: Overall gain/loss
- **CAGR**: Compound annual growth rate
- **Sharpe Ratio**: Risk-adjusted returns (>1 good, >2 excellent)
- **Sortino Ratio**: Downside risk-adjusted returns
- **Calmar Ratio**: Return / Maximum Drawdown

### Risk Metrics
- **Maximum Drawdown**: Peak-to-trough decline
- **Volatility**: Standard deviation of returns
- **Value at Risk (VaR)**: Maximum expected loss
- **Win Rate**: % of profitable trades
- **Profit Factor**: Gross profit / Gross loss

### Trading Metrics
- **Number of Trades**: Sample size matters
- **Average Trade**: Mean profit/loss per trade
- **Average Win vs. Average Loss**: Risk/reward
- **Consecutive Losses**: Drawdown tolerance

## Backtesting Best Practices

### Avoid Overfitting
- **In-sample vs. Out-of-sample**: Test on unseen data
- **Walk-forward analysis**: Rolling optimization
- **Parameter stability**: Small changes shouldn't crash strategy
- **Occam's Razor**: Simpler is better

### Realistic Assumptions
- **Transaction costs**: Commissions, slippage, spreads
- **Liquidity**: Can you execute at displayed prices?
- **Market impact**: Large orders move prices
- **Latency**: Delay between signal and execution

### Data Quality
- **Survivorship bias**: Include delisted companies
- **Look-ahead bias**: Only use data available at that time
- **Adjustment for splits/dividends**: Corporate actions
- **Accurate timestamps**: Correct sequencing

## Common Pitfalls

1. **Curve Fitting**: Optimized for past, fails in future
2. **Ignoring Costs**: Commissions kill high-frequency strategies
3. **Not Testing Out-of-Sample**: Works in backtest only
4. **Insufficient Data**: Need enough trades for statistical significance
5. **Regime Changes**: Markets evolve, strategies decay
6. **No Risk Management**: One bad trade wipes out account
7. **Overconfidence**: Backtest ≠ live performance
8. **Ignoring Slippage**: Assumed fills at best price

## Risk Management Rules

### Position Sizing
- **Fixed Fractional**: Risk 1-2% per trade
- **Kelly Criterion**: Optimal fraction (often too aggressive)
- **Volatility-based**: Adjust size based on ATR/volatility

### Portfolio Limits
- **Maximum Position Size**: 10-20% per position
- **Maximum Portfolio Heat**: Total risk <6-10% of capital
- **Correlation Limits**: Don't stack correlated bets
- **Sector Limits**: Diversify across sectors

### Stop Losses
- **Initial Stop**: Set at entry
- **Trailing Stop**: Lock in profits
- **Time Stop**: Exit if no progress
- **Maximum Loss**: Circuit breaker for day/week

## Machine Learning in Trading

### Applications
- **Price Prediction**: Regression models
- **Classification**: Buy/sell/hold signals
- **Feature Engineering**: Create predictive features
- **Sentiment Analysis**: NLP on news/social media
- **Portfolio Optimization**: Reinforcement learning

### Challenges
- **Non-stationarity**: Financial data changes over time
- **Low Signal-to-Noise**: Hard to predict
- **Overfitting**: Easy with complex models
- **Regime Changes**: Models trained on one regime fail in another
- **Computational Cost**: Real-time inference required

### Approaches
- **Random Forests**: Robust, handle non-linear relationships
- **Neural Networks**: Deep learning for pattern recognition
- **Reinforcement Learning**: Learn optimal policy
- **Ensemble Methods**: Combine multiple models
- **Time Series Models**: ARIMA, LSTM for sequences

## Technology Stack

### Languages
- **Python**: Most popular (pandas, numpy, scikit-learn)
- **C++**: High-frequency trading (speed)
- **R**: Statistical analysis
- **Julia**: Growing (speed + ease of Python)

### Libraries
- **Data**: pandas, numpy
- **Backtesting**: backtrader, zipline, bt
- **Machine Learning**: scikit-learn, TensorFlow, PyTorch
- **Charting**: matplotlib, plotly

### Platforms
- **QuantConnect**: Cloud backtesting, live trading
- **Quantopian** (defunct): Legacy research
- **Interactive Brokers**: API for live trading
- **Alpaca**: Commission-free API trading

## Regulations and Compliance

### Pattern Day Trader Rule (US)
- >4 day trades in 5 days with margin account
- Requires $25,000 minimum
- Applies to margin accounts

### Market Manipulation
- **Spoofing**: Fake orders to manipulate
- **Front-running**: Trading ahead of clients
- **Wash Trading**: Buying and selling own orders
- All illegal

### Data Rights
- Market data licenses required
- Redistribution restrictions
- Real-time vs. delayed data costs

## Getting Started

### 1. Education (Months 1-3)
- Learn Python and pandas
- Study market microstructure
- Read algorithmic trading books
- Paper trade manual strategies

### 2. Build Simple System (Months 4-6)
- Implement simple strategy (MA crossover)
- Backtest properly
- Understand all metrics
- Paper trade

### 3. Refine and Test (Months 7-12)
- Improve strategy
- Add risk management
- Out-of-sample testing
- Walk-forward analysis

### 4. Small Live Trading (Year 2)
- Start with small capital
- Single strategy
- Monitor closely
- Expect learning curve

### 5. Scale Gradually
- Prove profitability first
- Add capital slowly
- Diversify strategies
- Automate monitoring

## Realistic Expectations

### Returns
- **Beginner**: 0-5% (learning, likely losses)
- **Intermediate**: 5-15% annually
- **Advanced**: 15-30% annually
- **Professional**: 20-40% (with drawdowns)

### Success Rate
- 90-95% of retail algo traders lose money
- Requires significant time, skill, capital
- Edge erodes over time
- Constant adaptation needed

### Time Commitment
- **Development**: 500-1000+ hours initially
- **Maintenance**: 5-20 hours/week
- **Monitoring**: Daily check-ins
- **Learning**: Continuous

## Resources

### Books
- "Algorithmic Trading" - Ernest Chan
- "Quantitative Trading" - Ernest Chan
- "Advances in Financial Machine Learning" - Marcos López de Prado
- "Trading Systems" - Emilio Tomasini

### Courses
- Coursera: Machine Learning for Trading
- QuantInsti: Algo trading courses
- Udemy: Python for finance

### Communities
- QuantConnect Forum
- r/algotrading (Reddit)
- Elite Trader
- Wilmott Forums

### Tools
- **Backtesting**: Backtrader, Zipline, VectorBT
- **Data**: Yahoo Finance, Alpha Vantage, Quandl
- **Brokers**: Interactive Brokers, Alpaca
- **Cloud**: AWS, QuantConnect

## Key Takeaways

1. **Algo trading is hard**: Most fail
2. **Backtesting ≠ profits**: Live trading is different
3. **Risk management crucial**: Survive to profit
4. **Start simple**: Complex doesn't mean better
5. **Transaction costs matter**: Frequent trading expensive
6. **Continuous learning**: Markets evolve
7. **Proper testing essential**: Avoid overfitting
8. **Capital required**: Need cushion for drawdowns
9. **Time intensive**: Not passive income initially
10. **Low expectations**: 15-20% annually is excellent

## Warning Signs

- Promises of guaranteed returns
- Black box systems
- >50% annual returns claimed
- No track record
- Expensive courses/software
- "Holy grail" systems

Remember: If it sounds too good to be true, it is. Algorithmic trading can be profitable but requires skill, discipline, capital, and realistic expectations. Start small, test thoroughly, and prepare for losses while learning.

See also: [Technical Analysis](technical_analysis.md), [Risk Management](risk_management.md)

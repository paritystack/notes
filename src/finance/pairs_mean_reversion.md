# Pairs Trading and Mean Reversion

## Overview

Mean reversion strategies bet that prices, spreads, or ratios that have stretched away from their historical average will snap back. Pairs trading is the classic example: long one asset and short a related one, profiting when the spread converges. These strategies are market-neutral (or close to it), thrive in range-bound markets, and historically smooth equity-curve volatility — but they get killed when relationships break.

## Why Mean Reversion Works (When It Works)

- **Liquidity provision**: Mean-reverters supply liquidity to forced sellers/buyers
- **Behavioral overreaction**: Investors panic and overshoot; reversion follows
- **Statistical reality**: Many financial relationships are stationary around an anchor
- **Microstructure**: Order imbalances create temporary dislocations

**When it fails**: Trends, structural changes, fundamental divergence (e.g., one company missing earnings while its "pair" beats).

## The Core Setup: Pairs Trading

### The Trade

1. Find two correlated/cointegrated assets (e.g., KO and PEP)
2. Calculate the spread (price difference, ratio, or regression residual)
3. When spread is unusually wide → short the rich one, long the cheap one
4. When spread converges → close both legs for profit

### Picking Pairs

**Industry pairs** (most intuitive)
- Coca-Cola (KO) / PepsiCo (PEP)
- Visa (V) / Mastercard (MA)
- Home Depot (HD) / Lowe's (LOW)
- Ford (F) / General Motors (GM)
- Goldman Sachs (GS) / Morgan Stanley (MS)
- Chevron (CVX) / Exxon (XOM)

**ETF pairs**
- SPY / IVV (S&P 500 mirrors — tiny spread, low edge)
- XLF / KBE (sector vs. sub-sector)
- IWM / IWN (small-cap vs. small value)
- USO / BNO (WTI vs. Brent oil)
- GLD / SLV (gold vs. silver)

**Cross-listings / ADRs**
- BABA ADR vs. 9988.HK
- RIO ADR vs. RIO.AX

**Index arbitrage**
- ETF vs. basket of underlyings
- Futures vs. spot

## Cointegration vs. Correlation

**Correlation** = returns move together
**Cointegration** = prices have a stable long-run relationship

Two random walks can be highly correlated by chance. Cointegration requires their linear combination to be **stationary** (mean-reverting). For pairs trading, you want **cointegration**, not just correlation.

### Engle-Granger Test

1. Regress price A on price B: `A_t = α + β·B_t + ε_t`
2. Test the residuals `ε_t` for stationarity (Augmented Dickey-Fuller test)
3. If residuals are stationary → the pair is cointegrated, residuals = the spread to trade

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
import statsmodels.api as sm

def test_cointegration(price_a, price_b, significance=0.05):
    """Engle-Granger cointegration test. Returns (is_cointegrated, p_value, hedge_ratio)."""
    # OLS regression
    X = sm.add_constant(price_b)
    model = sm.OLS(price_a, X).fit()
    hedge_ratio = model.params[1]
    residuals = model.resid
    
    # ADF test on residuals
    adf_result = adfuller(residuals)
    p_value = adf_result[1]
    
    return p_value < significance, p_value, hedge_ratio

def find_pairs(price_df, p_threshold=0.05):
    """Test all pairs in a price DataFrame for cointegration."""
    pairs = []
    tickers = price_df.columns
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            t1, t2 = tickers[i], tickers[j]
            try:
                _, p, hedge = test_cointegration(price_df[t1], price_df[t2])
                if p < p_threshold:
                    pairs.append({'pair': (t1, t2), 'p_value': p, 'hedge_ratio': hedge})
            except Exception:
                continue
    return sorted(pairs, key=lambda x: x['p_value'])
```

## Spread Construction

Once you have a pair, you need a spread that reverts to a mean.

### Common Spread Definitions

**Price spread** (only for similar-priced assets)
```
spread = A - B
```

**Ratio**
```
spread = A / B
```

**Log spread**
```
spread = log(A) - β · log(B)
```

**Hedge-ratio (cointegration residual)** — most rigorous
```
spread = A - β · B
```
where β comes from OLS or rolling regression.

```python
def calculate_spread(price_a, price_b, hedge_ratio):
    """OLS-style spread."""
    return price_a - hedge_ratio * price_b

def z_score(spread, lookback=60):
    """Rolling z-score of the spread."""
    mean = spread.rolling(lookback).mean()
    std = spread.rolling(lookback).std()
    return (spread - mean) / std
```

## Entry / Exit Rules

### Z-Score Bands (canonical)

| Z-score | Action |
|---------|--------|
| +2.0 (or higher) | Short the spread (short A, long β·B) |
| –2.0 (or lower) | Long the spread (long A, short β·B) |
| 0.0 | Close position |
| ±3.5 | Stop-out (relationship may have broken) |

```python
def pairs_signal(z, entry=2.0, exit=0.0, stop=3.5):
    if abs(z) > stop:
        return 'flat (stop)'
    if z > entry:
        return 'short_spread'
    if z < -entry:
        return 'long_spread'
    if abs(z) < exit:
        return 'flat (target)'
    return 'hold'

def position_sizes(capital, price_a, price_b, hedge_ratio, signal):
    """
    Returns (shares_a, shares_b) for a dollar-neutral pairs trade.
    Positive shares = long, negative = short.
    """
    if signal == 'long_spread':
        notional = capital / 2
        shares_a = notional / price_a
        shares_b = -hedge_ratio * notional / price_b
    elif signal == 'short_spread':
        notional = capital / 2
        shares_a = -notional / price_a
        shares_b = hedge_ratio * notional / price_b
    else:
        shares_a, shares_b = 0, 0
    return shares_a, shares_b
```

### Time-based Exit

Force-close after N days if the spread hasn't reverted — prevents holding broken pairs forever.

### Walk-Forward Re-Estimation

Re-fit the hedge ratio every 1–3 months. Markets evolve; static parameters decay.

## Half-Life of Mean Reversion

Estimate how long it takes the spread to revert halfway to its mean. Useful for sizing holding period.

```python
def half_life(spread):
    """Ornstein-Uhlenbeck half-life via AR(1) regression."""
    spread = spread.dropna()
    lag = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    lag = lag.loc[delta.index]
    
    X = sm.add_constant(lag)
    model = sm.OLS(delta, X).fit()
    theta = -model.params[1]
    return np.log(2) / theta if theta > 0 else np.inf
```

Pairs with half-life < 5 days are usually noise; > 60 days are too slow for tactical trading. Sweet spot: 5–30 days.

## Mean Reversion in Single Names

You don't need a pair. Mean reversion exists in single stocks/indices at short horizons.

### RSI(2) Strategy (Connors)

- Buy SPY when RSI(2) < 5 and price > 200dma
- Sell when RSI(2) > 70 (or 5 days, whichever first)
- Works because broad indices revert short-term while trending long-term

### Bollinger Band Reversion

- Buy at lower band (mean − 2σ)
- Sell at middle band or upper band
- Best on range-bound, high-liquidity instruments
- Filter: only in established uptrend (price > 200dma)

```python
def rsi(prices, period=2):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)

def connors_rsi2_signal(prices, sma_200, rsi_period=2, buy_thresh=5, sell_thresh=70):
    rsi_vals = rsi(prices, rsi_period)
    if prices.iloc[-1] > sma_200.iloc[-1]:
        if rsi_vals.iloc[-1] < buy_thresh:
            return 'buy'
        if rsi_vals.iloc[-1] > sell_thresh:
            return 'sell'
    return 'hold'
```

### Overnight Drift / Gap Fade

- Stocks that gap down on no news tend to recover intraday
- Heavily-down close on broad indices → next-day mean reversion (limited edge in modern markets)

## Statistical Arbitrage (Stat Arb)

Pairs trading scaled to 100s/1000s of names. Each position is small; portfolio-level edge comes from law of large numbers.

### Typical Approach

1. Universe: liquid stocks (e.g., S&P 1500)
2. Compute residuals from a factor model (market, sector, size, value, momentum)
3. Rank by short-term return divergence from peers/factors
4. Buy bottom decile, short top decile (today's underperformers, today's outperformers)
5. Rebalance daily

### Decay

Classical stat-arb returns have decayed substantially since the 2000s due to crowding (Renaissance, AQR, Two Sigma, every quant fund). Edge now requires:
- Faster execution
- Better residualization (more factors)
- Alternative data
- Or moving to less crowded universes (international, micro-caps)

## Index Arbitrage

ETF prices can deviate from NAV briefly. Authorized Participants (APs) arbitrage the gap by creating/redeeming shares. Retail can't do this directly but can:
- Track premium/discount on closed-end funds
- Trade ETF vs. futures basis (institutional)
- Look for stale-NAV opportunities in foreign ETFs at open

## Risks of Mean Reversion

### Regime Breaks

The relationship that held for 5 years can break overnight (M&A, restructuring, scandal, regulatory change). This is the biggest killer.

**Mitigation**:
- Hard stop-out on extreme z-scores (>3.5)
- Position-size each pair small (no single break ruins the portfolio)
- Fundamental sanity check before entry
- Avoid earnings overlap (close before reports)

### Crowded Trades

When everyone runs the same model:
- Spreads barely widen → low entry
- Unwinds become violent (Aug 2007 quant crisis)
- Liquidity disappears in a panic

### Carrying Costs

- **Short borrow fees** can exceed reversion profit on hard-to-borrow names
- **Dividend payments on shorts** owed to lender
- **Financing costs** for leveraged pairs

### Trending Markets

Mean reversion gets crushed when:
- One asset rerates structurally (paradigm shift)
- Momentum dominates (e.g., 2020 growth blowoff)
- Volatility spikes (correlations break in crises)

## Trade Construction Checklist

- [ ] Cointegration p-value < 0.05 over multi-year window
- [ ] Half-life in 5–30 days
- [ ] Sufficient liquidity in both legs (no impact > 10bp expected)
- [ ] Borrow available on the short leg, fee < 50bp
- [ ] No earnings or major event within holding period
- [ ] No M&A speculation
- [ ] Dollar-neutral (or beta-neutral) position sizing
- [ ] Stop-loss defined ahead of entry
- [ ] Max holding period defined
- [ ] Position size < 2% of capital per pair

## Backtesting Pairs

Common pitfalls:

1. **Look-ahead bias** — using full-sample cointegration to filter, then "backtesting"
2. **Survivorship bias** — only including pairs that exist today
3. **Ignoring borrow costs** — short-side fees can erase edge
4. **Ignoring slippage** — wide spreads + thin books eat alpha
5. **Static hedge ratios** — must roll/re-estimate
6. **Insufficient sample** — need many pairs × many trades for statistical significance

```python
def backtest_pair(price_a, price_b, lookback=60, entry=2.0, exit=0.0,
                  stop=3.5, max_days=30, borrow_fee_annual=0.005):
    """
    Skeleton pairs backtest. Real version needs:
    - rolling hedge ratio re-estimation
    - daily borrow accrual
    - transaction costs
    - position-level P&L tracking
    """
    trades = []
    position = 0  # -1 short spread, 1 long, 0 flat
    entry_idx = None
    
    for t in range(lookback, len(price_a)):
        window = slice(t - lookback, t)
        _, _, hedge = test_cointegration(price_a[window], price_b[window])
        spread = price_a[window] - hedge * price_b[window]
        mean, std = spread.mean(), spread.std()
        z = (price_a.iloc[t] - hedge * price_b.iloc[t] - mean) / std
        
        if position == 0:
            if z > entry:
                position, entry_idx, entry_z = -1, t, z
            elif z < -entry:
                position, entry_idx, entry_z = 1, t, z
        else:
            held_days = t - entry_idx
            if abs(z) > stop or held_days > max_days or \
               (position == 1 and z >= exit) or (position == -1 and z <= -exit):
                # close, record trade
                trades.append({
                    'entry_t': entry_idx, 'exit_t': t,
                    'entry_z': entry_z, 'exit_z': z,
                    'side': position, 'held_days': held_days,
                })
                position = 0
    return pd.DataFrame(trades)
```

## Combining with Other Strategies

Mean reversion pairs well with:
- **Trend following** ([[momentum_trend]]) — diversification; opposite-style edges
- **Carry trades** — both shine in low-vol regimes
- **Vol selling** ([[volatility_trading]]) — similar payoff profile but uncorrelated drivers

Avoid stacking mean-revert with mean-revert (RSI(2) + pairs + vol selling) — they all crash in the same scenarios (vol spikes, correlation breakdowns).

## Real-World Examples

- **LTCM (1998)** — Treasury convergence trades on massive leverage; blew up when spreads diverged during Russian default panic
- **August 2007 Quant Quake** — every stat-arb fund forced to unwind simultaneously; 3-sigma daily losses for weeks
- **March 2020** — ETF discounts to NAV exploded as liquidity vanished; APs couldn't keep up
- **GameStop (2021)** — short-side mean reversion got run over by retail momentum

## Resources

- **Quantopian** archive — pairs trading lectures (still online via internet archive)
- **statsmodels** — cointegration tests, ADF, OLS
- **vectorbt** / **backtrader** — backtesting frameworks

### Books
- *Pairs Trading* — Ganapathy Vidyamurthy
- *Algorithmic Trading* — Ernie Chan
- *Quantitative Trading* — Ernie Chan
- *Inside the Black Box* — Rishi Narang
- *Active Portfolio Management* — Grinold & Kahn

### Papers
- Gatev, Goetzmann, Rouwenhorst — "Pairs Trading: Performance of a Relative-Value Arbitrage Rule" (2006)
- Khandani & Lo — "What Happened to the Quants in August 2007?"

## Key Takeaways

1. **Cointegration > correlation** — only cointegrated pairs revert reliably
2. **Z-score bands work** — ±2 entry, 0 exit, ±3.5 stop is a solid starting framework
3. **Half-life filters out junk** — keep pairs that revert in 5–30 days
4. **Diversify across pairs** — no single relationship is safe
5. **Watch for regime breaks** — fundamentals matter even in stat arb
6. **Stat-arb is crowded** — pure spread trading edge has decayed; need extra angle
7. **Borrow costs matter** — short-side fees can be lethal on hard-to-borrow names
8. **Mean reversion fails in trends** — diversify with [[momentum_trend]] for balance

## Where this connects

- [Momentum trend](momentum_trend.md) — mean reversion and trend following are complementary; diversify between them
- [Derivatives](derivatives.md) — pairs trades are often implemented with swaps or CFDs for capital efficiency
- [Portfolio management](portfolio_management.md) — market-neutral pairs trades reduce beta exposure

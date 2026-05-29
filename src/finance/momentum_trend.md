# Momentum and Trend Following

## Overview

Momentum strategies buy what's been going up and sell what's been going down — the empirical opposite of mean reversion ([[pairs_mean_reversion]]). It's one of the most robust anomalies in finance: documented across stocks, bonds, commodities, currencies, and across decades and countries. Trend following is the time-series cousin (same instrument, recent direction); cross-sectional momentum ranks across many assets at once.

## Why Momentum Works

- **Behavioral**: Investors under-react to news, overreact to trends, anchor on prices
- **Informational**: Information diffuses slowly; smart money accumulates first
- **Frictional**: Slow institutional rebalancing extends moves
- **Risk-based (debated)**: Crash risk premium for momentum winners

Momentum returns concentrate in trending markets and reverse violently in regime changes ("momentum crashes" — 2009, 2020).

## Two Flavors

### Cross-Sectional (Relative) Momentum

- Rank a universe by past return (e.g., last 12 months excluding most recent)
- Long top quintile, short bottom quintile
- Rebalance monthly/quarterly
- Pure: dollar-neutral, market-neutral

### Time-Series (Absolute) Momentum / Trend Following

- For each instrument, go long if past return > 0, short (or cash) if < 0
- Works across asset classes, especially futures
- Doesn't need a peer group
- The bread-and-butter of CTAs (Commodity Trading Advisors)

## Cross-Sectional Momentum (Stocks)

### Classic Jegadeesh-Titman (1993)

- Universe: U.S. stocks
- Signal: 12-month return, skip last 1 month (avoid short-term reversal)
- Long top decile, short bottom decile
- Rebalance monthly
- Returned ~12% annualized over 1965–1989; positive in nearly every decade since

```python
import pandas as pd

def momentum_score(prices, lookback=252, skip=21):
    """
    12-1 momentum: return over past 252 days, skipping last 21.
    prices: DataFrame, dates x tickers
    """
    return prices.shift(skip) / prices.shift(lookback) - 1

def momentum_portfolio(prices, n_long=10, n_short=10, lookback=252, skip=21):
    """Returns a series of monthly long-short portfolio returns."""
    scores = momentum_score(prices, lookback, skip)
    monthly = prices.resample('M').last()
    monthly_scores = scores.resample('M').last()
    monthly_rets = monthly.pct_change()
    
    portfolio_rets = []
    for date in monthly_scores.index[1:]:
        s = monthly_scores.loc[date].dropna()
        longs = s.nlargest(n_long).index
        shorts = s.nsmallest(n_short).index
        next_date = monthly_rets.index[monthly_rets.index.get_loc(date) + 1] \
                    if monthly_rets.index.get_loc(date) + 1 < len(monthly_rets) else None
        if next_date:
            long_ret = monthly_rets.loc[next_date, longs].mean()
            short_ret = monthly_rets.loc[next_date, shorts].mean()
            portfolio_rets.append({'date': next_date, 'return': long_ret - short_ret})
    return pd.DataFrame(portfolio_rets).set_index('date')
```

### Variants

| Variant | Lookback | Skip | Notes |
|---------|----------|------|-------|
| Jegadeesh-Titman | 12m | 1m | Classic |
| Short-term | 1–3m | 1w | More volatile, higher turnover |
| Long-term | 36–60m | — | Reverses (long-term mean reversion) |
| Industry momentum | 12m | 1m | Apply to sector ETFs |
| 52-week high | distance from 52w high | — | Often outperforms 12-1 |

### Filters That Improve Momentum

- **Quality screen** — exclude low-margin, high-debt names (reduces crashes)
- **Volatility scaling** — equal-risk weights instead of equal-dollar
- **Trend filter** — only long when market in uptrend (skip momentum in bear markets)
- **Skip recent reversal** — the "skip 1 month" rule is essential

## Time-Series Momentum / Trend Following

### Setup

For each instrument independently:

```
if last_N_month_return > 0:  go long
else:                         go short (or cash)
```

Common lookbacks: 1, 3, 6, 12 months. Use multiple lookbacks and average for smoother signal.

### Moskowitz-Ooi-Pedersen (2012)

Tested 12-month TSMOM across 58 instruments (commodities, equities, bonds, FX). Sharpe ~1.4 over 1985–2009, low correlation to traditional assets, positive in 2008. Foundational paper for trend following.

```python
def time_series_momentum(prices, lookback_months=12):
    """
    Returns +1 (long), -1 (short), or 0 (flat) for each instrument.
    prices: DataFrame of monthly prices, columns = instruments
    """
    monthly_ret = prices.pct_change(lookback_months)
    signal = monthly_ret.copy()
    signal[signal > 0] = 1
    signal[signal < 0] = -1
    return signal.shift(1)  # lag to avoid look-ahead

def trend_portfolio(prices, lookbacks=(3, 6, 12), vol_target=0.10):
    """
    Multi-lookback trend with volatility targeting.
    """
    signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    for lb in lookbacks:
        signals = signals + time_series_momentum(prices, lb)
    signals = signals / len(lookbacks)  # average to [-1, 1]
    
    # Volatility targeting per instrument
    returns = prices.pct_change()
    vol = returns.rolling(60).std() * (252 ** 0.5)
    weights = signals * (vol_target / vol)
    weights = weights.clip(-1, 1)  # cap leverage per instrument
    return weights
```

### Multi-Lookback Ensemble

Combining 1, 3, 6, and 12-month signals reduces single-horizon noise. Each lookback captures different cycle frequencies. Industry standard for CTAs.

## Moving-Average Trend Systems

Simple and robust. Used since the 1970s.

### Dual Moving Average Crossover

- Buy when fast MA (e.g., 50-day) crosses above slow MA (e.g., 200-day)
- Sell on the reverse cross

### Triple MA / Multi-Timeframe

- Confirm trend across 3 timeframes (daily, weekly, monthly)

### Donchian Channel (Turtle Trader system)

- Buy when price breaks above N-day high (e.g., 20 or 55 days)
- Exit at N-day low
- Classic from Richard Dennis's "Turtles" experiment

```python
def donchian_signal(prices, entry_lookback=20, exit_lookback=10):
    """Long-only Donchian breakout."""
    entry = prices.rolling(entry_lookback).max().shift(1)
    exit = prices.rolling(exit_lookback).min().shift(1)
    signal = pd.Series(index=prices.index, dtype=float)
    in_pos = False
    for t in range(len(prices)):
        p = prices.iloc[t]
        if not in_pos and not pd.isna(entry.iloc[t]) and p > entry.iloc[t]:
            in_pos = True
            signal.iloc[t] = 1
        elif in_pos and not pd.isna(exit.iloc[t]) and p < exit.iloc[t]:
            in_pos = False
            signal.iloc[t] = 0
        else:
            signal.iloc[t] = 1 if in_pos else 0
    return signal
```

## Position Sizing for Trend

Trend strategies are extremely sensitive to position sizing. Bad sizing = good system, bad results.

### Volatility Targeting

Size each position so its expected risk contribution is equal.

```python
def vol_target_size(capital, entry_price, atr, target_risk_pct=0.01):
    """
    atr = Average True Range (volatility proxy)
    target_risk_pct = % of capital risked per ATR move
    """
    dollar_risk = capital * target_risk_pct
    return int(dollar_risk / atr)
```

### ATR-Based Stop

Standard Turtle rule: stop at 2 × ATR below entry. Position sized so 2N move = 2% of capital.

### Risk Parity Across Instruments

In multi-asset trend portfolios, allocate by inverse volatility so a 1% bond move and a 1% commodity move contribute the same portfolio variance. Crucial for not having S&P futures dominate a CTA portfolio.

## Dual Momentum (Antonacci)

Combines absolute and relative momentum.

1. Compare U.S. equities to international equities → buy the winner
2. But only if that winner has positive absolute 12-month return (else go to bonds)

```python
def dual_momentum(us_eq, intl_eq, bonds, lookback=252):
    """Monthly signal: returns one of 'us_eq', 'intl_eq', 'bonds'."""
    r_us = us_eq.iloc[-1] / us_eq.iloc[-lookback] - 1
    r_intl = intl_eq.iloc[-1] / intl_eq.iloc[-lookback] - 1
    winner = 'us_eq' if r_us > r_intl else 'intl_eq'
    winner_ret = r_us if winner == 'us_eq' else r_intl
    return winner if winner_ret > 0 else 'bonds'
```

Historically: ~15% CAGR with bond-like drawdowns. The "absolute filter" avoids 2008 and 2022.

## Sector and Factor Momentum

### Sector Rotation by Momentum

Rank 11 GICS sectors by 6- or 12-month return, hold top 3 equal-weight, rebalance monthly. Simple, mechanical, and historically beats buy-and-hold S&P.

### Factor Momentum

Apply momentum across factors (value, quality, momentum, low-vol, size). Tilt toward factors that have been working. Less crowded than stock momentum.

## Combining Momentum with Other Signals

### Momentum + Quality

Buy momentum winners that also have high ROE, low debt, stable earnings. Avoids the "junk momentum" that crashes hardest.

### Momentum + Trend Filter

Only run cross-sectional long-short momentum when market in uptrend. Switch to cash or defensive when SPY < 200dma. Cuts the worst drawdowns.

### Momentum + Volatility Filter

Down-weight signals when realized vol is extreme — crashes tend to occur in high-vol regimes.

## Drawdowns and Momentum Crashes

Momentum's Achilles heel: violent reversals when the market regime flips.

### Notable crashes

| Period | Trigger | Long-short MOM drawdown |
|--------|---------|-------------------------|
| 1932 | End of Depression bear | –91% |
| 2009 (Mar) | Equity bottom rally | –40%+ |
| 2020 (Apr–May) | COVID growth-to-value flip | –30% |
| 2022 (Q4) | Inflation peak; growth crushed | –20%+ |

**Cause**: Past losers (low-quality, distressed) rally violently off bottoms; past winners (defensives) lag. Long-short MOM gets hit on both legs.

**Defense**:
- Dynamic volatility targeting (Barroso-Santa Clara managed momentum)
- Trend filter on the broader market
- Crash hedge (puts, gold, long Treasuries)
- Diversify across asset classes

## Real CTA / Trend-Following Funds

- **AQR Managed Futures**, **Man AHL**, **Winton**, **Aspect Capital**, **Campbell & Co.**
- Typically trade 50–200 futures markets across asset classes
- Multi-timeframe trend ensembles
- 10–20% annualized volatility
- Strong in 2008 (+15 to +30%), 2022 (+25 to +40%)
- Often poor in chop years (2011–17 dragged returns)

## Implementation for Retail

### Cross-Sectional Momentum

- Use a momentum ETF (MTUM, VFMO, QMOM, FDMO)
- Build manually with 30–50 stocks ranked by 12-1 momentum
- Rebalance monthly or quarterly

### Trend Following

- Tactical asset allocation with monthly trend filters across SPY, EFA, EEM, IEF, GLD, DBC
- ETFs that mimic CTAs: KMLM, DBMF, FMF
- Direct futures requires capital + infrastructure

### Tactical Buy/Hold

The simplest workable trend rule (Faber, 2007):
- Long SPY when price > 10-month SMA
- Cash when below
- Backtests show similar return to buy-hold with half the drawdown

```python
def faber_tactical(monthly_prices, sma_months=10):
    sma = monthly_prices.rolling(sma_months).mean()
    return (monthly_prices > sma).astype(int).shift(1)  # 1=invested, 0=cash
```

## Backtest Pitfalls

1. **Look-ahead bias** — using future data to compute signals
2. **Survivorship bias** — using current index constituents historically
3. **Transaction costs** — momentum has high turnover; cost-realistic backtests required
4. **Borrow costs on shorts** — significant for stock momentum
5. **Crowding decay** — pre-2003 momentum returns higher than post; some decay
6. **Ignoring crashes** — backtests over benign periods overstate Sharpe
7. **Overfitting lookback** — pick parameters robust across 6m, 12m, 18m, not just the best single one

## Combining Mean Reversion and Momentum

These two are negatively correlated — combining them smooths the equity curve dramatically.

- Mean reversion ([[pairs_mean_reversion]]): wins in choppy, range-bound markets
- Momentum/trend: wins in trending markets
- Hold both → diversification across regimes

Classic AQR "Style Premia" approach: value, momentum, carry, defensive — across asset classes — for a diversified factor sleeve.

## Resources

- **AQR** research papers — extensive momentum studies, "Fact, Fiction and Momentum Investing"
- **Two Centuries Investments** — long-historical momentum data
- **Allocate Smartly** — tactical allocation backtests
- **Portfolio Visualizer** — backtest tactical strategies free

### Books
- *Dual Momentum Investing* — Gary Antonacci
- *Quantitative Momentum* — Wesley Gray, Jack Vogel
- *Following the Trend* — Andreas Clenow
- *Trend Following* — Michael Covel
- *The Little Book of Trading* — Michael Covel
- *Active Portfolio Management* — Grinold & Kahn

### Papers
- Jegadeesh & Titman (1993) — original momentum paper
- Moskowitz, Ooi & Pedersen (2012) — "Time Series Momentum"
- Asness, Moskowitz & Pedersen (2013) — "Value and Momentum Everywhere"
- Barroso & Santa-Clara (2015) — "Momentum Has Its Moments"
- Faber (2007) — "A Quantitative Approach to Tactical Asset Allocation"

## Key Takeaways

1. **Momentum is real, persistent, and global** — across markets, asset classes, decades
2. **12-1 cross-sectional** is the canonical stock momentum
3. **Time-series momentum** works across asset classes (CTA approach)
4. **Skip the most recent month** — short-term reversal kills naive momentum
5. **Volatility-target** position sizes for risk balance
6. **Momentum crashes are real** — defenses: trend filter, vol targeting, crash hedges
7. **Combine with [[pairs_mean_reversion]]** — negative correlation smooths returns
8. **Simple trend rules work** — 10-month SMA cuts SPY drawdowns dramatically

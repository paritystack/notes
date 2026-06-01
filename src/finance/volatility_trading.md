# Volatility Trading

## Overview

Volatility is its own asset class. You can trade it directly via VIX futures, VIX options, and variance swaps, or indirectly via options strategies (straddles, strangles, condors). The key insight: **implied volatility (what options price in) is systematically higher than realized volatility (what actually happens)** — a persistent risk premium known as the **Variance Risk Premium (VRP)**. Most professional vol strategies harvest this premium, accepting tail risk in exchange.

## Core Concepts

### Implied vs. Realized Volatility

- **Implied Volatility (IV)** — the volatility priced into options (forward-looking)
- **Realized Volatility (RV)** — the volatility that actually occurred (backward-looking)
- **IV – RV spread** = Variance Risk Premium; historically positive on equity indices

### The Variance Risk Premium

Buyers of options (insurance) consistently pay more than the expected realized volatility. This premium accrues to sellers of volatility — analogous to insurance underwriters profiting from selling policies.

**Average VRP on SPX**: ~3–4 vol points (e.g., IV 18, RV 14)

```python
import numpy as np

def realized_vol(returns, periods_per_year=252):
    """Annualized realized volatility from daily returns."""
    return np.std(returns) * np.sqrt(periods_per_year)

def variance_risk_premium(iv, rv):
    """IV - RV spread in vol points."""
    return iv - rv
```

### Volatility Skew

For SPX, downside puts trade at higher IV than upside calls of equivalent moneyness — the "skew." Reflects demand for crash protection. Single-name stocks often show "smile" — both tails priced rich.

### Term Structure

Plot IV across expirations:
- **Contango** — front IV < back IV (normal, calm markets)
- **Backwardation** — front IV > back IV (stress; near-term fear)
- Backwardation in VIX futures is rare but historically a buy signal for equities

## The VIX

The CBOE Volatility Index — annualized 30-day implied volatility of SPX options, calculated from a portfolio of SPX puts and calls.

### VIX Levels Cheat Sheet

| VIX | Interpretation |
|-----|----------------|
| < 12 | Extreme complacency (rare) |
| 12–15 | Low vol regime |
| 15–20 | Normal |
| 20–30 | Elevated |
| 30–40 | Stress |
| > 40 | Crisis (2008, 2020, 2022) |
| > 60 | Panic peak |

### Key VIX Facts

- VIX cannot be traded directly — only via futures, options, ETPs
- VIX futures roll: most ETPs (UVXY, VXX) lose value over time due to contango decay
- Mean-reverts (high VIX → falls; low VIX → can stay low for years)
- Spikes are sharp; declines are slow

### VIX Futures Structure

The VIX futures curve is **usually in contango** (later months trade higher than spot).

```python
def vix_term_structure_state(vix_spot, vix_front, vix_second):
    """
    Returns shape of front of VIX curve.
    """
    if vix_front < vix_spot and vix_second < vix_front:
        return 'backwardation (stress)'
    if vix_front > vix_spot and vix_second > vix_front:
        return 'contango (calm)'
    return 'mixed'

def contango_roll_yield(front, second, days_to_roll):
    """
    Negative number = contango drag on long VIX ETPs.
    """
    return -(second - front) / front * (30 / days_to_roll)
```

### Long-Vol ETPs (VXX, UVXY)

- Hold rolling 30-day VIX futures
- Bleed in contango (~50%+ per year historically)
- Spike during crises (2020: VXX up 200%+)
- Use only as short-term tactical hedges, never long-term hold

### Short-Vol ETPs (SVXY, formerly XIV)

- Short rolling VIX futures
- Profit from contango
- 2017 went straight up; 2018-Feb wiped out XIV in one day (Volmageddon)
- Tail risk = bankruptcy

## Vol-Selling Strategies

The bread-and-butter of vol trading: collect the VRP.

### Short Strangle

- Sell out-of-the-money put + OTM call
- Profit if stock stays between strikes through expiration
- Defined-loss zone outside strikes; theoretical unlimited risk
- Sweet spot: 30–45 DTE, 16-delta strikes

```python
def strangle_pnl(call_premium, put_premium, stock_at_expiry,
                 call_strike, put_strike, contracts=1, multiplier=100):
    """P&L of a short strangle at expiration."""
    premium = (call_premium + put_premium) * contracts * multiplier
    call_loss = max(0, stock_at_expiry - call_strike) * contracts * multiplier
    put_loss = max(0, put_strike - stock_at_expiry) * contracts * multiplier
    return premium - call_loss - put_loss
```

### Iron Condor

- Sell OTM strangle, buy further-OTM strangle for protection
- Defined max loss; lower P&L than naked strangle
- Most retail-friendly vol selling structure

```
Long Put — Short Put — — — — — Short Call — Long Call
   $90        $95                $105         $110

Max profit: net credit collected
Max loss: wing width - net credit (per contract × 100)
```

### Iron Butterfly

- Sell ATM straddle, buy OTM wings
- Highest premium but narrow profit zone
- Best when expecting low movement around current price

### Calendar Spreads

- Sell front-month, buy back-month at same strike
- Profit from differential time decay + back-month vol
- Less directional than strangles
- Best at low IV environments expecting increase

### Diagonal Spreads

- Calendar + different strikes
- Combines directional view with vol view

## Vol-Buying Strategies

Less common (loses on average) but profitable when you're right about a coming move.

### Long Straddle / Strangle

- Buy ATM (or OTM) call + put
- Profit if move exceeds breakeven (premium paid)
- Use when: expecting big move, IV is cheap relative to expected catalysts
- Often a losing trade through routine earnings (see [[event_driven]])

### Backspread

- Sell 1 ATM, buy 2 OTM same direction
- Net credit/small debit; massive convexity if big move occurs
- Limited downside, big upside if move
- Useful for hedging or speculating on tail events

### Long VIX Calls

- Buy out-of-the-money VIX calls
- Cheap insurance for crash scenarios
- Most expire worthless; the one that pays can return 10–50x
- Tail-hedge sleeve sizing: 0.5–2% of portfolio per quarter

## The Vol Surface

Plot IV by strike (skew) and expiration (term structure). Volatility traders read the surface for relative-value opportunities.

### Common Surface Patterns

- **SPX**: steep downside skew, mild upside skew
- **Single names**: more symmetric "smile"
- **Commodities**: often upside skew (calls > puts) due to supply shocks
- **Currencies**: skew direction reflects expected central bank action

### Skew Trades

- **Skew steepener**: long downside skew via OTM puts vs. ATM
- **Risk reversals**: long call/short put or vice versa to express directional view in skew

## VIX Futures Trading

### Roll Yield Trade

- Short front-month VIX future, long back-month
- Profits from contango (curve flattens as front rolls toward spot)
- Risk: backwardation events (VIX spikes), front can rise dramatically faster than back
- Sizing matters; one bad spike can wipe out years of premium

### VIX Futures Spreads

- M1/M2, M2/M3 spreads
- Roll trades into expiration

## Realized Vol Strategies

### Variance Swaps (institutional)

- Pure exposure to realized variance
- Long var = receive realized − fixed strike × notional
- Used by pros to isolate VRP

### Volatility Targeting

- Adjust position size based on realized vol
- Higher vol → smaller positions; lower vol → larger
- Smooths equity curve; widely used in [[momentum_trend]] systems

```python
def vol_target_position(target_vol, current_vol, base_position):
    """Scale position so risk = target."""
    if current_vol == 0:
        return base_position
    return base_position * (target_vol / current_vol)
```

### Vol-Managed Portfolios

- Reduce equity exposure when realized vol is high
- Increase when low
- Moreira & Muir (2017) — improves Sharpe; works because of vol clustering

## Earnings Vol Trades

Detailed in [[event_driven]]. Summary:

- IV ramps into earnings, crushes after
- Selling strangles works on average (high win rate, occasional large losses)
- Buying premium usually loses unless you know something the market doesn't

```python
def earnings_iv_crush_estimate(pre_iv, expected_post_iv, days_to_earnings,
                                option_vega):
    """Rough P&L from IV crush on a vega-positive position."""
    iv_change = expected_post_iv - pre_iv  # negative for crush
    return option_vega * iv_change * 100  # vega is per 1 vol point
```

## Volatility Indicators

### VIX Term Structure (Contango/Backwardation)

```python
def vix_signal(spot_vix, m1_future, m2_future):
    """
    Simple risk-on/off signal from VIX curve.
    """
    if m1_future < spot_vix and m2_future < m1_future:
        return 'backwardation - stress/risk-off'
    if (m1_future - spot_vix) / spot_vix > 0.1:
        return 'steep contango - complacency'
    return 'normal'
```

### VVIX (Vol of VIX)

IV of VIX options. High VVIX = the market is pricing big VIX moves. Often signals tail risk.

### SKEW Index

CBOE index of OTM put pricing. High SKEW = expensive crash protection (often complacent equity market).

### Put/Call Ratio

- Equity put/call > 1.0 → bearish positioning (often contrarian buy)
- < 0.5 → bullish positioning (often contrarian sell)

### Implied Correlation

- IV of index vs. weighted average IV of constituents
- High correlation = expectation of market move; low = stock-picker's market

## Risk Management for Vol Sellers

Vol selling has **negative skew** — many small wins, occasional catastrophic losses. Without strict risk management, blowups are inevitable.

### Rules

1. **Always define max loss** — iron condors over naked strangles for retail
2. **Size for tail scenario** — assume vol spikes 3x overnight
3. **Diversify across underlyings** — single name blowups are common
4. **Avoid earnings/events** unless that's the trade
5. **Roll losers, don't hold to expiration on stressed positions**
6. **Hold dry powder** — opportunities expand after vol events
7. **Max 5–10% of portfolio in vol selling** at any time
8. **Hedge with long-vol tail** — buy cheap OTM puts/VIX calls as insurance

### Tail-Hedge Sizing

A common institutional approach (Universa, Spitznagel):
- Allocate 3–5% of portfolio to OTM put protection per year
- Other 95–97% in long-only equities
- Tail payoff in crashes more than offsets cost over time (debated empirically)

## Famous Blowups

- **LTCM (1998)** — leveraged short vol unwind on Russian default
- **Volmageddon (Feb 5, 2018)** — XIV (short vol ETF) lost 96% overnight; VIX up 116%
- **Long Capital (2018)** — multi-strategy fund killed by vol spike
- **Malachite Capital (2020)** — COVID vol explosion ended the fund
- **Many small option-selling traders** lost everything in March 2020

Lesson: short vol = picking up pennies in front of a steamroller. Survival requires hedges and discipline.

## Position Examples

### Conservative Vol Income (Iron Condor on SPX)

- 45 DTE
- 10-delta short strikes
- 10-point wings
- Roll at 21 DTE or 50% profit
- Expected: ~70% win rate, modest P&L per trade
- Position size: 5% of portfolio max

### Aggressive Premium Selling (Short Strangle on SPY)

- 30 DTE
- 16-delta strikes (~1 std dev)
- Undefined risk — must size accordingly
- Roll at 21 DTE
- ~80% win rate
- Catastrophic loss risk in vol spikes

### Tail Hedge (Long SPX Puts)

- 60–90 DTE
- 10–20% OTM
- 0.5–1% of portfolio per quarter
- Most expire worthless; one big payoff per cycle

### Volatility Carry Trade (Short VIX Futures)

- Short rolling front-month VIX futures
- Profits from contango
- Hedge with VIX call spreads to cap tail risk
- High Sharpe in calm regimes; risk of 2018-style wipeout

## Vol Trading vs. Directional Trading

| Trait | Directional | Volatility |
|-------|-------------|------------|
| What you're betting on | Price direction | Magnitude of moves |
| Time decay | Neutral to slight cost | Major P&L driver |
| Sensitivity to IV change | Low | Primary |
| Best regime | Trending markets | Range-bound (for sellers) |
| Skew of returns | Symmetric or slight positive | Strongly negative (sellers) |

## Combining Vol with Other Strategies

- **Vol selling + momentum** — both work in calm trending markets; diversify with crash hedge
- **Vol buying + earnings** — bet on specific catalysts where you have a view
- **Vol selling + mean reversion ([[pairs_mean_reversion]])** — both are negatively skewed; don't stack them naively
- **Long-vol overlay on equity portfolio** — buying tails reduces drawdown

## Backtesting Caveats

1. **Survivorship bias on ETPs** — XIV no longer exists; backtests must include it
2. **Look-ahead in vol calculation** — use t-1 vol, not t
3. **Bid-ask spreads matter** — wide on tail strikes
4. **Slippage in vol spikes** — markets become illiquid exactly when you need to exit
5. **Margin requirements change** — short vol margin can balloon overnight
6. **Earnings events embedded** — single-name vol backtests must handle these
7. **Index methodology changes** — VIX recalc methodology revised in 2014

## Resources

- **CBOE** — VIX, VVIX, SKEW, term structure data
- **vixcentral.com** — VIX futures curve, free
- **tastytrade** / **tastyworks** — vol trading platform + extensive education content
- **OptionMetrics** (paid) — historical option vol data
- **livevol** — analytics platform

### Books
- *Option Volatility & Pricing* — Sheldon Natenberg
- *Volatility Trading* — Euan Sinclair
- *Dynamic Hedging* — Nassim Taleb
- *The Volatility Smile* — Emanuel Derman
- *Trading Volatility* — Colin Bennett

### Papers
- Moreira & Muir (2017) — "Volatility-Managed Portfolios"
- Carr & Wu (2009) — "Variance Risk Premiums"
- Cheng (2019) — "VIX Premium"

## Key Takeaways

1. **Vol is an asset class** with its own risk premium (VRP)
2. **IV > RV on average** — selling vol is profitable but tail-risky
3. **VIX cannot be held directly** — ETPs decay in contango
4. **Term structure shape signals regime** — backwardation = stress
5. **Skew is permanent in equity indices** — driven by put demand
6. **Vol selling has negative skew** — many wins, occasional disasters
7. **Always define max loss** — iron condors > naked strangles
8. **Diversify tail hedges** — buy cheap OTM puts as insurance, especially in low-vol regimes
9. **Vol trading combines with [[event_driven]]** for earnings IV crush plays

## Where this connects

- [Options](options.md) — options are the primary instrument for volatility trading (IV vs RV)
- [Derivatives](derivatives.md) — VIX futures and variance swaps are volatility derivatives
- [Event driven](event_driven.md) — earnings IV crush is a volatility trading + event driven strategy

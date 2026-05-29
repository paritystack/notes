# Interest Rates and the Yield Curve

## Overview

Interest rates are the price of money — the rate of return on lending and the cost of borrowing. They drive nearly every asset price: bond values, equity multiples, currency exchange rates, mortgage payments, and corporate investment decisions. For a practical investor, mastering rates means understanding the yield curve, the Fed, duration, and the difference between real and nominal yields.

## Why Rates Matter for Every Asset

- **Bonds**: Prices move inversely to yields
- **Equities**: Higher discount rate compresses P/E multiples; growth stocks suffer most
- **Real estate** ([[reits]]): Cap rates track long rates; mortgage rates affect demand
- **Currencies**: Higher relative rates strengthen a currency
- **Commodities**: Higher real rates hurt gold (non-yielding) and pressure leveraged demand
- **Credit** ([[credit_markets]]): Spreads widen when rates rise sharply

The 10-year U.S. Treasury yield is the world's most important price — it discounts every long-duration cash flow on the planet.

## The Yield Curve

The yield curve plots Treasury yields across maturities: 1M, 3M, 6M, 1Y, 2Y, 5Y, 7Y, 10Y, 20Y, 30Y.

### Shapes

**Normal (upward-sloping)**
- Long rates > short rates
- Reflects term premium + expected growth
- Typical mid-cycle

**Flat**
- Long ≈ short
- Transitional; often signals late cycle

**Inverted (downward-sloping)**
- Short rates > long rates
- Market expects Fed to cut (recession coming)
- Has preceded every U.S. recession since 1955 (~12–18 month lead)

**Steep**
- Large gap long over short
- Typical early-cycle recovery; Fed easy, growth expectations rising

### Key Curve Spreads to Watch

| Spread | What it signals |
|--------|----------------|
| 10Y – 3M | Recession signal (Fed's preferred) |
| 10Y – 2Y | Recession signal (most cited in media) |
| 30Y – 5Y | Long-term inflation expectations |
| 5Y – 2Y | Mid-cycle steepness |
| 2Y – Fed funds | Market's view of Fed path |

```python
def curve_slope(yields):
    """Common yield curve spreads in basis points."""
    return {
        '10y_3m': (yields['10y'] - yields['3m']) * 100,
        '10y_2y': (yields['10y'] - yields['2y']) * 100,
        '30y_5y': (yields['30y'] - yields['5y']) * 100,
    }

def curve_regime(slope_10y_3m_bp):
    if slope_10y_3m_bp < -25: return 'deeply inverted (recession imminent)'
    if slope_10y_3m_bp < 0:   return 'inverted (warning)'
    if slope_10y_3m_bp < 50:  return 'flat (late cycle)'
    if slope_10y_3m_bp < 150: return 'normal (mid cycle)'
    return 'steep (early cycle / easing)'
```

### The Yield Curve in 3 Components

Yields can be decomposed as:

**Long yield = expected average short rate + term premium**

- **Expected short rates**: Where Fed funds is expected to be over the bond's life
- **Term premium**: Extra yield investors demand for locking in long duration (inflation risk, supply/demand)

When the curve inverts, it's often because expected short rates are higher near-term than long-term — the market is pricing rate cuts. Negative term premium (rare) means investors will accept lower yields to lock in duration.

## The Federal Reserve (Fed)

### Dual Mandate
1. **Maximum employment**
2. **Price stability** (2% PCE inflation target)

### Tools

**Fed funds rate**
- Overnight rate banks lend reserves to each other
- Target range (e.g., 5.25–5.50%)
- Set at FOMC meetings (8 per year)
- Transmits to all other rates

**Open Market Operations / Balance Sheet**
- Buying Treasuries/MBS = QE (Quantitative Easing) — adds reserves, lowers long rates
- Selling or letting bonds run off = QT — drains reserves, raises long rates

**Interest on Reserve Balances (IORB)**
- Rate Fed pays banks on reserves
- Floors the fed funds rate

**Reverse Repo (RRP)**
- Where money market funds park cash overnight
- Effective floor on short rates

**Forward Guidance**
- Communicating future policy intent
- Dot plot (SEP — Summary of Economic Projections)

**Discount Window / Standing Repo Facility**
- Emergency liquidity for banks

### FOMC Mechanics

- **8 meetings/year**: Statement + economic projections (quarterly: Mar/Jun/Sep/Dec)
- **Dot plot**: Each member's projected fed funds rate
- **Press conference**: Chair Q&A after each meeting
- **Minutes**: Released 3 weeks after the meeting
- **Beige Book**: 8 regional anecdotes released before each FOMC

### Reading the Fed

```python
def fed_stance(current_rate, neutral_rate=2.5, cpi_yoy=2.0, unemployment=4.0,
               nairu=4.0, inflation_target=2.0):
    """
    Estimate Fed's stance using a simple Taylor-rule-style heuristic.
    """
    inflation_gap = cpi_yoy - inflation_target
    unemployment_gap = nairu - unemployment  # positive = tight labor
    
    # Taylor rule: r = neutral + 1.5*inflation_gap + 0.5*output_gap
    suggested_rate = neutral_rate + 1.5 * inflation_gap + 0.5 * unemployment_gap
    
    delta = current_rate - suggested_rate
    if delta < -1.0: return 'very dovish (behind the curve)'
    if delta < -0.25: return 'dovish'
    if delta < 0.25: return 'neutral'
    if delta < 1.0: return 'hawkish'
    return 'very hawkish (restrictive)'
```

## Real vs. Nominal Rates

**Nominal rate** = stated yield (e.g., 10Y at 4.5%)

**Real rate** = nominal − inflation expectations (e.g., 4.5% − 2.3% = 2.2%)

Real rates are what matter for:
- **Investment decisions** (corporate hurdle rates)
- **Gold** (negatively correlated with real rates)
- **Equity multiples** (higher real rate = lower P/E)
- **Currency strength**

**TIPS** (Treasury Inflation-Protected Securities) yield = real rate directly. The TIPS-Treasury spread is the **breakeven inflation rate** — the market's expected inflation over that horizon.

```python
def real_rate(nominal_yield, expected_inflation):
    """Approximate real rate. Exact: (1+r) = (1+n)/(1+π)."""
    return nominal_yield - expected_inflation

def breakeven_inflation(nominal_yield, tips_yield):
    """Market-implied inflation expectation."""
    return nominal_yield - tips_yield
```

## Bond Duration and Convexity

### Duration

Duration measures price sensitivity to interest rate changes.

**Modified Duration** ≈ % price change per 1% yield change

A 10-year Treasury with modified duration of 8.5 falls ~8.5% if yields rise 1%.

```python
def modified_duration(price, yield_, cash_flows, times):
    """
    cash_flows: list of coupon + principal payments
    times: list of years to each payment
    """
    weighted_sum = sum(t * cf / (1 + yield_) ** t for t, cf in zip(times, cash_flows))
    macaulay = weighted_sum / price
    return macaulay / (1 + yield_)

def price_change_estimate(modified_dur, yield_change_pct):
    """Linear estimate of % price change."""
    return -modified_dur * yield_change_pct
```

### Convexity

Duration is a linear approximation; convexity captures the curvature. Longer bonds and lower coupons have higher convexity. Positive convexity is good — gains from falling yields exceed losses from rising yields by the same amount.

### Duration by Asset Class (approximate)

| Asset | Duration |
|-------|----------|
| Cash / T-bills | ~0 |
| 2Y Treasury | ~1.9 |
| 5Y Treasury | ~4.5 |
| 10Y Treasury | ~8.5 |
| 30Y Treasury | ~19 |
| Long-duration stocks (tech/growth) | ~15–25 (implied) |
| High-yield credit | ~3–5 (shorter due to coupons) |

## Quantitative Easing (QE) and Tightening (QT)

### QE
- Fed buys Treasuries and MBS, paying with newly created reserves
- Expands balance sheet
- Pushes down long-term yields directly + signals easy policy
- Used in 2008–14, 2020–22

### QT
- Fed lets bonds mature without reinvesting ("runoff")
- Drains reserves from the system
- Generally raises long yields modestly
- Risk: liquidity stress in funding markets (e.g., 2019 repo spike)

### Balance Sheet as a Policy Tool

```
Reserves on Fed balance sheet ↑ → easier financial conditions
                              ↓ → tighter financial conditions
```

Track via the Fed's **H.4.1** release (weekly) and Treasury **TGA** balance.

## Term Premium

The extra yield investors demand to lock in long-duration vs. rolling short-term:

**Drivers (positive term premium)**
- Inflation uncertainty
- Heavy Treasury issuance / large deficits
- Foreign buyers retreating
- Fed QT
- Geopolitical risk

**Drivers (negative term premium)**
- QE / strong central bank buying
- Pension/insurance liability matching demand
- Safe-haven flight to quality
- Deflation risk

The ACM and Kim-Wright term premium models are widely cited; both are published by the Fed.

## How Rates Drive Equity Valuations

Discounted cash flow logic:

```
P = Σ CF_t / (1 + r)^t
```

Where `r` = real risk-free rate + equity risk premium.

**Implications:**
- Higher real rates → lower equity values (especially long-duration cash flows)
- Growth stocks (cash flows far in the future) are most rate-sensitive
- 1% rise in 10Y real yield ≈ 10–15% drop in growth-stock multiples historically
- Value stocks (cash flows now) less affected
- Financials/banks benefit from steeper curves (borrow short, lend long)

```python
def equity_duration_estimate(forward_pe, dividend_growth_rate, payout_ratio=0.5):
    """
    Rough estimate of equity duration (years).
    High-multiple, low-payout names have higher duration.
    """
    return forward_pe * (1 - payout_ratio) / max(0.01, 1 - dividend_growth_rate)
```

## Practical Rate Trades

### Trading the Curve

**Steepener** — Long short-end, short long-end (profits if curve steepens)
- Setup: Fed about to cut while inflation expectations rising
- Vehicle: 2s10s futures spread, ETFs (STPP)

**Flattener** — Short short-end, long long-end
- Setup: Fed about to hike while long-term growth view weakens
- Vehicle: Bond futures spread

**Bull steepener** — Both ends rally, short end more (Fed cuts)
**Bear steepener** — Both sell off, long end more (inflation/issuance fear)
**Bull flattener** — Both rally, long end more (deflation fear)
**Bear flattener** — Both sell off, short end more (Fed hiking)

### Duration Bets

```python
def duration_pnl(notional, duration, yield_change_bp):
    """
    notional: dollar size
    duration: modified duration
    yield_change_bp: change in yield (basis points)
    Returns approximate dollar P&L (positive = profit on long).
    """
    return -notional * duration * (yield_change_bp / 10000)
```

Example: $1M long 10Y (dur ~8.5) when yields fall 25bp → +$1M × 8.5 × 0.0025 = **+$21,250**

### TIPS vs. Nominal

- Bullish inflation → long TIPS / short nominal
- Bearish inflation (or rising real rates) → opposite
- Vehicles: TIP, SCHP, individual TIPS, ETFs with breakeven exposure

## Global Rates Considerations

### Cross-Country Spreads

- **Bunds (Germany 10Y)** — Euro safe haven
- **JGBs (Japan 10Y)** — long held under yield curve control
- **Gilts (UK 10Y)** — post-Truss volatility
- **EM rates** — much higher; reflect currency/credit risk

### Currency Impact

Higher U.S. rates → stronger USD → headwind for:
- Emerging markets (USD debt service)
- Commodities (priced in USD)
- U.S. multinational earnings

### USD Funding Markets

- **SOFR** (Secured Overnight Financing Rate) — replaces LIBOR
- **Repo market** — overnight collateralized lending
- **Cross-currency basis swaps** — measure of USD scarcity globally

## Key Rates and Spreads to Monitor

| Rate | What it tells you |
|------|-------------------|
| Fed funds (target) | Policy stance |
| SOFR | Funding cost |
| 2Y Treasury | Market's Fed view |
| 10Y Treasury | Discount rate for everything |
| 30Y Treasury | Long-term inflation + term premium |
| 10Y TIPS | Real rate (the one for stocks/gold) |
| 10Y breakeven | Market inflation expectation |
| MOVE Index | Treasury volatility |
| HY OAS | Credit risk appetite |
| Mortgage rates (30Y FRM) | Housing transmission |

## Common Mistakes

1. **Confusing rates with rate changes** — what matters is the path vs. expectations
2. **Ignoring real rates** — nominal alone is misleading in different inflation regimes
3. **Forgetting duration** — a "safe" long bond loses 20%+ if rates rise 1%
4. **Fighting the Fed** — short the bond market into a cutting cycle and you'll lose
5. **Linear thinking on QE/QT** — flow effects vs. stock effects vary
6. **Yield curve fundamentalism** — inversion signals recession but timing is wide

## Resources

- **FRED** — all rates and spreads, free
- **U.S. Treasury Direct** — daily yield curve
- **NY Fed** — SOFR, repo data, term premium estimates
- **Fed H.4.1** — weekly balance sheet
- **CME FedWatch** — market-implied Fed path
- **BIS** — global rates statistics
- **MacroMicro** / **Trading Economics** — country comparisons

### Books
- *Inside the Yield Book* — Sidney Homer & Martin Leibowitz
- *Fixed Income Securities* — Bruce Tuckman
- *The Federal Reserve and the Financial Crisis* — Ben Bernanke
- *Lords of Finance* — Liaquat Ahamed

## Key Takeaways

1. **10Y Treasury is the world's most important price** — affects every asset
2. **Real rates, not nominal** — for stocks, gold, and FX
3. **Curve inversion is the most reliable recession signal** — but takes 12–18 months
4. **Duration is risk** — long bonds aren't safe in a rate-hiking cycle
5. **Don't fight the Fed** — policy regime determines what works
6. **Watch the path, not the level** — markets price expectations, not absolutes
7. **Term premium drives long end** — supply/demand, not just expected growth

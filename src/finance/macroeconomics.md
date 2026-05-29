# Macroeconomics for Investors

## Overview

Macroeconomics studies the economy as a whole — growth, inflation, employment, interest rates, and trade. For investors, macro is the regime: the same stock or bond behaves very differently in a recession vs. an expansion, in deflation vs. high inflation, or under a hawkish vs. dovish central bank. Understanding macro lets you tilt allocations, time risk-on/risk-off shifts, and avoid fighting the cycle.

## The Macro Investor's Mental Model

1. **Growth** (real GDP) — is the economy expanding or contracting?
2. **Inflation** (CPI/PCE) — is purchasing power eroding?
3. **Policy** (rates, fiscal) — is the Fed/government easing or tightening?
4. **Liquidity** (credit growth, money supply) — is capital cheap and abundant?
5. **Sentiment** (positioning, surveys) — are investors fearful or greedy?

These four levers (plus sentiment) explain most of the variation in asset returns across cycles. See [[interest_rates]] for the policy lever in depth, [[market_cycles]] for cycle phases, and [[credit_markets]] for the liquidity lens.

## Key Economic Indicators

### Growth Indicators

**GDP (Gross Domestic Product)**
- Total value of goods/services produced
- Quarterly release (advance, second, third estimates)
- Real GDP strips inflation; nominal GDP includes it
- Trend U.S. growth ~2%; recessions are typically two consecutive negative quarters (NBER actually defines them more broadly)

**Industrial Production**
- Monthly output of manufacturing, mining, utilities
- Cyclical sectors lead the broader economy

**Retail Sales**
- Consumer spending = ~70% of U.S. GDP
- Watch ex-autos and ex-gas for underlying trend

**PMI (Purchasing Managers' Index)**
- Above 50 = expansion, below 50 = contraction
- ISM Manufacturing and Services PMI
- Forward-looking via new orders sub-index

### Inflation Indicators

**CPI (Consumer Price Index)**
- Monthly basket of consumer prices
- Headline (all items) vs. Core (ex food & energy)
- Year-over-year (YoY) and month-over-month (MoM)
- Sticky vs. flexible CPI breakdown for trend signal

**PCE (Personal Consumption Expenditures)**
- Fed's preferred inflation gauge
- Broader basket weights that adjust to consumer substitution
- Target: 2% YoY core PCE

**PPI (Producer Price Index)**
- Wholesale prices; leads CPI by 1–3 months
- Goods PPI feeds into goods CPI

**Wage growth**
- Average hourly earnings (BLS), Atlanta Fed wage tracker, ECI (Employment Cost Index)
- 3.5%+ wage growth is consistent with 2%+ inflation given productivity

### Employment Indicators

**Non-Farm Payrolls (NFP)**
- First Friday of the month
- Jobs added/lost, unemployment rate, wage growth, participation rate
- 100k–150k/month sustains a stable rate; >250k is strong

**Initial Jobless Claims**
- Weekly; high-frequency leading indicator
- Spikes above 350k–400k signal weakening labor market

**JOLTS (Job Openings and Labor Turnover)**
- Openings, hires, quits
- Quits rate measures worker confidence

**Unemployment Rate (U-3)**
- Headline rate; trails the cycle
- U-6 adds underemployed and discouraged workers

### Sentiment & Forward Indicators

- **Consumer Confidence** (Conference Board, U. of Michigan)
- **NFIB Small Business Optimism**
- **LEI (Leading Economic Index)** — 10-component composite
- **Yield Curve** — see [[interest_rates]] for inversion signals

## Inflation Regimes

Inflation is the single biggest macro variable for asset allocation.

### Deflation (CPI < 0%)
- Cash and long-duration bonds win
- Equities suffer (Japan 1990s)
- Gold mixed (depends on real rates)
- Fed cuts aggressively, may use QE

### Disinflation (CPI 0–2%)
- Goldilocks for risk assets
- Long-duration assets (growth stocks, long bonds) outperform
- Most of 2010–2019 U.S. environment

### Moderate Inflation (CPI 2–4%)
- Nominal growth strong, equities do well
- Value, financials, real assets perform
- Fed at neutral

### High Inflation (CPI > 4%)
- Real assets win: commodities, gold, real estate
- Equities mixed — pricing-power businesses survive, long-duration suffers
- Bonds get crushed (2022)
- TIPS over nominal Treasuries

### Stagflation (high inflation + low/negative growth)
- Worst environment for 60/40 portfolios
- Gold, commodities, defensive value
- 1970s playbook

## Business Cycle Phases

```
Early ──→ Mid ──→ Late ──→ Recession ──→ Early
expansion      expansion   cycle                    expansion
```

| Phase | Growth | Inflation | Policy | Best Assets |
|-------|--------|-----------|--------|-------------|
| Early | Rising | Low | Easy | Small-caps, cyclicals, high-yield credit |
| Mid | Strong | Rising | Neutral | Equities broad, tech, industrials |
| Late | Slowing | High | Tight | Energy, materials, staples, cash building |
| Recession | Negative | Falling | Easing | Long Treasuries, gold, staples, healthcare |

See [[market_cycles]] for sector-rotation specifics.

## Recession Indicators

No single indicator is reliable; watch a cluster:

1. **Yield curve inversion** (10Y minus 3M or 10Y minus 2Y) — has preceded every U.S. recession since 1955, with ~12–18 month lead
2. **Sahm Rule** — recession signal when 3-month avg unemployment rises 0.5pp above its prior 12-month low
3. **LEI** — six consecutive monthly declines
4. **ISM Manufacturing < 45** — sustained
5. **Credit spreads widening** — see [[credit_markets]]
6. **Initial claims rising trend** — 4-week MA above 300k
7. **Real M2 contraction** — historically rare and bearish

```python
def yield_curve_inversion_signal(yield_10y, yield_3m):
    """Returns True if curve is inverted (recession warning)."""
    return yield_10y < yield_3m

def sahm_rule(unemployment_rate_3mo_avg, prior_12mo_low):
    """Triggers when 3-mo avg unemployment is 0.5pp above prior 12-mo low."""
    return (unemployment_rate_3mo_avg - prior_12mo_low) >= 0.5
```

## How Macro Drives Asset Classes

### Equities
- **Earnings** = nominal GDP proxy (long run)
- **Multiples** = inverse of real rates + risk premium
- Rising real rates compress P/Es, especially long-duration growth
- Falling rates + stable earnings = bull market

### Bonds
- Prices move inverse to rates
- Long duration most sensitive — see [[interest_rates]]
- Inflation is the enemy of nominal bonds; TIPS adjust principal to CPI

### Commodities
- Demand: cyclical with global growth (especially China)
- Supply: cycles take years
- Inflation hedge, especially energy and metals

### Currencies (USD)
- Higher U.S. rates relative to peers → stronger USD
- Risk-off → USD strength (safe haven)
- Strong USD = headwind for U.S. multinationals (~40% of S&P revenue is foreign)

### Real Estate / [[reits]]
- Sensitive to long rates (cap rates move with 10Y)
- Inflation can help (rent escalators) or hurt (cap rate expansion)

## Central Banks (Brief)

Central bank policy is the single biggest short-term driver of asset prices. See [[interest_rates]] for the full Fed mechanics.

**Major central banks to track:**
- **Federal Reserve (Fed)** — USD, dominant global
- **European Central Bank (ECB)** — EUR
- **Bank of Japan (BoJ)** — JPY, yield curve control
- **People's Bank of China (PBoC)** — CNY, credit impulse drives global growth
- **Bank of England (BoE)** — GBP

## Fiscal Policy

Government spending and taxation can be as impactful as monetary policy:

- **Deficit spending** stimulates demand (CARES Act 2020, IRA 2022)
- **Tax cuts** boost corporate earnings (TCJA 2017)
- **Debt issuance** absorbs liquidity (Treasury supply matters for bond yields)
- **Debt-to-GDP** trajectory affects long-term rates and currency

## Global Macro Linkages

- **U.S. dollar cycle** — strong USD hurts emerging markets (USD debt service)
- **China credit impulse** — leads global manufacturing PMIs by ~6 months
- **Oil prices** — supply shocks transmit to global inflation
- **Cross-border capital flows** — driven by relative growth + rate differentials

## Practical Macro Playbook for Investors

### Building a Macro Dashboard

Track these monthly:

| Indicator | Source | Frequency | Why |
|-----------|--------|-----------|-----|
| CPI (headline + core) | BLS | Monthly | Inflation regime |
| Core PCE | BEA | Monthly | Fed's target |
| NFP, unemployment | BLS | Monthly | Labor market |
| ISM Manufacturing & Services | ISM | Monthly | Forward growth |
| Yield curve (10Y–2Y, 10Y–3M) | Treasury | Daily | Recession signal |
| Initial jobless claims | DoL | Weekly | High-frequency labor |
| Fed funds rate + dot plot | Fed | FOMC | Policy stance |
| Credit spreads (HY OAS) | FRED | Daily | Risk appetite |
| LEI | Conference Board | Monthly | Composite leading |
| Retail sales | Census | Monthly | Consumer |

### Macro Decision Rules

```python
def macro_risk_on_score(indicators):
    """
    Simple macro risk-on/off score (0-10).
    Higher = more risk-on (favor equities, HY, EM).
    Lower = risk-off (favor cash, Treasuries, gold).
    """
    score = 5  # neutral
    
    # Growth
    if indicators['ism_mfg'] > 50: score += 1
    if indicators['ism_mfg'] > 55: score += 1
    if indicators['ism_mfg'] < 45: score -= 2
    
    # Inflation regime
    if 1.5 < indicators['core_pce_yoy'] < 3.0: score += 1
    if indicators['core_pce_yoy'] > 4.0: score -= 2
    
    # Policy
    if indicators['fed_stance'] == 'easing': score += 1
    if indicators['fed_stance'] == 'tightening': score -= 1
    
    # Curve
    if indicators['yield_curve_10y_3m'] < 0: score -= 2
    
    # Credit
    if indicators['hy_spread_bp'] < 400: score += 1
    if indicators['hy_spread_bp'] > 600: score -= 2
    
    return max(0, min(10, score))
```

### Allocation Tilts by Macro Regime

| Regime | Equities | Bonds | Commodities | Cash |
|--------|----------|-------|-------------|------|
| Reflation (growth↑ inflation↑) | OW cyclicals/value | UW long duration | OW | UW |
| Goldilocks (growth↑ inflation↓) | OW growth/tech | OW long duration | UW | UW |
| Stagflation (growth↓ inflation↑) | UW (def value) | UW nominal, OW TIPS | OW gold/energy | OW |
| Deflation (growth↓ inflation↓) | UW (def quality) | OW long Treasuries | UW | OW |

## Common Macro Mistakes

1. **Forecasting one variable in isolation** — macro is interconnected; inflation depends on growth, policy, expectations
2. **Mistaking timing for direction** — yield-curve inversions take 12–18 months to bite
3. **Overreacting to single data points** — trends matter; one NFP print isn't a regime change
4. **Ignoring the Fed** — "Don't fight the Fed" is cliché because it works
5. **Conflating recession with bear market** — markets bottom 4–6 months before recession ends
6. **Anchoring on the last cycle** — every cycle has different drivers

## Data Sources

- **FRED** (Federal Reserve Economic Data) — free, comprehensive U.S. + global
- **BEA** (Bureau of Economic Analysis) — GDP, PCE
- **BLS** (Bureau of Labor Statistics) — CPI, NFP, JOLTS
- **Census Bureau** — retail sales, housing
- **Federal Reserve** — H.4.1 balance sheet, H.6 money supply, SLOOS
- **Conference Board** — LEI, consumer confidence
- **ISM** — manufacturing and services PMI
- **Trading Economics** — global calendar
- **MacroTrends** — long historical charts

## Resources

### Books
- *The Big Picture* — Barry Ritholtz
- *Manias, Panics, and Crashes* — Kindleberger
- *Principles for Navigating Big Debt Crises* — Ray Dalio
- *The Lords of Easy Money* — Christopher Leonard
- *A History of Interest Rates* — Sidney Homer & Richard Sylla

### Newsletters / Sites
- The Macro Compass
- Wall Street Journal Daily Shot
- Bianco Research
- Apollo Sløk (free weekly chart pack)
- Federal Reserve research papers (NBER, FEDS)

## Key Takeaways

1. **Regime first, security selection second** — macro determines which asset classes work
2. **Cluster signals** — no single indicator forecasts the cycle reliably
3. **Real rates, not nominal** — what matters for equities and gold
4. **Watch the Fed** — policy is the dominant short-term driver
5. **Inflation is the master variable** — it determines whether bonds work or fail
6. **Yield curve > 90% of forecasters** — for U.S. recessions, historically
7. **Don't overtrade macro** — slow-moving; quarterly rebalancing is usually enough

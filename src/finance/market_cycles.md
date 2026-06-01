# Market Cycles and Sector Rotation

## Overview

Markets move in cycles. Bull and bear phases repeat, sector leadership rotates, and investor psychology swings between fear and greed. Understanding cycle phases helps you tilt allocation toward what's working, avoid late-cycle traps, and stay calm when others panic. This is the playbook for translating macro ([[macroeconomics]], [[interest_rates]]) into actionable positioning.

## Bull and Bear Markets

### Definitions

- **Bull market**: 20%+ rise from a low, no >20% drawdown
- **Bear market**: 20%+ decline from a high
- **Correction**: 10–20% decline
- **Pullback**: 5–10% decline
- **Secular bull/bear**: Multi-year, multi-cycle trend (e.g., 1982–2000 secular bull)

### Historical S&P 500 Cycles

| Period | Type | Length | Magnitude |
|--------|------|--------|-----------|
| 1929–32 | Bear | 33 mo | –86% |
| 1949–56 | Bull | 86 mo | +267% |
| 1973–74 | Bear | 21 mo | –48% |
| 1982–87 | Bull | 60 mo | +229% |
| 2000–02 | Bear | 31 mo | –49% |
| 2009–20 | Bull | 132 mo | +400%+ |
| 2022 | Bear | 9 mo | –25% |

**Average bull**: ~5 years, +150%. **Average bear**: ~14 months, –35%.

## The Business Cycle and Markets

### Phase Sequence

```
Early Recovery → Mid-Cycle → Late-Cycle → Recession → Early Recovery
```

Markets lead the economy by ~6 months: they bottom before recession ends and top before recession begins.

### Phase Characteristics

#### Early Recovery
- GDP turning positive, unemployment peaking
- Fed easy / cutting
- Inflation low
- Credit spreads narrowing
- **Equities**: explosive gains; small-caps, cyclicals lead
- **Bonds**: steep curve, long duration mixed
- **Credit**: high-yield rallies
- **Best sectors**: financials, consumer discretionary, industrials, small-caps

#### Mid-Cycle
- GDP solid, unemployment falling
- Fed at neutral
- Inflation rising but anchored
- Earnings broad-based growth
- **Equities**: grinding higher, lower volatility
- **Bonds**: range-bound
- **Best sectors**: technology, industrials, materials

#### Late-Cycle
- GDP slowing, unemployment near lows
- Fed tightening
- Inflation high, wage pressures
- Yield curve flattening/inverting
- Credit spreads widening
- **Equities**: narrowing leadership, high P/Es
- **Best sectors**: energy, materials, staples, healthcare, cash

#### Recession
- GDP contracting, unemployment rising
- Fed pivoting to ease
- Earnings collapse
- Credit spreads blow out
- **Equities**: bear market; defensives outperform
- **Bonds**: long Treasuries shine
- **Best sectors**: utilities, staples, healthcare, long Treasuries, gold

## Sector Rotation Model

Different sectors lead at different phases. Classic rotation:

```
Early ────────→ Mid ────────→ Late ────────→ Recession
Financials      Technology    Energy         Staples
Discretionary   Industrials   Materials      Utilities
Transports      Materials     Healthcare     Healthcare
Small-caps                    Staples        Long bonds
                                            Gold
```

### Sector Sensitivity Cheat Sheet

| Sector | Cyclical? | Rate-sensitive | Inflation hedge |
|--------|-----------|----------------|-----------------|
| Financials | Yes (mid-cycle) | Yes (loves steep curve) | Mixed |
| Technology | Yes (growth) | Yes (long duration) | No |
| Healthcare | No (defensive) | Low | Low |
| Consumer Discretionary | Yes | Moderate | No |
| Consumer Staples | No (defensive) | Low | Mild |
| Industrials | Yes | Moderate | Mild |
| Energy | Yes (cyclical) | Low | Yes |
| Materials | Yes | Moderate | Yes |
| Utilities | No (bond-proxy) | Yes (inverse) | No |
| Real Estate ([[reits]]) | Mixed | Yes (inverse) | Mixed |
| Communications | Mixed | Moderate | Low |

```python
def sector_tilt(phase):
    tilts = {
        'early_recovery': ['financials', 'discretionary', 'industrials', 'small_caps'],
        'mid_cycle':      ['technology', 'industrials', 'materials'],
        'late_cycle':     ['energy', 'materials', 'staples', 'healthcare', 'cash'],
        'recession':      ['utilities', 'staples', 'healthcare', 'long_treasuries', 'gold'],
    }
    return tilts.get(phase, [])
```

## Regime Detection

### Trend Filters

**Long-term trend**: Price vs. 200-day moving average
- Above 200dma → bull regime
- Below 200dma → bear regime (raise cash, defensive bias)

**Intermediate trend**: 50dma vs. 200dma
- 50 > 200 ("golden cross") → bullish
- 50 < 200 ("death cross") → bearish

```python
def trend_regime(price, sma_50, sma_200):
    if price > sma_200 and sma_50 > sma_200:
        return 'bull'
    if price < sma_200 and sma_50 < sma_200:
        return 'bear'
    return 'transition'
```

### Breadth Indicators

Broad participation = healthy bull. Narrow leadership = late cycle.

- **% of stocks above 200dma** — below 30% = oversold, above 80% = overbought
- **Advance/Decline line** — should rise with index in healthy bull
- **New highs vs. new lows** — divergence = warning
- **NYSE McClellan Oscillator** — short-term breadth momentum

### Volatility Regime

**VIX** as a fear gauge (see [[volatility_trading]]):
- VIX < 15: complacency, low-vol regime
- VIX 15–25: normal
- VIX 25–40: elevated stress
- VIX > 40: crisis (often near bottoms)

### Credit as a Risk Signal

- **HY OAS < 350bp**: risk-on
- **HY OAS 350–500bp**: normal
- **HY OAS > 500bp**: stress
- **HY OAS > 800bp**: recession/crisis

Credit usually leads equities at turns. See [[credit_markets]].

## Defensive vs. Cyclical Positioning

### Cyclical (offense)
- Discretionary, financials, industrials, materials, energy, tech
- Small-caps, value, EM
- High-yield credit, lower-quality
- Higher beta, higher returns in bull markets

### Defensive (defense)
- Staples, utilities, healthcare, telecoms
- Large-caps, quality, dividend aristocrats
- Treasuries, IG credit, cash, gold
- Lower beta, better drawdown profile

### Cyclical/Defensive Ratio

A useful tactical signal — the relative performance of XLY (consumer discretionary) vs. XLP (consumer staples), or industrials vs. utilities:

```python
def cyc_def_ratio(cyclical_etf_price, defensive_etf_price, lookback=50):
    """
    Rising ratio = risk-on (cyclicals leading)
    Falling = risk-off (defensives leading)
    """
    ratio = cyclical_etf_price / defensive_etf_price
    return ratio  # compare to its own moving average for signal
```

## Investor Psychology Cycle

Each cycle traces a recognizable arc of emotions:

```
Disbelief → Hope → Optimism → Belief → Thrill → Euphoria → Complacency
                                                            ↓
Despondency ← Capitulation ← Panic ← Fear ← Anxiety ← Denial
```

### Sentiment Indicators

- **AAII Investor Survey** — retail bull/bear; contrarian at extremes
- **Investors Intelligence** — newsletter writer sentiment
- **CNN Fear & Greed Index** — composite
- **Put/Call Ratio** — elevated puts = fear (often bullish contrarian)
- **Margin debt** — peaks late-cycle
- **IPO activity** — quality matters; junk IPOs at tops
- **Retail flows** — chase tops, panic-sell bottoms

```python
def sentiment_extreme(aaii_bulls_pct, aaii_bears_pct):
    """
    Bull-Bear spread. Extremes (>40 or <-20) often mark tops/bottoms.
    """
    spread = aaii_bulls_pct - aaii_bears_pct
    if spread > 40: return 'extreme greed (contrarian sell)'
    if spread < -20: return 'extreme fear (contrarian buy)'
    return 'normal'
```

## Secular vs. Cyclical Trends

A cyclical bull/bear lasts months to a few years. A **secular** trend spans decades.

### Identified U.S. secular periods

| Period | Type | Driver |
|--------|------|--------|
| 1949–66 | Secular bull | Post-war boom |
| 1966–82 | Secular bear | Stagflation |
| 1982–2000 | Secular bull | Disinflation, productivity |
| 2000–13 | Secular bear (sideways) | Dot-com bust, GFC |
| 2013–present | Secular bull? | Tech, low rates (debatable post-2022) |

Secular regimes determine what asset classes/sectors structurally lead. The 1970s favored commodities; the 1980s–90s favored equities; the 2010s favored U.S. growth/tech.

## Cycle Timing Tools (Quantitative)

### Drawdown-Based Allocation

```python
def drawdown_allocation(current_drawdown_pct, max_equity=0.7, min_equity=0.3):
    """
    Increase equity exposure as drawdowns deepen (rebalancing into weakness).
    """
    if current_drawdown_pct > -10:
        return max_equity * 0.8  # near highs, slight underweight
    if current_drawdown_pct > -20:
        return max_equity  # full target
    if current_drawdown_pct > -30:
        return min(max_equity * 1.15, 1.0)  # add
    return min(max_equity * 1.3, 1.0)  # crisis = add aggressively
```

### Risk-On/Off Score (composite)

```python
def risk_score(macro, breadth, sentiment, credit):
    """
    Composite 0-10 risk-on score combining cycle signals.
    """
    score = 5
    
    # Macro
    if macro['ism'] > 50: score += 1
    if macro['curve_inverted']: score -= 2
    
    # Breadth
    if breadth['pct_above_200dma'] > 60: score += 1
    if breadth['pct_above_200dma'] < 30: score -= 1  # oversold, will rebound
    
    # Sentiment (contrarian)
    if sentiment['aaii_bull_bear'] < -20: score += 1  # extreme fear = buy
    if sentiment['aaii_bull_bear'] > 40: score -= 1
    
    # Credit
    if credit['hy_oas_bp'] < 400: score += 1
    if credit['hy_oas_bp'] > 600: score -= 2
    
    return max(0, min(10, score))
```

## Bear Market Playbook

### Recognizing a Bear Market Early

- Yield curve inverted 12+ months ago
- Fed in tightening cycle
- Breadth deteriorating (narrowing leadership, fewer new highs)
- Credit spreads widening
- Trend break below 200dma with rising volume

### Defensive Moves

1. **Reduce equity beta** — shift to staples, utilities, healthcare
2. **Increase quality** — lower debt, higher ROE, dividend payers
3. **Raise cash** — opportunity ammo for bottoms
4. **Add Treasuries** — long duration as deflationary hedge (works only when inflation is contained)
5. **Add gold** — crisis hedge
6. **Hedge with puts** — see [[options]]
7. **Trim leverage** — cut margin, deleverage

### Bear Market Rallies

Bears include vicious 15–25% counter-trend rallies. Don't chase. Wait for:
- New 52-week highs in major indices
- Sustained breadth thrust (90% up days)
- Fed pivot confirmed
- Credit spreads compressing

### Buying the Bottom

You won't pick it. Instead:
- Scale in via tranches as drawdown deepens
- Buy after capitulation signs (volume climax, VIX > 40, sentiment crash)
- Watch for divergences (breadth bottoms before price)

## Bull Market Playbook

### Early Bull
- Maximum equity exposure
- Cyclicals, small-caps, value over growth/defensive
- Use leverage cautiously
- Add high-yield credit

### Mid Bull
- Trend follow
- Quality + growth
- Trim losers, ride winners
- Add to international/EM if outperforming

### Late Bull / Topping
- Watch for narrowing leadership
- Raise quality, trim speculatives
- Take profits in extended winners
- Hedge with puts; build cash gradually
- Don't try to short the top

## Common Cycle Mistakes

1. **Fighting the trend** — shorting strong markets, buying weak ones early
2. **Top-ticking** — trying to call the exact peak
3. **Capitulating at bottoms** — panic-selling into max fear
4. **Single-indicator dependence** — no signal works alone
5. **Anchoring to the last cycle** — every cycle has unique drivers
6. **Ignoring breadth** — index can rise on 5 stocks; that's not a bull market
7. **Forgetting policy regime** — sector rotation is conditional on Fed and macro
8. **Confusing cyclical with secular** — short-term moves vs. multi-decade trends

## Practical Workflow

### Monthly Check-In

1. Where are we in the business cycle? (review [[macroeconomics]] dashboard)
2. What's the trend regime? (200dma, breadth)
3. What's credit doing? (HY OAS trend)
4. What's volatility regime? (VIX percentile)
5. Sentiment extreme? (AAII, F&G)
6. Sector leadership consistent with cycle phase?

### Quarterly Rebalance

- Realign to target allocation
- Tilt sectors based on phase
- Tax-loss harvest losers ([[tax_strategies]])
- Trim winners that exceed bands

### Yearly Review

- Update cycle thesis
- Review secular drivers
- Adjust strategic allocation if regime has changed

## Resources

- **Investech Research**, **Stockcharts.com** — sector relative-strength tools
- **Sentimentrader** — extensive sentiment data
- **Conference Board LEI** — leading indicator composite
- **Yardeni Research** — sector earnings/valuation tables
- **Fred** — historical recessions
- **Ned Davis Research** — cycle research (institutional)

### Books
- *Mastering the Market Cycle* — Howard Marks
- *Stocks for the Long Run* — Jeremy Siegel
- *Devil Take the Hindmost* — Edward Chancellor (manias)
- *Anatomy of the Bear* — Russell Napier
- *Big Debt Crises* — Ray Dalio

## Key Takeaways

1. **Cycle phase drives sector leadership** — don't fight rotation
2. **Markets lead the economy by ~6 months** — bottoms come before "data" turns
3. **Trend filters work** — 200dma keeps you out of disasters
4. **Breadth matters as much as price** — narrowing leadership warns
5. **Credit leads equities at turns** — watch HY spreads
6. **Sentiment is contrarian at extremes** — buy fear, sell euphoria
7. **Cycles repeat; details differ** — pattern recognition, not prediction
8. **Defensive in late-cycle, aggressive after capitulation** — discipline beats forecasting

## Where this connects

- [Macroeconomics](macroeconomics.md) — the business cycle is the macro foundation of market cycles
- [Momentum trend](momentum_trend.md) — trend following exploits multi-quarter market cycles
- [Portfolio management](portfolio_management.md) — sector rotation and risk-on/risk-off shifts asset allocation
- [Behavioral finance](behavioral_finance.md) — investor psychology drives cycle amplitude at peaks and troughs

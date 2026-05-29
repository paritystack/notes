# Event-Driven Trading

## Overview

Event-driven strategies trade around scheduled or surprise corporate events: earnings releases, mergers, FDA approvals, index reconstitutions, spin-offs, dividends, and management changes. Each event creates a predictable price/volatility pattern that traders can monetize — often with limited market correlation. These trades are tactical, time-bounded, and lean on edges from setup mechanics rather than directional macro bets.

## Major Event Categories

| Event | Pattern | Typical Edge |
|-------|---------|--------------|
| Earnings | Pre-announce IV ramp, post-announce IV crush | Vol selling, gap fades |
| M&A | Target trades below offer price | Spread capture |
| Index inclusion | Forced buying drives price | Front-running, post-fade |
| Spin-off | Parent + spinoff often mispriced | Long spinoff, ignored small caps |
| FDA approval / clinical trial | Binary biotech jumps | Pre-event positioning, post-event mean reversion |
| Dividend announcement | Ex-div price drop = dividend | Tax/ownership arbitrage |
| Buyback / Issuance | Persistent supply/demand | Slow drift |
| Activist 13D filing | Stock pops 5–15% | Quick reaction trade |
| Bankruptcy / Restructuring | Capital structure dislocations | Distressed credit |

## Earnings Plays

The most accessible event-driven strategy for retail.

### Pre-Earnings Setup

- IV (implied volatility) rises into earnings as demand for options ramps
- Stock often drifts upward 1–2 weeks before (positive earnings drift)
- Options become expensive

### Post-Earnings

- IV collapses ("IV crush") within minutes of release — often 30–50% IV reduction
- Stock moves to new equilibrium quickly (most of the move in first 60 seconds)
- Drift continues for days/weeks (Post-Earnings Announcement Drift — PEAD)

### Common Earnings Trade Structures

**Long premium (long straddle / strangle)**
- Buy ATM call + ATM put before earnings
- Profits if move > breakeven (premium paid)
- Risk: usually you OVERPAY for premium given IV; IV crush slaughters small moves

**Short premium (short straddle / iron condor)**
- Sell ATM straddle (collect premium)
- Profits if move < premium collected
- Risk: large moves cause unbounded loss (uncapped) or large defined loss (condor)
- Win rate ~60–70% historically but big losers offset

**Calendar spread**
- Sell front-month (high IV, will crush), buy back-month (less crush)
- Profits from differential IV crush
- Lower P&L, lower risk than naked

**PEAD (Post-Earnings Drift)**
- Buy stocks that beat earnings by >5% AND raise guidance, hold 30–90 days
- Short stocks that miss and guide down
- Edge: slow institutional repositioning
- Best on small-mid caps; large-caps largely arbitraged

```python
def earnings_surprise_pct(actual_eps, consensus_eps):
    """Earnings surprise as % of consensus."""
    if consensus_eps == 0:
        return None
    return (actual_eps - consensus_eps) / abs(consensus_eps) * 100

def pead_signal(surprise_pct, guidance_change, threshold=5.0):
    """Simple PEAD signal."""
    if surprise_pct > threshold and guidance_change > 0:
        return 'long'
    if surprise_pct < -threshold and guidance_change < 0:
        return 'short'
    return 'pass'

def expected_move(stock_price, atm_straddle_price):
    """Options market's implied move into earnings."""
    return atm_straddle_price / stock_price  # as a percentage
```

### Earnings Trade Checklist

- [ ] Liquid options (tight bid/ask, OI > 500 on ATM strikes)
- [ ] Check expected move vs. recent realized
- [ ] Confirm announcement timing (BMO vs. AMC)
- [ ] Beware of pre-announcements
- [ ] Avoid biotech/spec names with binary outcomes unless that's the trade
- [ ] Define max loss before entry
- [ ] Plan to close before/at IV crush — don't hold long premium overnight unless big convex bet
- [ ] Watch for guidance, conference call surprises post-print

## Merger Arbitrage (Risk Arb)

Trade the spread between target's current price and the deal offer.

### Cash Deal Mechanics

- Acquirer offers $50/share cash
- Target trades at $48.50 (3% below offer)
- Buy target → if deal closes at $50, capture 3% over 3–6 months
- Annualized: ~6–10%
- Risk: deal breaks → target falls to standalone price (often $35), big loss

### Stock Deal

- Acquirer offers 0.6 shares of acquirer per target share
- Need to short acquirer to hedge stock-component risk
- Spread = (0.6 × Acquirer price) − Target price

```python
def merger_arb_spread(target_price, offer_price):
    """Gross spread + annualized."""
    spread = (offer_price - target_price) / target_price
    return spread * 100  # in percent

def annualized_arb_return(spread_pct, days_to_close):
    return spread_pct * (365 / days_to_close)

def deal_break_loss(target_price, standalone_estimate):
    return (standalone_estimate - target_price) / target_price * 100
```

### Spread Tells

| Spread behavior | Likely cause |
|-----------------|--------------|
| Tightens steadily | Confidence growing, closing soon |
| Widens with no news | Regulatory/financing concern |
| Blows out > 10% | Major regulatory hurdle or financing risk |
| Trades through deal | Bidding war expected (rare) |

### Common Deal Risks

- **Antitrust** (FTC/DOJ, EU Commission, China SAMR)
- **Shareholder vote rejection**
- **Financing fails** (rare in modern strategics; common in PE LBOs at extremes)
- **Material Adverse Change (MAC) clause** invoked
- **Hostile competing bid**
- **Regulatory in foreign jurisdiction** (CFIUS for national security)

### Position Sizing

- Max 5% of portfolio per deal
- Diversify across 10–20 deals
- Avoid leverage; one broken deal can ruin a year
- Use stop on widening: if spread doubles from entry, reassess

## Index Inclusion / Reconstitution

When a stock is added to a major index (S&P 500, Russell 1000), index funds must buy. Forced buying creates predictable price moves.

### Pattern

- **Pre-announcement**: speculation often pushes price up
- **Announcement → effective date** (usually 1–4 weeks): stock rallies 5–15%
- **Day of inclusion**: most index funds buy at the close (MOC orders)
- **Post-inclusion**: stock often gives back some of the gains over weeks

### Trade ideas

- Long candidates before official announcement (informed bet)
- Long announced additions through effective date
- Short the post-inclusion fade (1–4 weeks later)
- Spread trade: long additions, short deletions

### Key indices

- **S&P 500** — discretionary committee; announcements ~1 week ahead
- **Russell 1000/2000** — formulaic; "rebalancing" in late June, candidates known
- **NASDAQ-100** — annual reconstitution in December
- **MSCI/FTSE** — quarterly rebalances

## Spin-Offs

A parent company distributes shares of a subsidiary to its shareholders. Newly listed entity has no analyst coverage, gets dumped by parent shareholders who didn't want it.

### Why they're mispriced

- **Forced selling**: index funds, parent shareholders dump
- **Small size**: institutional desks ignore until liquidity grows
- **No coverage**: research silence = inefficient pricing
- **Often undervalued**: management has incentive to set conservative initial guidance

### Joel Greenblatt's playbook (You Can Be a Stock Market Genius)

- Buy spin-offs 1–4 weeks after distribution
- Hold 12–24 months
- Documented multi-decade outperformance vs. market

### Risk

- Some spinoffs are "garbage truck" — parent dumping unwanted business
- Examine balance sheet, free cash flow, management incentives
- Watch for over-leveraged spin-offs

## FDA / Clinical Trial Events

Biotech and pharma stocks make binary moves on:

- **FDA PDUFA dates** (approval decisions)
- **Phase 2/3 trial readouts**
- **AdCom (Advisory Committee) meetings**
- **NDA/BLA filings**

### Trade structures

- **Long stock pre-event**: leveraged binary bet, often expensive given IV
- **Long calls only**: defined risk, profit on approval pop
- **Long strangles**: bet on large move either direction (IV expensive but moves can be 50–80%)
- **Short premium after event**: harvest the IV crush

### Reality check

- Phase 3 success rates: ~50% historical
- FDA approval after positive Phase 3: ~85%
- Don't average down on negative readouts — gaps don't fill
- Position size small (< 1% of capital per binary bet)

## Buybacks and Issuance

### Buybacks

- Companies announcing large buybacks ($1B+) often outperform 12-month forward
- Edge larger when buyback funded by cash (vs. debt-funded)
- ETF: PKW (Buyback Achievers)

### Secondary offerings

- Issuing more shares is dilutive; stock typically drops 3–10% on announcement
- Sometimes a buying opportunity if dilution is one-time and funds clear catalysts (capex, acquisitions)
- Avoid serial issuers (junior miners, biotech with no revenue)

## Activist Investing (Following 13D Filings)

Investors crossing 5% ownership must file a 13D within 10 days. Activist filings often pop the stock 5–15%.

### Trade

- Subscribe to 13D filing alerts
- Buy on filing day (if missed, often a fade opportunity in first hour)
- Hold 6–24 months as activist pushes for changes
- Edge: activists target undervalued companies and force change

### Key activists to track

- Elliott Management
- Trian Fund Management (Peltz)
- Starboard Value
- ValueAct Capital
- Engaged Capital
- Carl Icahn

## Dividend Events

### Ex-Dividend Date

- Stock drops by ~ dividend amount at open on ex-date
- Tax treatment (qualified vs. ordinary) determines if "harvesting" makes sense
- Beware: most dividend capture doesn't work after taxes & frictions

### Special Dividends

- Large one-time distributions ($1+ per share) require options adjustments
- Sometimes a buying signal (cash-rich management returning capital)

### Dividend Increases

- Companies raising dividends consistently outperform
- ETFs: NOBL (Dividend Aristocrats), VIG (Dividend Appreciation)

## Bankruptcy and Restructuring (Brief)

Distressed event-driven is a specialty (Oaktree, Apollo, Elliott), but retail-friendly slices:

- **Pre-bankruptcy equity**: usually goes to zero — avoid
- **Distressed bonds**: senior secured may recover 60–90 cents on the dollar
- **Post-reorg equity**: new shares often trade cheap initially; institutional under-ownership
- **Risk arbitrage in restructuring** (Chapter 11 plan confirmation trades)

## Earnings IV Crush Math

```python
def iv_crush_pnl(option_price_pre, option_price_post,
                 underlying_move_pct, expected_move_pct,
                 contracts=1, multiplier=100):
    """
    Approximate P&L on a long straddle through earnings.
    Pre-earnings option price reflects expected_move IV.
    Post-earnings reflects realized move minus IV reset.
    """
    return (option_price_post - option_price_pre) * contracts * multiplier

def short_straddle_pnl(premium_collected, underlying_move, stock_price,
                        strike, contracts=1, multiplier=100):
    """
    P&L on a short straddle. Loss if move > premium collected.
    """
    intrinsic = max(0, underlying_move - 0)  # simplified
    return (premium_collected - max(abs(underlying_move - 0) - 0, 0)) * contracts * multiplier
```

## Combining Events with Other Strategies

- **Earnings + momentum**: trade PEAD on stocks already in momentum uptrends
- **Merger arb + vol selling**: collect option premium on closing deals
- **Spin-offs + value**: spin-offs that screen cheap on FCF
- **Index inclusion + technicals**: confirm with breakout patterns

## Risk Management for Event Trades

1. **Position-size for binary outcomes** — assume worst case
2. **Avoid concentration in one event type** — diversify across categories
3. **Hard exits at event resolution** — don't hold "in case it bounces"
4. **Track realized vs. expected moves** — your edge depends on accurate odds
5. **Beware of news leaks** — pre-event drift can mean others know more
6. **Tax-aware**: ST gains on event trades = ordinary income; consider [[tax_strategies]]
7. **Liquidity check** — exit may be hard if event surprises against you

## Tracking Events

### Data sources

- **Earnings**: Yahoo Earnings Calendar, EarningsWhispers, Bloomberg
- **M&A**: SEC EDGAR (13D, 14A, 8-K), DealReporter, Reuters
- **FDA**: BioPharmCatalyst, FDA.gov, ClinicalTrials.gov
- **Index changes**: S&P announcements page, FTSE Russell
- **Activist**: WhaleWisdom, 13D Monitor, Activist Insight
- **Buybacks/issuance**: Press releases, 8-K filings

### Workflow

- Build weekly event calendar
- Pre-position 1–3 days ahead (or fade post-event)
- Maintain database of historical event-move statistics
- Post-mortem each trade: was the edge what you thought?

## Common Mistakes

1. **Buying long options into earnings** — IV crush usually beats the move
2. **Concentrating on a single deal** — one broken merger can wipe quarters of arb
3. **Holding losers in biotech** — binary events don't average down
4. **Ignoring borrow** — short legs of stock-deal arb can have brutal borrow costs
5. **Trading on rumors** — usually priced in by the time you hear
6. **Forgetting tax implications** — short-term gains dominate ([[tax_strategies]])
7. **Over-trading** — events generate friction; pick spots
8. **Following Twitter activist tips** — usually after the run

## Resources

### Books
- *You Can Be a Stock Market Genius* — Joel Greenblatt (spinoffs, special situations)
- *Merger Masters* — Kate Welling, Mario Gabelli
- *Risk Arbitrage* — Guy Wyser-Pratte
- *Distressed Debt Analysis* — Stephen Moyer

### Sites
- **EarningsWhispers** — whisper numbers and reaction stats
- **BioPharmCatalyst** — biotech event calendars
- **WhaleWisdom** — 13F/13D tracking
- **MergerArbitrageLimited** — deal spread tracking
- **SECEdgar full-text search**

## Key Takeaways

1. **Earnings IV crush is the most accessible retail event trade**
2. **Selling premium beats buying** through earnings, on average
3. **Merger arb is steady income** with rare large drawdowns; diversify
4. **Index inclusion creates forced flows** — predictable for a few days
5. **Spin-offs are persistently mispriced** — patience pays
6. **Binary events (FDA) need tiny position sizes**
7. **Activist filings (13D) trigger short-term pops** — fade or follow
8. **Document expected vs. realized** for every event trade — your edge requires calibration

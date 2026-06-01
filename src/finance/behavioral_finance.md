# Behavioral Finance

## Overview

Classical finance assumes rational actors who maximize expected utility and price assets efficiently. Real investors are loss-averse, overconfident, herd-prone, and easily framed. Behavioral finance studies the systematic ways human psychology deviates from that ideal — and how those deviations create the mispricings and anomalies that strategy pages elsewhere try to harvest. This is the "why" underneath sentiment extremes ([[market_cycles]]), the momentum and value premia ([[momentum_trend]], [[valuation]]), and most self-inflicted portfolio damage ([[risk_management]], [[portfolio_management]]). Knowing the biases doesn't immunize you — it just lets you build process that does.

## Efficient Markets vs. Reality

The Efficient Market Hypothesis (EMH) says prices reflect available information, so you can't consistently beat the market without taking more risk.

| Form | Claim | Implication |
|------|-------|-------------|
| **Weak** | Prices reflect all past prices | Technical analysis ([[technical_analysis]]) shouldn't work |
| **Semi-strong** | Prices reflect all public info | Fundamental analysis ([[fundamental_analysis]]) shouldn't add alpha |
| **Strong** | Prices reflect all info incl. private | Even insiders can't profit |

EMH is a useful baseline, but reality is "efficiently inefficient": markets are *mostly* efficient, yet persistent anomalies survive because arbitrage has limits.

### Limits to Arbitrage

Smart money can't always correct mispricings:

- **Noise-trader risk** — irrational prices can get *more* irrational before reverting (Keynes: "markets can stay irrational longer than you can stay solvent")
- **Costs & constraints** — shorting is expensive, hard to borrow, or capped
- **Career/funding risk** — a manager who fades a bubble early gets redeemed before being proven right
- **Fundamental risk** — the hedge is imperfect, so the "arb" carries real exposure

```
EMH ideal:          Mispricing → arbitrage → instant correction
Reality:            Mispricing → limited arb → slow, noisy reversion (or none)
```

## Prospect Theory

Kahneman & Tversky's prospect theory replaces expected-utility with how people *actually* weigh gains and losses. Three core findings:

1. **Reference dependence** — value is felt relative to a reference point (usually purchase price or recent high), not absolute wealth
2. **Loss aversion** — losses hurt ~2x as much as equivalent gains feel good
3. **Diminishing sensitivity** — the jump from \$0→\$100 feels bigger than \$1,000→\$1,100; the curve is concave in gains, convex in losses

```
        value
          │            ........  (concave: risk-averse in gains)
          │        ....
          │     ...
   ───────┼──●────────────────  outcome
       ..│   reference point
     .  │
   .    │   (convex + steeper: risk-seeking to avoid losses,
  .     │    and steeper slope = loss aversion)
```

The steeper loss side explains why investors hold losers (gambling to get back to break-even) and sell winners early (locking in a sure gain) — the **disposition effect**.

### Probability Weighting

People overweight small probabilities and underweight near-certainties — why lottery tickets and far-OTM options ([[options]]) sell, and why tail insurance feels overpriced.

## Cognitive Biases

Two broad families: biases that distort **beliefs** (what you think is true) and biases that distort **actions** (what you do about it).

### Belief Biases

| Bias | What happens | Market damage |
|------|--------------|---------------|
| **Overconfidence** | Overestimate skill/precision | Overtrading, undersized risk buffers, concentrated bets |
| **Confirmation** | Seek info that agrees with thesis | Ignore disconfirming data; thesis drift |
| **Anchoring** | Fixate on a reference number | "It was \$100, so \$60 is cheap" regardless of fundamentals |
| **Hindsight** | "I knew it all along" | Overlearn from luck; false confidence in forecasts |
| **Recency / availability** | Overweight recent, vivid events | Chase last year's winner; fight the last war |
| **Narrative fallacy** | Prefer a good story to base rates | Buy compelling stories at any price |

### Action Biases

| Bias | What happens | Market damage |
|------|--------------|---------------|
| **Loss aversion** | Losses hurt 2x gains | Paralysis; refusing to realize losses |
| **Disposition effect** | Sell winners, hold losers | Caps upside, compounds downside, tax-inefficient ([[tax_strategies]]) |
| **Herding** | Follow the crowd | Bubbles and crashes; buying tops, selling bottoms |
| **FOMO** | Fear of missing out | Chase parabolic moves; enter at maximum risk |
| **Sunk cost** | Honor past spend | Average down into a broken thesis |
| **Mental accounting** | Treat money by bucket | "House money" recklessness; ignoring fungibility |
| **Home bias** | Overweight the familiar | Under-diversified ([[portfolio_management]]); concentrated in employer stock |
| **Anchoring to cost basis** | Decisions tied to entry price | The market doesn't know or care what you paid |

## Market Anomalies

Persistent return patterns EMH struggles to explain — many are behavioral in origin and underpin entire strategy pages.

| Anomaly | Pattern | Behavioral driver | See |
|---------|---------|-------------------|-----|
| **Momentum** | Recent winners keep winning (3–12 mo) | Underreaction then herding | [[momentum_trend]] |
| **Value** | Cheap (low P/B, P/E) beats expensive long-run | Overextrapolation of growth | [[valuation]] |
| **Size** | Small-caps outperform (risk-adjusted, noisy) | Neglect, liquidity premium | [[stocks]] |
| **Post-earnings drift** | Prices drift in surprise direction for weeks | Underreaction to news | [[event_driven]] |
| **Calendar effects** | Jan small-cap pop, "sell in May," turn-of-month | Tax-loss selling, flows | [[tax_strategies]] |
| **Low-volatility** | Low-beta stocks beat risk-adjusted | Lottery preference for high-beta | [[risk_management]] |
| **IPO underperformance** | New issues lag long-run | Overoptimism, hype timing | [[stocks]] |

The catch: anomalies decay once published and arbitraged, and many vanish after costs. Treat them as behavioral tendencies, not guarantees.

## Sentiment as a Contrarian Signal

Crowd psychology becomes a tradable signal at extremes — buy fear, sell euphoria. See the detailed indicator list and the emotion-cycle arc in [[market_cycles]] ("Investor Psychology Cycle" and "Sentiment Indicators"); the short version:

- **AAII bull/bear spread** — retail survey; extremes are contrarian
- **Put/call ratio** — high = fear (often a bullish setup) ([[options]])
- **CNN Fear & Greed** — composite gauge
- **Margin debt, IPO/junk issuance, retail flows** — peak late-cycle euphoria

```
Crowd emotion:   Euphoria ──────► (sell zone, max risk)
                    ▲                     │
              Complacency            Capitulation
                    │                     ▼
                 Optimism ◄────── Despondency (buy zone, max opportunity)
```

The contrarian edge is real but dangerous: extremes can get more extreme, so pair sentiment with trend/breadth confirmation rather than fading strength blindly.

## Debiasing and Process

You can't delete biases, but a rules-based process turns the System 1 (fast, emotional) decision into a System 2 (slow, deliberate) one.

- **Written investment policy** — entry/exit/sizing rules decided when calm, not mid-drawdown
- **Pre-commitment** — automatic rebalancing, dollar-cost averaging, stop rules ([[financial_planning]])
- **Pre-mortem** — "it's a year later and this blew up — why?" surfaces ignored risks
- **Decision journal** — record thesis + confidence at entry; review to separate skill from luck and fight hindsight
- **Base rates first** — anchor on historical odds before the seductive story
- **Checklists** — force disconfirming questions before sizing up
- **Separate the analyst from the trader** — review positions as if you didn't own them; ignore cost basis

```python
def disposition_check(position):
    """Flag the disposition effect: clinging to losers, dumping winners.
    Decisions should rest on forward thesis, not entry price."""
    pnl_pct = (position['price'] - position['cost_basis']) / position['cost_basis']
    if pnl_pct < -0.10 and position['thesis_intact'] is False:
        return "SELL candidate: down >10% AND thesis broken — don't anchor to cost"
    if pnl_pct > 0.20 and position['thesis_intact'] is True:
        return "HOLD candidate: winner with intact thesis — don't sell to feel good"
    return "no flag"


def rules_gate(trade, plan):
    """Pre-commitment gate: a trade must clear the written plan, not the mood."""
    checks = {
        'in_universe':   trade['symbol'] in plan['universe'],
        'size_ok':       trade['risk_pct'] <= plan['max_risk_per_trade'],
        'has_exit':      trade.get('stop') is not None,
        'not_fomo':      not trade.get('chasing_parabolic', False),
    }
    failed = [k for k, ok in checks.items() if not ok]
    return "APPROVED" if not failed else f"BLOCKED: {failed}"
```

## Resources

- **Behavioral economics** — Kahneman & Tversky's original prospect-theory papers
- **AAII Sentiment Survey**, **CNN Fear & Greed Index** — live crowd-sentiment gauges
- **AQR research library** — academic-grade work on factor/anomaly persistence
- **Morningstar "Mind the Gap"** — annual study quantifying the behavior gap (investor returns vs. fund returns)

### Books
- *Thinking, Fast and Slow* — Daniel Kahneman (System 1/2, biases)
- *Misbehaving* — Richard Thaler (the field's history)
- *The Psychology of Money* — Morgan Housel (behavior over math)
- *Your Money and Your Brain* — Jason Zweig (neuroeconomics, practical)
- *Fooled by Randomness* — Nassim Taleb (luck vs. skill)

## Key Takeaways

1. **Markets are mostly efficient, but not perfectly** — limits to arbitrage let anomalies persist
2. **Losses hurt ~2x as much as gains feel good** — loss aversion drives most bad decisions
3. **The disposition effect is the costliest habit** — selling winners, holding losers, anchored to cost basis
4. **Biases come in two flavors** — distorted beliefs (overconfidence, confirmation, anchoring) and distorted actions (herding, FOMO, sunk cost)
5. **Behavioral patterns underpin real premia** — momentum, value, post-earnings drift trace to under/overreaction ([[momentum_trend]], [[valuation]], [[event_driven]])
6. **Sentiment is contrarian at extremes** — but extremes can deepen; confirm with trend
7. **You can't unlearn biases — you can out-process them** — written rules, pre-commitment, journals, checklists
8. **The behavior gap is real money** — investors underperform their own funds by mistiming; discipline is the edge

## Where this connects

- [Market cycles](market_cycles.md) — behavioral biases amplify cycle peaks and troughs
- [Portfolio management](portfolio_management.md) — behavioral finance explains why investors underperform their own funds
- [Momentum trend](momentum_trend.md) — momentum partly exploits investors' underreaction to information

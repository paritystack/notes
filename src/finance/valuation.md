# Company Valuation

## Overview

Valuation is the process of estimating what a company, asset, or security is worth. It's both art and science: rigorous frameworks discipline thinking, but assumptions about growth, margins, and discount rates dominate outputs. The best valuators triangulate multiple methods, document assumptions, and reverse-engineer market expectations rather than producing single-point estimates. Valuation underpins fundamental investing ([[fundamental_analysis]]), M&A, LBOs ([[private_markets]]), IPOs, and credit analysis ([[credit_markets]]).

## The Core Principle

A company's value equals the present value of the cash it will generate for owners over its remaining life. Every valuation method is an approximation of this:

```
Value = Σ (Expected Future Cash Flow_t / (1 + Discount Rate)^t)
```

The challenge is forecasting cash flows and choosing the right discount rate. Different methods package these inputs differently.

## Method 1: Discounted Cash Flow (DCF)

The most rigorous valuation approach. Projects free cash flows explicitly, discounts at WACC, adds a terminal value.

### Steps

1. **Project Free Cash Flow** for 5–10 years
2. **Compute Terminal Value** (perpetuity or exit multiple)
3. **Discount** everything at WACC
4. **Sum** to Enterprise Value
5. **Bridge** to Equity Value (subtract net debt, add cash)
6. **Divide** by shares outstanding for per-share intrinsic value

### Free Cash Flow

```
Unlevered FCF (to firm) = EBIT × (1 − Tax) + D&A − CapEx − ΔNWC

Levered FCF (to equity) = Net Income + D&A − CapEx − ΔNWC − Net Debt Repayment
```

Most DCFs use unlevered FCF + WACC, then bridge to equity.

```python
def fcf_to_firm(ebit, tax_rate, da, capex, change_in_nwc):
    """Unlevered free cash flow."""
    return ebit * (1 - tax_rate) + da - capex - change_in_nwc

def fcf_to_equity(net_income, da, capex, change_in_nwc, net_debt_repayment=0):
    """Levered free cash flow."""
    return net_income + da - capex - change_in_nwc - net_debt_repayment
```

### Discount Rate (WACC)

The weighted average cost of capital — blended required return across debt and equity.

```
WACC = (E/V × Re) + (D/V × Rd × (1 − Tc))
```

- E = market value of equity
- D = market value of debt
- V = E + D
- Re = cost of equity
- Rd = cost of debt (pre-tax)
- Tc = corporate tax rate

```python
def wacc(equity_value, debt_value, cost_of_equity, pretax_cost_of_debt,
         tax_rate):
    total = equity_value + debt_value
    e_weight = equity_value / total
    d_weight = debt_value / total
    return e_weight * cost_of_equity + d_weight * pretax_cost_of_debt * (1 - tax_rate)
```

### Cost of Equity (CAPM)

```
Re = Rf + β × (Rm − Rf)
```

- Rf = risk-free rate (10Y Treasury typically; see [[interest_rates]])
- β = beta vs. market
- (Rm − Rf) = equity risk premium (typically 4.5–6%)

```python
def cost_of_equity_capm(risk_free_rate, beta, equity_risk_premium):
    return risk_free_rate + beta * equity_risk_premium
```

For small / non-public companies, add a size premium (1–4%) and company-specific risk premium.

### Terminal Value

Often 60–80% of total DCF value. Two common methods:

**Perpetuity (Gordon) Growth**:
```
TV = FCF_final × (1 + g) / (WACC − g)
```

Where g = long-run growth rate (typically 2–3%, capped at GDP growth).

**Exit Multiple**:
```
TV = EBITDA_final × Exit Multiple
```

Use comparable transaction multiples or industry averages.

Cross-check: implied perpetuity-growth from an exit multiple, and vice versa. If they diverge wildly, revisit assumptions.

```python
def terminal_value_gordon(fcf_final, growth_rate, wacc):
    if growth_rate >= wacc:
        raise ValueError("Growth rate must be less than WACC")
    return fcf_final * (1 + growth_rate) / (wacc - growth_rate)

def terminal_value_multiple(ebitda_final, exit_multiple):
    return ebitda_final * exit_multiple

def dcf(fcfs, terminal_value, wacc):
    """
    fcfs: list of projected free cash flows, years 1..N
    terminal_value: value at end of year N
    Returns Enterprise Value.
    """
    n = len(fcfs)
    pv_fcfs = sum(fcf / (1 + wacc) ** (t + 1) for t, fcf in enumerate(fcfs))
    pv_terminal = terminal_value / (1 + wacc) ** n
    return pv_fcfs + pv_terminal

def equity_value(enterprise_value, total_debt, cash, minority_interest=0):
    return enterprise_value - total_debt + cash - minority_interest

def share_price(equity_value, diluted_shares):
    return equity_value / diluted_shares
```

### Complete DCF Example

```python
# Inputs
projection_years = 5
revenue_y0 = 1000
revenue_growth = [0.10, 0.08, 0.07, 0.05, 0.04]
ebit_margin = 0.20
tax_rate = 0.22
da_pct_revenue = 0.05
capex_pct_revenue = 0.06
nwc_pct_revenue = 0.03

wacc_value = 0.09
terminal_growth = 0.025
total_debt = 500
cash = 200
diluted_shares = 100

# Project FCFs
rev = revenue_y0
prev_nwc = revenue_y0 * nwc_pct_revenue
fcfs = []
for g in revenue_growth:
    rev = rev * (1 + g)
    ebit = rev * ebit_margin
    da = rev * da_pct_revenue
    capex = rev * capex_pct_revenue
    nwc = rev * nwc_pct_revenue
    change_nwc = nwc - prev_nwc
    prev_nwc = nwc
    fcf = ebit * (1 - tax_rate) + da - capex - change_nwc
    fcfs.append(fcf)

tv = terminal_value_gordon(fcfs[-1], terminal_growth, wacc_value)
ev = dcf(fcfs, tv, wacc_value)
eq = equity_value(ev, total_debt, cash)
price = share_price(eq, diluted_shares)
```

### Sensitivity and Scenario Analysis

Always present DCF as a range, not a point. Build a sensitivity table varying WACC ± 1% and terminal growth ± 0.5%. The range shows how much your assumptions are driving the answer.

```python
def sensitivity_table(base_fcfs, base_wacc, base_growth, wacc_range, growth_range,
                       debt, cash, shares):
    table = {}
    for w in wacc_range:
        for g in growth_range:
            tv = terminal_value_gordon(base_fcfs[-1], g, w)
            ev = dcf(base_fcfs, tv, w)
            eq = equity_value(ev, debt, cash)
            table[(w, g)] = eq / shares
    return table
```

### Reverse DCF

Instead of computing fair value, ask: "What growth and margin assumptions does the current market price imply?" If the implied assumptions are unrealistic, the stock is mispriced.

```python
def implied_growth(current_price, current_fcf, wacc, years=10):
    """Solves for terminal growth rate that justifies current price."""
    # Simplified — iterate to find g
    from scipy.optimize import brentq
    def f(g):
        return dcf([current_fcf * (1 + g) ** t for t in range(1, years+1)],
                   terminal_value_gordon(current_fcf * (1 + g) ** years, g, wacc),
                   wacc) - current_price
    try:
        return brentq(f, -0.05, wacc - 0.001)
    except ValueError:
        return None
```

### Pros and Cons

**Pros:**
- Intrinsic value based on fundamentals
- Forces explicit assumptions
- Independent of market sentiment

**Cons:**
- Garbage-in, garbage-out
- Terminal value usually dominates
- Highly sensitive to WACC and growth
- Useless for early-stage, unprofitable, or cyclical companies without significant adjustment

## Method 2: Relative Valuation (Multiples)

Compare valuation ratios against peers, sector averages, or historical ranges. Fast, market-aware, but inherits any market mispricing.

### Common Multiples

| Multiple | Formula | Best For |
|----------|---------|----------|
| P/E | Price / EPS | Mature, profitable companies |
| Forward P/E | Price / Forward EPS | Growth companies, removes one-time items |
| PEG | P/E / Growth Rate | Cross-compare growers |
| P/B | Price / Book Value | Banks, insurance, asset-heavy |
| P/S | Price / Sales | Unprofitable companies, early stage |
| EV/EBITDA | EV / EBITDA | Cross-capital-structure comparison |
| EV/EBIT | EV / EBIT | Lower-capex companies |
| EV/Sales | EV / Revenue | Pre-profit, high-growth |
| EV/FCF | EV / Free Cash Flow | Most rigorous; harder to game |
| P/FCF | Price / FCF per share | Quality-focused investors |
| P/AFFO | Price / AFFO | REITs (see [[reits]]) |
| Dividend Yield | Annual Div / Price | Income stocks |
| EV/Reserves | EV / Proved reserves | Oil & gas |
| EV/Subscribers | EV / users | Telecom, streaming |

### Enterprise Value vs. Equity Value

**Equity multiples** (P/E, P/B): use market cap as numerator.
**EV multiples** (EV/EBITDA, EV/Sales): use enterprise value.

```
EV = Market Cap + Total Debt − Cash + Minority Interest + Preferred Equity
```

EV multiples are capital-structure-neutral — comparing leveraged and unleveraged companies fairly.

### PEG Ratio

```
PEG = P/E ÷ Growth Rate (in %)
```

Peter Lynch popularized: PEG < 1 = potentially undervalued. PEG > 2 = expensive. Caveats: growth rate assumption matters; doesn't work for negative or cyclical earnings.

### Sector-Specific Multiples

Different industries use different yardsticks because economic models differ.

| Industry | Primary Multiples |
|----------|-------------------|
| Software / SaaS | EV/Revenue, EV/ARR, Rule of 40 |
| Banks | P/B, P/TBV, P/E |
| Insurance | P/B, P/E |
| REITs | P/FFO, P/AFFO, NAV ([[reits]]) |
| Oil & Gas | EV/EBITDA, EV/Reserves, EV/Production |
| Mining | EV/Resource, NAV |
| Telecom | EV/EBITDA, EV/Subscriber |
| Tobacco / staples | Dividend Yield, P/E |
| Tech growth | EV/Revenue, P/Sales, EV/Gross Profit |
| Biotech (pre-revenue) | EV/Pipeline NPV |
| Banks / Insurance | ROE × P/B relationship |
| Hotels / REITs | Cap Rate, NAV |
| Airlines / Cruise | EV/EBITDAR (rent-adjusted) |

### Multiple Selection Process

1. Define comparable set (industry, size, growth, geography, business model)
2. Calculate multiples for each comp; remove outliers
3. Compute average / median / range
4. Apply to target company's metric
5. Cross-check against multiple metrics

```python
def relative_valuation(target_metric, comp_multiples, percentile=50):
    """
    target_metric: e.g., target company's EBITDA
    comp_multiples: list of comparable EV/EBITDA values
    Returns implied valuation at chosen percentile.
    """
    import numpy as np
    comp_multiples = sorted(comp_multiples)
    multiple = np.percentile(comp_multiples, percentile)
    return target_metric * multiple
```

### Pros and Cons

**Pros:**
- Fast, market-current
- Easy to communicate
- Captures market sentiment about sector

**Cons:**
- Inherits market mispricing (bubble → bubble valuations)
- Comparability is rarely clean
- Doesn't reveal underlying drivers
- Can mask quality differences

## Method 3: Asset-Based Valuation

Value = sum of net assets. Best for asset-heavy, liquidation, or holding companies.

### Variants

**Book Value** (accounting basis)
```
Book Value = Total Assets − Total Liabilities − Preferred Equity
```

Often understates economic value (depreciation creates artificially low asset book).

**Liquidation Value**
- What assets would fetch in a forced/orderly sale
- Subtract liquidation costs (auction fees, severance)
- Floor value in distressed situations

**Replacement Cost**
- Cost to recreate the company's assets from scratch
- Useful for capital-intensive industries

**Adjusted (Tangible) Book Value**
- Strip out goodwill, intangibles
- For banks: tangible common equity (TCE) is key

```python
def book_value_per_share(total_equity, preferred_equity, diluted_shares):
    return (total_equity - preferred_equity) / diluted_shares

def tangible_book_value(total_equity, goodwill, intangibles, preferred_equity=0):
    return total_equity - goodwill - intangibles - preferred_equity
```

### When to Use

- Banks (P/TBV anchored)
- Real estate / REITs (NAV)
- Holding companies (Berkshire, sum-of-parts)
- Distressed / liquidation scenarios
- Asset-heavy industrials (paper, mining, shipping)

### When NOT to Use

- Software, brand-driven, IP-heavy businesses (intangibles are the value)
- Subscription / network-effect businesses
- High-growth tech

## Method 4: Sum-of-the-Parts (SOTP)

Value each business segment separately, sum them, subtract corporate overhead and debt. Useful for conglomerates and holding companies.

### Steps

1. Identify distinct segments
2. Project segment-level financials
3. Apply appropriate valuation method per segment (DCF, multiples, NAV)
4. Sum segment values
5. Subtract conglomerate discount (5–25% historically)
6. Subtract net corporate debt

```python
def sotp_valuation(segments, net_debt, conglomerate_discount=0.10):
    """
    segments: list of dicts with 'name', 'value'
    Returns implied equity value.
    """
    gross = sum(s['value'] for s in segments)
    discounted = gross * (1 - conglomerate_discount)
    return discounted - net_debt
```

Examples where SOTP useful: Alphabet (search, cloud, YouTube, Other Bets), Berkshire (insurance, railroads, energy, equities), Disney (parks, streaming, media networks).

## Method 5: Comparable Company Analysis (Comps)

A subset of relative valuation focused on publicly-traded peers.

### Selection Criteria

- Same industry / sub-industry
- Similar size (revenue, market cap)
- Similar growth profile
- Similar margins / capital intensity
- Same geography (if relevant)

### Process

1. Build comp set (5–15 companies)
2. Calculate multiples for each
3. Adjust for one-time items
4. Compute median / mean / range
5. Apply to target

### Common Adjustments

- Strip one-time gains / losses
- Normalize tax rates
- Adjust for accounting differences (operating leases pre-ASC 842, R&D capitalization)
- Pro-forma for recent M&A
- Currency translation

## Method 6: Precedent Transactions

Value based on prices paid in recent M&A deals for similar companies. Includes a control premium.

### Considerations

- **Recency**: deals > 3 years old often stale
- **Strategic vs. financial**: strategics pay synergy premium; PE buyers ([[private_markets]]) pay near pure financial value
- **Control premium**: typically 20–40% over standalone share price
- **Distressed deals**: bid in stressed market is not fair comparison

```python
def precedent_implied_value(target_metric, transaction_multiples):
    """transaction_multiples: list of EV/EBITDA from comparable deals."""
    import numpy as np
    return target_metric * np.median(transaction_multiples)
```

## Method 7: LBO Analysis

How much could a private equity buyer pay and still hit target IRR? Often used as a floor valuation in M&A. Covered in detail in [[private_markets]].

### Logic

1. Assume exit in 5 years at certain multiple
2. Assume target debt structure (typically 5–7x EBITDA leverage)
3. Project FCF and debt paydown
4. Solve for entry price that gives ~20–25% sponsor IRR

```python
def lbo_max_entry_ev(exit_ev, equity_check, debt_at_entry, debt_at_exit,
                     hold_years=5, target_irr=0.22):
    """
    Solve for max entry EV given target IRR.
    Assumes sponsor wants target_irr on equity.
    """
    target_exit_equity = equity_check * (1 + target_irr) ** hold_years
    exit_equity_available = exit_ev - debt_at_exit
    # If exit equity > target, sponsor can afford more debt or pay more
    # Returns sanity check
    return {
        'target_exit_equity': target_exit_equity,
        'actual_exit_equity': exit_equity_available,
        'enough_return': exit_equity_available >= target_exit_equity,
    }
```

## Method 8: Dividend Discount Model (DDM)

For dividend-paying stocks (utilities, REITs, mature dividend payers).

### Gordon Growth Model

```
P = D_1 / (r − g)
```

Where:
- D_1 = next year's dividend
- r = required return (cost of equity)
- g = perpetual dividend growth rate

```python
def ddm_gordon(next_dividend, required_return, growth_rate):
    if growth_rate >= required_return:
        raise ValueError("Growth must be less than required return")
    return next_dividend / (required_return - growth_rate)
```

### Two-Stage DDM

Higher growth for N years, then perpetual lower growth. Better for companies in transition.

### Three-Stage DDM

High growth → transition → mature. Used for early-stage dividend payers.

### When DDM Works

- Mature, stable dividend payers
- Predictable payout ratios
- Cost of equity > expected dividend growth

### When DDM Fails

- Non-dividend payers
- Dividends cut / suspended
- Cyclical earnings

## Method 9: Residual Income Model

Value = book value + PV of future "excess returns" above cost of capital.

```
Value = Book Value + Σ (ROE - Cost of Equity) × Book Value_t / (1 + r)^t
```

Useful for banks and capital-intensive companies where book value is meaningful.

## Method 10: Real Options Valuation

Treats strategic flexibility (option to expand, defer, abandon) as a financial option. Useful for:
- R&D-heavy companies
- Mining / energy reserves
- Early-stage businesses
- Phased capital projects

Uses Black-Scholes-like math (see [[options]] and [[derivatives]]). Rarely used in pure equity research but common in M&A and corporate strategy.

## Industry-Specific Valuation Notes

### Technology / Software

- **Rule of 40**: revenue growth + EBITDA margin > 40 = healthy SaaS
- **EV/ARR**: enterprise value to annual recurring revenue
- **LTV/CAC**: lifetime value to customer acquisition cost
- **Net Revenue Retention (NRR)**: > 110% = strong expansion
- DCF challenging: project growth deceleration carefully

### Banks and Financial Institutions

- **P/B ratio** primary
- **ROE × P/B relationship**: if ROE > cost of equity, P/B > 1 justified
- **Tangible book value** more conservative
- **Net Interest Margin (NIM)** trend
- **Loan loss provisioning**, charge-offs
- Stress-test capital under adverse scenarios

```python
def justified_price_to_book(roe, cost_of_equity, growth_rate):
    """Sustainable P/B for a bank."""
    if cost_of_equity == growth_rate:
        return float('inf')
    return (roe - growth_rate) / (cost_of_equity - growth_rate)
```

### Insurance

- **P/B + ROE**: similar to banks
- **Combined ratio**: < 100 means underwriting profit
- **Investment portfolio yield**
- **Embedded value** for life insurers

### REITs

- See [[reits]] for full treatment
- P/FFO, P/AFFO, NAV
- Cap rates implied vs. private market
- Sector-specific dynamics

### Energy / Commodities

- Oil & gas: **EV/Reserves**, **EV/EBITDAX**, **NAV of reserves**
- Mining: **EV/Resource**, **NPV of mine plan**
- Both: commodity-price-deck sensitivity is dominant
- See [[commodities]]

### Cyclicals (Steel, Chemicals, Autos)

- Normalize earnings over a full cycle
- Trough vs. peak EBITDA matters
- **EV/EBITDA on mid-cycle estimates**
- **P/B near cyclical troughs**

### Early-Stage / Pre-Revenue

- TAM × market share × multiple
- Pipeline NPV (biotech)
- VC funding round implied valuations (see [[private_markets]])
- Comparable startup valuations

## Cost of Capital in Different Environments

WACC isn't static. It moves with:
- **Risk-free rate** ([[interest_rates]])
- **Equity risk premium** (varies by market regime, [[market_cycles]])
- **Beta** (company-specific risk)
- **Credit spreads** ([[credit_markets]])

In low-rate environments (2010s), DCFs produced very high valuations for long-duration growth. In 2022's rate spike, those same companies' DCFs collapsed — pure cost-of-capital effect.

### Discount Rate Cheat Sheet

| Type | Typical Range |
|------|---------------|
| Risk-free (10Y Treasury) | 4–5% (current) |
| Investment-grade WACC | 6–10% |
| High-yield / EM WACC | 10–15% |
| Venture-stage WACC | 25–60% (or scenario-based) |
| Private equity hurdle rate | 8% (with 20% carry beyond) |

## Margin of Safety

Buy below your estimate of intrinsic value to absorb estimation error.

```
Margin of Safety = (Intrinsic Value − Market Price) / Intrinsic Value
```

Benjamin Graham: target 30%+ margin of safety. Howard Marks: "Investment success doesn't come from buying good things, but from buying things well."

```python
def margin_of_safety(intrinsic_value, market_price):
    return (intrinsic_value - market_price) / intrinsic_value
```

## Reverse-Engineering Market Expectations

Instead of asking "what is this worth?" ask "what does the current price imply?" Then judge whether implied expectations are realistic.

### Example

If a stock trades at 25× P/E and has 8% cost of equity:
- Implied long-term growth ≈ 8% − 4% (1/PE) = 4%? (rough Gordon)
- Realistic for the industry? If yes, fair-valued; if not, mispriced.

This technique (championed by Damodaran) cuts through the trap of building speculative DCFs that confirm a desired conclusion.

## Quick Valuation Heuristics

| Rule | Use |
|------|-----|
| Rule of 72 | Years to double = 72 / annual return % |
| PEG < 1 | Potentially cheap (caveats) |
| P/B < 1 with positive earnings | Often value/distressed signal |
| EV/EBITDA < 8 | Cheap for stable business; sector-dependent |
| Dividend yield > 10Y Treasury + 2% | Income opportunity (check sustainability) |
| FCF yield > 8% | Worth investigating |
| EBITDA growth > 2× revenue growth | Margin expansion |
| Net debt / EBITDA < 2× | Conservative leverage |

## Common Mistakes

1. **False precision** — "Worth $47.23" implies certainty that doesn't exist; use ranges
2. **Garbage in, garbage out** — DCF only as good as inputs
3. **Terminal value dominance** — 60–80% of value in TV is fragile
4. **Wrong cost of capital** — using book vs. market weights, ignoring credit risk
5. **Cherry-picking comps** — selecting peers to confirm desired valuation
6. **Forgetting one-time items** — depreciation accelerations, restructuring charges, gains
7. **Ignoring qualitative factors** — moat, management, customer concentration
8. **Linear extrapolation** — assuming current growth continues forever
9. **Mismatched discount rate and cash flow** — nominal CF, real WACC, etc.
10. **EV ↔ Equity confusion** — using one method's denominator with another's numerator
11. **Operating leases** — pre-ASC 842, often missed off-balance-sheet
12. **Stock-based comp as non-cash** — it IS a real cost; add back at peril
13. **Synergies in M&A DCFs** — over-credit to seller
14. **Single-point valuations** — without sensitivity, you don't really know
15. **Anchoring to current price** — circular reasoning

## Best Practices

1. **Triangulate with multiple methods** — DCF + comps + precedents
2. **Build sensitivity tables** — vary 2–3 key inputs
3. **Run scenarios** — base / bull / bear with explicit assumptions
4. **Stress-test the moat** — what if competition takes 10% margin?
5. **Conservative assumptions** by default
6. **Document your model** — assumptions, sources, dates
7. **Reverse-DCF the market price** before forming your own view
8. **Update for new information** — quarterly earnings, guidance changes
9. **Match discount rate to cash flow** — same currency, same nominal/real
10. **Calibrate against past forecasts** — were you systematically too bullish?

## Scenario / Probability-Weighted Valuation

For binary or wide-outcome situations:

```python
def probability_weighted_value(scenarios):
    """
    scenarios: list of dicts with 'probability' and 'value'.
    """
    assert abs(sum(s['probability'] for s in scenarios) - 1.0) < 1e-6
    return sum(s['probability'] * s['value'] for s in scenarios)

# Example: pharma stock with binary FDA outcome
scenarios = [
    {'name': 'approval', 'probability': 0.60, 'value': 80},
    {'name': 'CRL/delay', 'probability': 0.30, 'value': 35},
    {'name': 'rejection', 'probability': 0.10, 'value': 15},
]
# Expected value: 0.6*80 + 0.3*35 + 0.1*15 = $60.00
```

Useful for biotech, M&A targets ([[event_driven]]), and distressed situations.

## Tools and Data Sources

### Free
- **SEC EDGAR** — 10-K, 10-Q, proxy filings
- **Yahoo Finance** — basic data, multiples
- **Macrotrends** — long historical financial data
- **Damodaran's website** (NYU Stern) — datasets, country risk premiums, sector betas, valuation papers
- **GuruFocus** (some free) — historical multiples, GF Score
- **Stockanalysis.com** — clean financial statement display

### Paid / Professional
- **Bloomberg Terminal** — gold standard
- **FactSet, Capital IQ, Refinitiv** — institutional
- **Tikr, Finchat, Stratosphere** — affordable retail Bloomberg alternatives
- **Koyfin** — accessible institutional-style data
- **Morningstar** — Morningstar Equity Research reports

## Resources

### Books
- *Valuation: Measuring and Managing the Value of Companies* — McKinsey (Koller, Goedhart, Wessels) — the standard text
- *The Little Book of Valuation* — Aswath Damodaran
- *Damodaran on Valuation* — Aswath Damodaran
- *Investment Valuation* — Damodaran (encyclopedic)
- *Security Analysis* — Benjamin Graham, David Dodd (foundational)
- *The Intelligent Investor* — Benjamin Graham
- *Common Stocks and Uncommon Profits* — Philip Fisher
- *The Outsiders* — William Thorndike (capital allocation case studies)
- *Investment Banking* — Joshua Rosenbaum, Joshua Pearl (comps, precedents, LBO)
- *Margin of Safety* — Seth Klarman (out of print, but legendary)

### Online
- **Aswath Damodaran's blog and YouTube** — best free valuation education
- **Mauboussin essays** (Counterpoint Global)
- **Old Berkshire Hathaway letters**
- **Howard Marks memos**

## Key Takeaways

1. **Value = PV of future cash flows** — everything else is approximation
2. **Use multiple methods** — triangulation reveals confidence
3. **Ranges, not points** — false precision is the enemy of good valuation
4. **WACC dominates DCF** — 1% change in WACC ≈ 15–25% change in value
5. **Terminal value is fragile** — sanity-check with multiples
6. **Sector context matters** — apply industry-appropriate methods
7. **Reverse-DCF before your own DCF** — see what market expects
8. **Margin of safety required** — 20–30%+ discount to intrinsic
9. **Quality > cheapness** — Buffett: "A great business at a fair price > a fair business at a great price"
10. **Calibrate your forecasts** — track accuracy; humbleness scales with experience

See also: [[fundamental_analysis]], [[stocks]], [[reits]], [[private_markets]], [[credit_markets]], [[interest_rates]], [[market_cycles]], [[event_driven]]

## Where this connects

- [Private markets](private_markets.md) — DCF and comparable transactions are used for private company valuation
- [Interest rates](interest_rates.md) — the discount rate in DCF models is driven by interest rates
- [Credit markets](credit_markets.md) — credit analysis uses valuation to assess debt capacity
- [Portfolio management](portfolio_management.md) — valuation determines whether assets are cheap or expensive relative to fundamentals
- [Corporate finance](corporate_finance.md) — the issuer-side decisions (capital structure, budgeting, M&A) that the cost of capital and DCF feed

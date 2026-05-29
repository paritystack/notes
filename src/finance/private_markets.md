# Private Markets: PE, VC, and Alternatives

## Overview

Private markets — private equity (PE) and venture capital (VC) — invest in companies that aren't publicly traded. They've grown from $1T globally in 2000 to over $13T in 2024, increasingly becoming a core allocation for endowments, pensions, and family offices. For individual investors, access is widening through interval funds, registered private funds, and platforms — but the structure, fees, illiquidity, and reporting are radically different from public markets.

## Private vs. Public Markets: Why It Matters

| Feature | Public | Private |
|---------|--------|---------|
| Pricing | Continuous, market | Quarterly appraisal |
| Liquidity | Daily | 7–12 year lockup |
| Disclosure | SEC mandated | Limited |
| Investor base | Anyone | Accredited / Qualified |
| Reporting | Audited quarterly | Quarterly, varies |
| Volatility (reported) | High | Smoothed (artificially low) |
| Fees | 0.05%–1% | 2/20 plus expenses |
| Manager dispersion | Narrow | Wide |
| Universe size | ~4,000 U.S. stocks | 17,000+ U.S. private companies |

### "Illiquidity Premium"

Private markets historically deliver 200–400bp annualized excess over public equities for PE buyouts; arguably 0–500bp for VC (very wide dispersion). Whether this premium fully compensates for lockup, fees, and selection bias is debated — but top-quartile managers have decisively outperformed.

## Private Equity (PE)

### Strategies

| Strategy | Target Companies | Hold Period | Returns Source |
|----------|------------------|-------------|----------------|
| Leveraged Buyout (LBO) | Mature, cash-generating | 4–7 years | Leverage + operational improvement + multiple expansion |
| Growth Equity | Growing, profitable | 3–5 years | Revenue growth |
| Distressed | Stressed/bankrupt | 2–5 years | Restructuring, balance sheet repair |
| Secondaries | Existing fund stakes | Varies | Discount to NAV, J-curve mitigation |
| Real Assets | Infrastructure, real estate, energy | 5–10 years | Yield + appreciation |
| Mezzanine | Subordinated debt + equity warrants | 3–5 years | Coupon + upside |

### LBO Mechanics

```
Acquisition price = $1,000M
Equity         = $300M (sponsor capital)
Debt           = $700M (term loans, high-yield bonds)
                ──────
Total          = $1,000M

Hold 5 years:
- Pay down $250M of debt with FCF
- Grow EBITDA from $100M to $140M
- Exit at same 10x multiple = $1,400M

Exit waterfall:
- Pay off remaining $450M debt
- Equity proceeds = $950M
- Sponsor return = 3.17x money multiple, ~26% IRR
```

The three return drivers:
1. **Multiple expansion** (or contraction): pay 10x EBITDA, sell at 12x
2. **EBITDA growth**: operational improvement, M&A, expansion
3. **Debt paydown**: leverage that's repaid by FCF accrues to equity

```python
def lbo_return(entry_ev, exit_ev, entry_debt, exit_debt,
                entry_ebitda, exit_ebitda, hold_years):
    """Calculate LBO equity returns."""
    entry_equity = entry_ev - entry_debt
    exit_equity = exit_ev - exit_debt
    money_multiple = exit_equity / entry_equity
    irr = money_multiple ** (1 / hold_years) - 1
    
    # Decompose return drivers
    entry_multiple = entry_ev / entry_ebitda
    exit_multiple = exit_ev / exit_ebitda
    
    return {
        'money_multiple': money_multiple,
        'irr': irr,
        'multiple_expansion': exit_multiple / entry_multiple,
        'ebitda_growth': exit_ebitda / entry_ebitda,
        'debt_paydown_pct': (entry_debt - exit_debt) / entry_debt,
    }
```

### Growth Equity

Sits between VC and PE. Targets profitable, growing companies (often $20M–$200M revenue) seeking expansion capital. Less leverage than LBO, more mature than VC.

Examples: General Atlantic, Insight Partners, TA Associates, Summit Partners.

### Distressed PE

Buys debt or equity of stressed companies cheap, restructures, exits. Often involves Chapter 11 navigation. Wide returns dispersion — best in recessions.

Examples: Oaktree, Apollo, Cerberus.

## Venture Capital (VC)

### Stage Definitions

| Stage | Check Size | Valuation | Risk | Examples |
|-------|------------|-----------|------|----------|
| Pre-seed | $100k–$1M | $3–$10M | Extreme | Founder + idea |
| Seed | $500k–$3M | $5–$15M | Very high | Product + early traction |
| Series A | $5M–$20M | $20–$80M | High | Product-market fit |
| Series B | $20M–$50M | $80–$300M | High | Scaling sales |
| Series C+ | $50M+ | $300M+ | Moderate | Growth, eyeing exit |
| Pre-IPO / Late | $100M+ | $1B+ | Lower | Profitable, near liquidity event |

### Power Law Returns

VC returns follow a power law: most investments return zero, a few return 10x, very few return 100x+. The top 1% of returns drive most fund performance.

Distribution of typical Series A investments (rough):
- 30–50% complete loss
- 30–40% return capital or 2–3x (zombies / modest exits)
- 10–20% deliver 5x+
- 1–5% are "fund returners" (>20x)

```python
def vc_fund_return(investments):
    """
    investments: list of (invested, exited) pairs in millions.
    Demonstrates power law: a few winners dominate.
    """
    sorted_inv = sorted(investments, key=lambda x: x[1] / x[0], reverse=True)
    total_invested = sum(i[0] for i in investments)
    total_exited = sum(i[1] for i in investments)
    
    multiple = total_exited / total_invested
    top3_contribution = sum(i[1] for i in sorted_inv[:3]) / total_exited
    
    return {
        'total_multiple': multiple,
        'top_3_pct_of_returns': top3_contribution,
    }
```

### Why VCs Need 30+ Investments

Mathematical necessity. If 5% of investments return 10x and the rest return 0.5x average:
- Expected return per investment: 0.05 × 10 + 0.95 × 0.5 = 0.975x — actually below 1x
- But add a 1% chance of 50x: 0.05 × 10 + 0.01 × 50 + 0.94 × 0.5 = 1.47x

Diversification across many bets is required to capture the right tail.

## Fund Structure (PE and VC)

### General Partner (GP) and Limited Partner (LP)

- **GP**: the fund manager (Sequoia, Blackstone, etc.) — typically commits 1–5% of fund capital
- **LP**: the investor (pension, endowment, family office) — provides 95%+ of capital, no operational control

### Capital Commitments and Calls

LPs commit capital upfront (e.g., $10M to a fund), but don't write the check immediately. GP issues **capital calls** as deals close, typically over 3–5 years. LPs must have liquidity ready.

### Fund Life

Typical PE/VC fund lifecycle:

```
Year 0   :  Fund close
Year 1–5 :  Investment period (capital deployed)
Year 4–10:  Harvest period (exits, distributions)
Year 10  :  Fund termination (extensions common, 12 years total)
```

### The J-Curve

Early years show negative returns (fees + write-downs of failures) before winners mature and exits begin. Plot of cumulative net return vs. time looks like a "J":

```
NAV
  ↑
  │              ╱──── (exits begin)
  │           ╱
  │        ╱
  │_____╱
  │  ╲ ╱
  │   V (J-curve trough, year 3–5)
  └──────────────────→ time
```

Implication: don't judge a fund's IRR in years 1–4. Need 7+ years for meaningful return picture.

### Fees: The 2-and-20 Model

- **Management fee**: 2% of committed capital annually (PE/VC standard)
- **Carried interest** (carry): 20% of profits above a hurdle (typically 8% IRR)
- **Catch-up**: GP gets 100% of profits between hurdle and full catch-up before 80/20 split

#### Example Waterfall

```
Fund returns 2.5x ($250M on $100M invested over 8 years)
Hurdle: 8% IRR → ~$185M
Profits above hurdle: $250M – $185M = $65M

Catch-up (100% to GP until GP has gotten 20% of total profits):
- Total profits: $250M – $100M = $150M
- GP target: 20% × $150M = $30M
- After hurdle, GP gets all profit until $30M reached
- $30M comes out of the $65M above hurdle

Remaining: $35M split 80/20 → LP $28M, GP $7M

Total LP: $100M (capital) + $185M (hurdle) + $28M = $313M? No wait.
Actually: LP gets back $100M + $85M (8% IRR) + 80% of $65M − catch-up = the math.
```

The point: carry creates significant performance fees. A top-quartile fund returns 2–3x gross and 1.7–2.5x net of fees.

### Other Fees and Costs

- **Organizational expenses** (paid by fund)
- **Transaction fees** (deal sourcing, due diligence)
- **Monitoring fees** (paid by portfolio companies to GP — controversial)
- **Broken-deal expenses**
- **Total drag**: often 3–4% per year all-in vs. 0.05–0.5% for index funds

## Accreditation Requirements (U.S.)

Historically, direct private fund access required being an **accredited investor**:

- $200k income (single) / $300k (MFJ) for 2 years, OR
- $1M net worth (excluding primary residence)

SEC also recognizes "Qualified Purchaser" ($5M investments) for larger funds (3(c)(7) funds).

Recent expansions allow Series 7/65/82 holders to qualify regardless of wealth.

## Access Routes for Retail

### Directly Accessible

- **Publicly traded BDCs** (Business Development Companies): MAIN, ARCC, OBDC — invest in middle-market debt + equity; pay high dividends (8–11%)
- **Publicly traded PE firms**: BX (Blackstone), KKR, APO, CG — own GP economics; correlated to public markets
- **Listed VC**: limited; some BDCs do VC-like investments

### Interval Funds & Tender-Offer Funds

- Periodic redemption (monthly/quarterly), gated at ~5% per period
- Access to private credit, real estate, PE
- Examples: Cliffwater Corporate Lending Fund (CCLFX)

### Platforms (Accredited)

- **iCapital, CAIS, Moonfare, Yieldstreet**: aggregate retail access to top funds, lower minimums ($25k vs. $5M+)
- Additional fee layer (often 50–100bp)

### Direct Investing

- **AngelList, Republic, EquityZen, Forge Global**: pre-IPO secondary shares, syndicates
- High risk, low quality control, illiquidity

### Continuation Funds & Secondaries

- Buy existing LP stakes from sellers
- Discounts of 5–25% to NAV common
- Mitigates J-curve (mature assets)
- Funds: Lexington, Coller, Ardian, Strategic Partners

## Performance Measurement

### Common Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| IRR | Internal rate of return | Time-weighted return on capital deployed |
| Multiple (MOIC) | Total distributions / capital called | How many times your money back |
| DPI | Distributions to paid-in | Realized only (cash returned) |
| RVPI | Remaining value to paid-in | Unrealized NAV |
| TVPI | DPI + RVPI | Total value (distributions + NAV) / paid-in |
| PME | Public Market Equivalent | Compares to indexing same cash flows |

### Why IRR Can Mislead

- Heavily skewed by early distributions (high IRR but low absolute return)
- Multiple matters more for absolute wealth
- Top quartile by IRR is not always top by multiple

```python
def npv(rate, cashflows):
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows))

def irr(cashflows, guess=0.1, max_iter=100, tol=1e-6):
    """Newton's method IRR."""
    r = guess
    for _ in range(max_iter):
        n = npv(r, cashflows)
        # Derivative
        dn = sum(-t * cf / (1 + r) ** (t + 1) for t, cf in enumerate(cashflows))
        if abs(n) < tol:
            return r
        r -= n / dn
    return r

def tvpi(distributions, nav, capital_called):
    return (sum(distributions) + nav) / capital_called

def dpi(distributions, capital_called):
    return sum(distributions) / capital_called
```

### Public Market Equivalent (PME)

Compares fund cash flows against a public benchmark (S&P 500). Asks: "What if I had invested the same amount on the same date in SPY?"

- PME > 1: outperformed
- PME < 1: underperformed (after fees)

Kaplan-Schoar PME is the most common variant. As a rule, ~50% of PE funds beat their PME benchmark; top quartile decisively so.

## Manager Selection

### Top Quartile vs. Median Dispersion

| Asset | Top quartile minus median IRR |
|-------|-------------------------------|
| Public equity (mutual funds) | ~1.5pp |
| Hedge funds | ~5pp |
| Private equity | ~10pp |
| Venture capital | ~25pp |

Selecting top-tier managers matters far more in private markets than public. Top decile VC funds return 5x+; bottom decile return < 1x.

### Persistence of Performance

- Some persistence in PE (top GPs continue outperforming, especially buyout)
- Less persistence in VC (one big winner can mark an entire fund)
- Best predictor: deal flow quality + team continuity

### Access Problem

Top funds are often closed or oversubscribed:
- Sequoia, Benchmark, Andreessen Horowitz, Founders Fund — extremely hard to get into
- Many require pre-existing LP relationships, $5–10M minimums

## Pitfalls and Critiques

### Smoothed Returns ("Volatility Laundering")

Quarterly appraisal-based NAVs don't mark-to-market daily. PE looks much less volatile than public equity, but the underlying risk is similar.

True economic volatility ≈ delevered public-equivalent + leverage premium.

### Leverage as a Hidden Return Driver

LBO returns are often (partly) compensation for high leverage. Strip out leverage and the operational alpha is more modest.

### Carry Distortion

GPs are incentivized to take risk (call options on upside). Watch for misaligned fund structures.

### Fee Drag

3–4% annual fee drag is brutal. Top-quartile funds overcome it; median doesn't. Be ruthless about manager selection.

### Capital Call Risk

LPs must keep liquidity for calls. Endowments and pensions occasionally fail to meet calls in liquidity crises (2008), forfeiting positions at fire-sale prices.

### Vintage Year Effects

Returns vary dramatically by vintage. Funds raised at market peaks (2000, 2007, 2021) consistently underperform. Vintage diversification matters.

## Real Assets

Often grouped with private markets:

### Infrastructure

- Toll roads, airports, utilities, pipelines
- Long-life assets, inflation-linked cash flows
- Funds: Macquarie, Brookfield, Global Infrastructure Partners
- Public proxies: BIP, MIC (formerly), pipeline MLPs

### Real Estate Private Equity

- Opportunistic vs. core-plus vs. core
- See [[reits]] for public alternatives
- Funds: Blackstone Real Estate, Starwood

### Natural Resources

- Timberland, farmland, mineral rights
- Inflation hedge characteristics
- Funds: NCREIF Farmland Index, TIAA-CREF

### Commodity Trading Advisors (CTAs)

- See [[momentum_trend]] for trend-following strategies
- Liquid alternative to private trend funds

## Hedge Funds (Brief)

Distinct from PE/VC but often bucketed in "alternatives." Key differences:

- More liquid (monthly/quarterly redemptions)
- Typically focused on absolute returns
- Wider strategy variety (long/short, global macro, event-driven, multi-strat)
- 2-and-20 fee structure standard
- AUM ~$4–5T globally

Common strategies: equity long/short, market neutral, global macro, managed futures, multi-strategy, event-driven, distressed.

## Allocation Considerations

### Typical Endowment Allocation

Yale-style "endowment model":
- 30–40% private equity
- 15–25% real assets
- 20–30% absolute return (hedge funds)
- 10–15% public equity
- 5–10% bonds

For individual investors:
- 0–20% alternatives is typical
- Higher only if liquidity not needed for 10+ years
- Vintage diversification across 3–5 years

### Liquidity Planning

```python
def liquidity_plan(commitments, expected_call_schedule):
    """
    Estimate cash needed by year for capital calls.
    """
    cumulative = {}
    for c in commitments:  # each = (commitment_size, vintage_year)
        size, vintage = c
        for offset, pct in enumerate(expected_call_schedule):
            year = vintage + offset
            cumulative[year] = cumulative.get(year, 0) + size * pct
    return cumulative

# Typical PE call schedule: 25% Y1, 25% Y2, 20% Y3, 15% Y4, 10% Y5, 5% Y6
```

## Tax Considerations

- **K-1 partnership filings** (not 1099) — delays tax prep
- **UBTI risk** in IRAs (Unrelated Business Taxable Income)
- **Carried interest** taxed as LTCG for GPs (politically contested)
- **State filings** in every state the fund invests
- See [[tax_strategies]] for more

## Common Mistakes

1. **Chasing recent vintages** at peak valuations
2. **Concentrating in one GP** without diversification
3. **Overlooking fees** — compounding 3% drag is brutal
4. **Forgetting capital calls** liquidity needs
5. **Trusting smoothed NAVs** — real economic vol is higher
6. **Buying retail private products** with high fee + load layers
7. **Picking on brand name** vs. evaluating recent fund performance
8. **Not diversifying vintage years**
9. **Overconcentrating in PE/VC** without rest-of-portfolio rebalancing room
10. **Selling secondaries at fire-sale prices** in liquidity crunch

## Resources

### Data Providers
- **Cambridge Associates** — benchmark data (paid, institutional)
- **Preqin** — fund data, dry powder, deal flow
- **PitchBook** — startup/PE/VC database
- **Burgiss** — performance benchmarks
- **NVCA / PitchBook Venture Monitor** — quarterly VC report

### Books
- *King of Capital* — Carey & Morris (Blackstone history)
- *Private Equity at Work* — Eileen Appelbaum
- *Venture Deals* — Brad Feld, Jason Mendelson
- *Secrets of Sand Hill Road* — Scott Kupor
- *The Power Law* — Sebastian Mallaby (VC history)
- *Investing in Private Equity* — Frank Russo

### Sites
- **Institutional Investor** — industry news
- **Axios Pro Rata** — daily VC/PE newsletter
- **Equity Zen / Forge** — secondary market data
- **Sifted** / **The Information** — European/global VC coverage

## Key Takeaways

1. **Manager dispersion is wide** — top quartile decisively beats median
2. **Fees compound brutally** — 2-and-20 is a high bar
3. **J-curve is real** — judge funds at year 7+, not year 3
4. **Power law in VC** — diversification across 30+ deals required
5. **Smoothed NAVs hide volatility** — true risk higher than reported
6. **Vintage matters** — peak-vintage funds chronically underperform
7. **Liquidity planning is essential** — capital calls don't wait
8. **TVPI > IRR** for measuring absolute wealth
9. **Top fund access is the alpha** — without it, public proxies (BX, KKR, BDCs) suffice
10. **Allocate from non-liquidity-sensitive portion** of portfolio

# REITs (Real Estate Investment Trusts)

## Overview

A REIT (Real Estate Investment Trust) is a company that owns, operates, or finances income-producing real estate. REITs let you invest in real estate without buying property directly — they trade like stocks, offer high dividend yields, and provide exposure to a multi-trillion-dollar asset class. They're a distinct asset class from stocks and bonds, with their own valuation metrics, sensitivities, and tax treatment.

## REIT Basics

### Structure Requirements (U.S.)

To qualify as a REIT, a company must:

1. Invest at least **75% of assets** in real estate, cash, or Treasuries
2. Derive at least **75% of gross income** from real estate (rents, mortgage interest, sales)
3. Distribute at least **90% of taxable income** as dividends to shareholders
4. Be managed by a board of directors / trustees
5. Be a taxable corporation
6. Have at least **100 shareholders** after first year
7. No more than 50% of shares held by 5 or fewer individuals ("5/50 rule")

In exchange: REITs pay **no corporate income tax** on distributed earnings — they pass income directly to investors (who pay tax personally, see [[tax_strategies]]).

### Why REITs Exist

Before REITs (1960), only the wealthy could invest in income-producing real estate. The REIT structure democratized access:
- Liquidity (trade like stocks)
- Diversification (own pieces of hundreds of properties)
- Professional management
- Pass-through taxation
- Low minimums

## Types of REITs

### Equity REITs (~96% of market)

Own and operate physical properties. Revenue from rent.

| Sector | What they own | Tenant type | Lease length |
|--------|---------------|-------------|--------------|
| Residential | Apartments, single-family rentals | Individuals | 1 year |
| Industrial | Warehouses, logistics | Amazon, FedEx | 5–10 years |
| Office | Office buildings | Companies | 5–15 years |
| Retail | Malls, strip centers, freestanding | Stores | 5–10 years |
| Healthcare | Hospitals, senior living, medical office | Operators | 10–20 years |
| Self-storage | Storage facilities | Individuals | Monthly |
| Data centers | Server farms | Tech companies | 5–15 years |
| Cell towers | Wireless infrastructure | Carriers | 5–10 years |
| Lodging | Hotels | Operators / guests | Nightly |
| Specialty | Casinos, prisons, billboards | Various | Long |
| Timber | Forestland | Logging/paper cos | Long |
| Net lease | Single-tenant properties | Various | Very long, triple-net |

### Mortgage REITs (mREITs)

Don't own property; own real estate debt (mortgages, MBS — see [[credit_markets]]).

- Borrow short, lend long → highly leveraged
- Earnings = net interest spread
- Very rate-sensitive (curve flattening = pain)
- Examples: NLY, AGNC
- Higher yields (12–15%), much higher risk
- Frequent dividend cuts in stress

### Hybrid REITs

Mix of equity + mortgage holdings. Less common.

### Public vs. Private vs. PNLR

| Type | Liquidity | Transparency | Minimum | Fees |
|------|-----------|--------------|---------|------|
| Public listed | Daily | High (SEC) | 1 share | Low |
| Public non-listed (PNLR) | Quarterly buybacks | Medium | $1k–10k | High (3–5% upfront) |
| Private | Lockup years | Low | $25k+ accredited | High (1.5–2.5%) |

For most retail investors, **stick to public listed REITs** or REIT ETFs. PNLR sales pitches often deceive on liquidity and fees.

## How REITs Make Money

```
Net Operating Income (NOI) = Rental Income - Operating Expenses
Funds From Operations (FFO) = Net Income + D&A - Property Sale Gains
Adjusted FFO (AFFO) = FFO - Recurring CapEx - Straight-line rent adjustments
```

NOI measures property-level profitability. FFO is the REIT-equivalent of earnings (adds back non-cash depreciation, which is meaningless for real estate). AFFO is the cleanest measure of cash flow available for dividends.

```python
def cap_rate(noi, property_value):
    """Capitalization rate: yield on property."""
    return noi / property_value

def ffo(net_income, depreciation, amortization, gains_on_sales=0):
    return net_income + depreciation + amortization - gains_on_sales

def affo(ffo_value, recurring_capex, straight_line_adj=0):
    return ffo_value - recurring_capex - straight_line_adj

def dividend_coverage(affo, dividends_paid):
    """Should be > 1.0; 1.1–1.3 is healthy."""
    return affo / dividends_paid
```

## REIT Valuation Metrics

Don't use P/E for REITs — it's distorted by depreciation. Use FFO/AFFO multiples and NAV.

### Price/FFO and Price/AFFO

REIT equivalent of P/E.

| Metric | Typical Range |
|--------|--------------|
| P/FFO | 10x–25x (sector-dependent) |
| P/AFFO | 12x–30x |

High-growth REITs (data centers, towers) trade at 25–35x AFFO; mature retail/office at 8–15x.

### Net Asset Value (NAV)

Estimate of property portfolio's fair market value minus debt. Compare to share price:
- **Premium to NAV** → market expects growth
- **Discount to NAV** → distressed or out of favor
- Historic average: REITs trade near NAV; 10%+ discount is often a buying signal

### Implied Cap Rate

```python
def implied_cap_rate(market_cap_plus_debt, noi):
    """Market-implied cap rate on the REIT's properties."""
    return noi / market_cap_plus_debt
```

Compare to private-market cap rates for the same property type. A REIT implying 7% caps when private market is 5% suggests undervaluation.

### Dividend Yield

REIT yields range 2.5% (high-growth) to 8%+ (distressed or high-payout sectors). Note:
- High yield ≠ good buy
- Sustainability matters more than current yield
- Check dividend coverage ratio (AFFO / dividends)

### NAV Premium/Discount Tools

Sites like NAREIT, Green Street, and individual REIT investor presentations provide NAV estimates.

## Interest Rate Sensitivity

REITs are highly sensitive to interest rates because:

1. They rely on debt to fund acquisitions
2. Cap rates compete with the 10-year Treasury yield
3. Dividend yields are valued vs. risk-free rate
4. Higher rates → lower commercial real estate values

### Duration Estimate

REITs have an effective equity duration of roughly **6–10 years** — somewhere between equities (~15+) and bonds.

When 10Y rises 1%, REITs historically fall ~7–12% over 6 months.

```python
def reit_rate_sensitivity(reit_etf_return, treasury_10y_change_pct):
    """Empirical regression coefficient (illustrative). Real value varies."""
    return -8 * treasury_10y_change_pct  # ~-8% per 1pp rise
```

See [[interest_rates]] for more on duration mechanics.

### Why Some REITs Resist Rate Shocks

- Short lease terms with mark-to-market resets (residential, storage)
- Pricing power (towers, data centers in supply-constrained markets)
- Fixed-rate, long-dated debt (large-caps with strong balance sheets)
- Growth that outpaces rate impact

## Sector Deep Dives

### Industrial / Logistics

- E-commerce tailwinds (Amazon, last-mile)
- High rent growth, low vacancy
- Long leases with annual escalators
- Top names: PLD (Prologis), STAG, REXR

### Data Centers

- AI/cloud demand boom (2023+)
- High development capital intensity
- Power/cooling constraints = moats
- Top names: EQIX (Equinix), DLR (Digital Realty)

### Cell Towers

- Long contracts (5–10 year MLAs with 3% escalators)
- High operating leverage (incremental tenant = near-100% margin)
- Top names: AMT, CCI, SBAC
- Rate-sensitive; struggled 2022–23

### Residential

- Sun Belt apartments (CPT, MAA, ESS, AVB)
- Single-family rentals (INVH, AMH)
- Rent growth tied to wage growth + housing supply

### Healthcare

- Senior housing (recovering from COVID)
- Medical office (stable)
- Life sciences (Boston, SF biotech demand cyclical)
- Top names: WELL, VTR, HCP

### Self-Storage

- Sticky tenants, low capex
- Pricing power via dynamic pricing algorithms
- Top names: PSA, EXR, CUBE, LSI

### Net Lease (Triple Net)

- Tenant pays taxes, insurance, maintenance
- Very long leases (10–25 years), low management
- Bond-like cash flows
- Top names: O (Realty Income), STAG, ADC, NNN

### Office

- Post-COVID disruption; WFH demand shock
- Best: trophy assets in supply-constrained markets
- Worst: commodity Class B/C in tier-2 cities
- Top names: BXP, VNO, ARE (life sciences-tilted)

### Retail

- Class A malls vs. dead malls — bifurcated
- Grocery-anchored centers resilient
- Top names: SPG, REG, KIM, FRT

### Hospitality

- Volatile (RevPAR cycles)
- Operationally intensive
- Top names: HST, RHP

## REIT ETFs

For broad exposure without single-name risk:

| ETF | Style | Expense Ratio | Notes |
|-----|-------|---------------|-------|
| VNQ | U.S. broad REIT | 0.12% | Vanguard, largest |
| SCHH | U.S. REITs ex-mortgage | 0.07% | Schwab |
| IYR | U.S. real estate | 0.40% | iShares |
| REET | Global REITs | 0.14% | International exposure |
| VNQI | International REITs | 0.12% | Vanguard ex-US |
| ICF | Cohen & Steers Realty Majors | 0.34% | Active-flavored |
| NETL | Net lease | 0.60% | Sector tilt |

## REIT Dividends and Taxes

### Tax Treatment

Most REIT dividends are **non-qualified ordinary income** — taxed at marginal income rate, not LTCG rates. Implications:

- Avoid REITs in taxable accounts when possible
- Hold in tax-deferred (Trad IRA / 401k) — see [[tax_strategies]]
- Section 199A: 20% Qualified Business Income (QBI) deduction available on REIT dividends through 2025 (sunsets without legislation)

### Dividend Components

| Type | Tax Treatment |
|------|---------------|
| Ordinary dividends | Marginal income rate |
| Capital gain distributions | LTCG rate |
| Return of capital | Not taxed; reduces basis (deferred LTCG) |

REITs send 1099-DIV with breakdown in Box 1a/1b/2a/3.

## Risks Specific to REITs

1. **Interest rate risk** — biggest macro driver
2. **Property type concentration** — office in 2020, retail in 2017
3. **Tenant concentration** — single big tenant going bankrupt
4. **Geographic concentration** — local market collapse
5. **Leverage** — most REITs run 30–50% LTV; refinancing risk
6. **Dividend cut risk** — required to distribute 90%; cuts signal stress
7. **Dilution** — REITs grow via share issuance; check NAV-per-share trajectory
8. **Cap rate expansion** — if buyer demand falls, property values drop
9. **Liquidity** (PNLRs especially) — gated redemptions
10. **Management quality** — external management = conflicts of interest (avoid)

## Internal vs. External Management

- **Internal** — management is employees of the REIT (better alignment)
- **External** — manager is paid AUM fees regardless of performance (red flag; historical underperformance)

Prefer internally managed REITs.

## REIT Cycles

REITs follow real estate cycles, which lag the broader business cycle:

| Phase | Behavior |
|-------|----------|
| Recovery | Vacancies fall, rents rise slowly |
| Expansion | New supply, rents rise faster |
| Hypersupply | Vacancies rise, rent growth peaks |
| Recession | Vacancies high, rents fall |

Each property sector has its own cycle. Industrial peaked 2022; office bottoming 2024+ (debatable).

## Building a REIT Allocation

### Strategic (long-term)

- 5–15% of portfolio in REITs is typical
- Diversifier vs. stocks/bonds (correlation ~0.5–0.7 to stocks)
- Inflation hedge (rent escalators, replacement cost)

### Tactical Tilts

Within REITs, tilt by:
- **Cycle phase**: industrial in early cycle, defensive in late
- **Macro view**: rate hikes → favor short-duration sectors (residential, storage)
- **Theme**: AI → data centers, e-commerce → industrial
- **Valuation**: deep discount to NAV → contrarian buy

### Direct Stock vs. ETFs

- **ETFs**: simple, diversified, lower risk
- **Individual REITs**: higher dividend yield possible, sector targeting, more research

## Real Estate Beyond REITs

For completeness — alternatives to REITs:

- **Direct ownership** (rentals): leverage, tax shelter, illiquid
- **Real estate crowdfunding** (Fundrise, RealtyMogul, CrowdStreet): minimums $500–$10k
- **Real estate private equity funds**: accredited only, see [[private_markets]]
- **Real estate operating companies** (non-REIT): JLL, CBRE, Marcus & Millichap
- **Homebuilders**: DHI, LEN, PHM (different exposure — to new home sales)
- **Real estate debt funds**: see [[credit_markets]] for CMBS, debt funds

## REIT Investing Workflow

### Screening

1. **Sector view**: which property type do you favor?
2. **Quality screen**: low leverage (Debt/EBITDA < 6x), high coverage (AFFO/Div > 1.2)
3. **Valuation**: P/AFFO vs. peers and history; discount to NAV
4. **Management**: internal preferred, insider ownership a plus
5. **Growth**: AFFO/share growth track record

### Monitoring

- Quarterly: FFO/AFFO trajectory, same-store NOI growth, occupancy
- Annual: NAV update, leverage, capex trends
- Macro: 10Y Treasury, cap rate environment
- Sector: supply pipeline, lease rollover

## Common Mistakes

1. **Chasing yield** — high yields often signal coming dividend cuts
2. **Using P/E instead of P/FFO** — useless for REITs
3. **Ignoring rate risk** — REITs got crushed 2022 along with bonds
4. **Buying PNLRs from advisors** — high fees, illiquid, often inferior to public REITs
5. **Concentrating in one sector** — office, retail, hospitality cycles diverge
6. **Holding in taxable** — wasting tax-advantaged space
7. **Ignoring lease rollover schedules** — big tenant move-outs
8. **Trusting NAV from REIT itself** — companies overstate; use Green Street or NAREIT
9. **Buying mortgage REITs for yield** — high risk; many blow up in vol regimes
10. **Forgetting development risk** — REITs that build are bets on construction cost & lease-up

## Resources

### Data and Research
- **NAREIT** (nareit.com) — industry stats, sector definitions
- **Green Street Advisors** — premier private REIT research (paid)
- **Seeking Alpha** — REIT-focused authors (Brad Thomas, Hoya Capital)
- **SEC EDGAR** — REIT filings, 10-Ks, supplementals
- **REIT investor presentations** — quarterly supplementals are essential

### Books
- *The Intelligent REIT Investor* — Brad Case, Stephanie Krewson-Kelly
- *Educated REIT Investing* — Stephanie Krewson-Kelly
- *Investing in REITs* — Ralph Block (older but foundational)

### Sites
- **REIT.com** — NAREIT's investor hub
- **Hoya Capital** — sector-level coverage
- **WideMoatResearch** / **REIT/base** — paid newsletters

## Key Takeaways

1. **REITs are required to pay out 90%** of taxable income — dividend-heavy by structure
2. **Use FFO/AFFO not P/E** — depreciation distorts net income
3. **NAV discount/premium** is a key tactical signal
4. **Sectors diverge dramatically** — don't paint all REITs with one brush
5. **Highly rate-sensitive** — duration of 6–10 years
6. **Most REIT dividends = ordinary income** — hold in tax-advantaged accounts
7. **Avoid external management and PNLRs**
8. **Industrial, data centers, towers** were 2010s leaders; cycle has rotated
9. **Stick to public listed** for liquidity and transparency
10. **5–15% portfolio allocation** is a typical strategic weight

## Where this connects

- [Portfolio management](portfolio_management.md) — REITs provide real estate exposure with stock-like liquidity
- [Interest rates](interest_rates.md) — REITs are sensitive to rate changes (like bonds but with equity upside)
- [Tax strategies](tax_strategies.md) — REIT dividends have specific tax treatment (ordinary income, QBI deduction)

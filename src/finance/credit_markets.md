# Credit Markets

## Overview

Credit markets are where corporations, governments, and consumers borrow. The U.S. corporate bond market alone is ~$10T; leveraged loans are $1.5T; structured credit (MBS, ABS, CLOs) is multi-trillion. Credit is bigger than equity markets globally — and it's where most institutional capital lives. For investors, credit offers higher yields than Treasuries, but introduces **default risk** alongside [[interest_rates]] sensitivity. The relationship between credit spreads and the business cycle ([[market_cycles]]) is one of the most reliable risk-on/risk-off signals available.

## Investment Grade vs. High Yield

### Credit Rating Tiers

| Tier | Moody's | S&P / Fitch | Default rate (avg) |
|------|---------|-------------|---------------------|
| Prime | Aaa | AAA | 0.00% |
| High grade | Aa1–Aa3 | AA+ to AA– | 0.02% |
| Upper-medium | A1–A3 | A+ to A– | 0.06% |
| Lower-medium | Baa1–Baa3 | BBB+ to BBB– | 0.18% |
| **Investment Grade ↑ / High Yield ↓** | | | |
| Speculative | Ba1–Ba3 | BB+ to BB– | 0.7% |
| Highly speculative | B1–B3 | B+ to B– | 3% |
| Substantial risk | Caa | CCC | 12% |
| Extreme risk | Ca | CC | High |
| In default | C/D | D | — |

### Investment Grade (IG)

- BBB– or higher (Baa3 or higher Moody's)
- ~$7T U.S. market
- Low default rates (1–2% historically over decade)
- Yields 50–250bp over comparable Treasuries
- Eligible for pension funds, insurance company holdings, central bank QE

### High Yield (HY) / Junk

- BB+ or lower (Ba1 or lower)
- ~$1.5T U.S. market
- Default rates 2–4% in normal times, 10–15% in recessions
- Yields 300–800bp over Treasuries (more in crisis)
- Equity-like risk and return in stress periods

```python
def expected_loss(default_rate, recovery_rate):
    """Expected credit loss = PD × (1 - RR)."""
    return default_rate * (1 - recovery_rate)

def required_spread(default_rate, recovery_rate, risk_premium=100):
    """
    Minimum spread to compensate for expected loss + risk premium.
    Returns spread in basis points.
    """
    el = expected_loss(default_rate, recovery_rate)
    return el * 10000 + risk_premium
```

## Credit Spreads

**Spread** = bond yield − comparable Treasury yield

Measured in basis points (bp). 100bp = 1%.

### Spread Types

- **G-spread**: vs. linearly interpolated Treasury
- **I-spread**: vs. swap curve
- **Z-spread**: zero-volatility spread (constant spread to spot curve)
- **OAS (Option-Adjusted Spread)**: removes value of embedded options (call, put, prepay)
- **CDS spread**: from credit default swap market

### Historical Spread Ranges

| Index | Tight (calm) | Wide (crisis) | 20-yr avg |
|-------|--------------|---------------|-----------|
| IG OAS | 80bp | 600bp (2008) | 145bp |
| HY OAS | 250bp | 2,000bp (2008) | 500bp |
| EM USD bonds | 200bp | 1,000bp | 350bp |
| Leveraged loans | 250bp | 1,500bp | 450bp |

### Spreads as Risk Signal

| HY OAS | Regime |
|--------|--------|
| < 350bp | Risk-on / complacency |
| 350–500bp | Normal |
| 500–700bp | Stress emerging |
| > 700bp | Crisis (often equity bottom near here) |
| > 1000bp | Severe distress (buy the bottom historically) |

```python
def credit_regime(hy_oas_bp):
    if hy_oas_bp < 350: return 'risk_on'
    if hy_oas_bp < 500: return 'normal'
    if hy_oas_bp < 700: return 'stress'
    if hy_oas_bp < 1000: return 'crisis'
    return 'severe_distress (historically: equity buy zone)'
```

Credit typically **leads equities** at major turns. Watch HY OAS for early warning of equity drawdowns.

## Yield-to-Maturity vs. Total Return

A bond's **YTM** assumes you hold to maturity and reinvest coupons at the same yield. Real returns differ because:

- You may sell early (price moves with rates)
- Coupons reinvested at then-prevailing rates
- Defaults remove principal
- Calls limit upside in falling-rate environments

```python
def bond_total_return(start_price, end_price, coupons_received,
                       holding_period_years):
    """Simple total return calculation, annualized."""
    total = (end_price - start_price + coupons_received) / start_price
    return (1 + total) ** (1 / holding_period_years) - 1
```

## Bond Math: Duration and Convexity

Same as covered in [[interest_rates]] but with credit twist:

- **Spread duration**: sensitivity to spread changes (separate from rate duration)
- **DTS (Duration Times Spread)**: better risk measure for HY
- HY behaves more like equity than IG — convexity reverses in stress

### Spread Duration Example

| Bond | Yield | Spread | Spread Duration | Rate Duration |
|------|-------|--------|----------------|---------------|
| 10Y Treasury | 4.5% | 0 | 0 | 8.5 |
| 10Y IG corporate | 5.5% | 100bp | 7.5 | 8.0 |
| 10Y HY corporate | 9.0% | 450bp | 4.5 | 5.5 |
| 10Y CCC bond | 14% | 950bp | 2.5 | 3.0 |

Lower-quality bonds have shorter duration because higher yields create more discounting.

## Credit Default Swaps (CDS)

A CDS is insurance on a bond defaulting:
- Buyer pays an annual premium (the CDS spread)
- Seller pays out if reference entity defaults

Used for:
- Hedging credit exposure
- Speculating on default risk
- Synthetic exposure without owning the bond

### CDS Market Structure

- **Single-name CDS**: on specific issuer
- **Index CDS**: on baskets (CDX.IG, CDX.HY for North America; iTraxx for Europe)
- **CDS Indices** are the most liquid credit instruments

```python
def cds_premium(notional, spread_bp):
    """Annual CDS premium payment."""
    return notional * spread_bp / 10000

def cds_pnl_on_credit_event(notional, recovery_rate=0.40, premium_paid_to_date=0):
    """P&L for CDS buyer if credit event occurs."""
    return notional * (1 - recovery_rate) - premium_paid_to_date
```

### Reading CDS Spreads

CDS spreads often more accurate than bond spreads (more liquid, less affected by repo and balance sheet constraints).

A single-name CDS at 500bp implies ~5% annual default probability (assuming 40% recovery).

### Famous CDS Events

- **2008 GFC**: AIG sold $400B+ of CDS protection without capital backing → Treasury rescue
- **Greek default 2012**: CDS triggered after restructuring controversy
- **Archegos 2021**: total return swaps (cousin product) blew up Credit Suisse and others

## Corporate Bond Strategies

### Buy-and-Hold Investment Grade

- High-quality issuers (Apple, Microsoft, JPM)
- Hold to maturity for predictable income
- Laddered portfolio across 1–10 year maturities

### Active IG Credit

- Sector selection (utilities vs. tech vs. financials)
- Duration positioning (long when expecting rate cuts)
- Issue selection (analysis of company fundamentals)

### High Yield Strategies

- **Yield-focused**: BB-rated names, avoid CCCs except in cycle troughs
- **Distressed/special situations**: buy fallen angels, restructurings
- **Capital structure arbitrage**: long senior secured / short equity or unsecured

### Spread Trades

- Long IG / short Treasuries (long credit spread)
- Long HY / short IG (long credit risk relative to quality)
- Long/short between sectors

### Curve Trades

- IG short end (1–3Y) vs. long end (10Y+)
- Different macro views priced at different points

## Leveraged Loans

Senior secured bank loans to non-investment-grade companies, typically floating rate (SOFR + spread).

### Characteristics

- **Floating rate** — low duration (~0.25 years), good in rising-rate environments
- **Senior secured** — first claim on assets; higher recoveries (~65% vs. 40% for HY bonds)
- **Often issued for LBO financing** ([[private_markets]])
- **Covenant-lite (cov-lite)** structure dominant since 2017 — fewer protections than legacy loans
- **Market size**: ~$1.5T U.S.

### Loan vs. HY Bond

| Feature | Leveraged Loan | HY Bond |
|---------|----------------|---------|
| Rate | Floating | Fixed |
| Seniority | Senior secured | Senior unsecured (typically) |
| Recovery | 65% | 40% |
| Duration | ~0.25 | 4–6 years |
| Liquidity | Lower | Higher |
| Tradable in CLOs | Yes (primary collateral) | No |

### Trading Vehicles

- Bank loan mutual funds / ETFs (BKLN, FFRHX, SRLN)
- CLO equity / mezzanine (more sophisticated)

## Mortgage-Backed Securities (MBS)

Bonds backed by pools of mortgages. The U.S. MBS market is ~$11T.

### Agency vs. Non-Agency

| Type | Issuer | Guarantee | Risk |
|------|--------|-----------|------|
| Agency MBS | Fannie Mae, Freddie Mac, Ginnie Mae | Government-backed | Prepayment, rates |
| Non-agency (Private Label) | Banks, REITs | None | Credit + prepayment |

### Agency MBS

- Backed by U.S. government (explicit for Ginnie, implicit for Fannie/Freddie)
- Major holdings of Federal Reserve (~$2T in 2024)
- Trade like Treasuries but with prepayment uncertainty
- Yield premium of 50–150bp over Treasuries

### Prepayment Risk

Homeowners refinance when rates fall → bond holders get principal back at par when bond was trading at premium. This is **negative convexity**: prices rise less in falling rates and fall more in rising rates.

```python
def prepayment_speed_psa(age_months, base_cpr=0.06, ramp_months=30):
    """
    PSA (Public Securities Association) prepayment model.
    100 PSA = 0.2% CPR ramping to 6% CPR over 30 months.
    """
    if age_months < ramp_months:
        return base_cpr * (age_months / ramp_months)
    return base_cpr  # 100 PSA
```

### Non-Agency MBS

- Jumbo loans (above conforming limits)
- Subprime (largely defunct since 2008)
- Investor properties
- Credit-sensitive: defaults reduce cash flows

### MBS Strategies

- **Specified pools**: select pools by borrower characteristics (e.g., low FICO = slower prepayments)
- **CMOs (Collateralized Mortgage Obligations)**: tranche cash flows into PAC, support, IO, PO classes
- **TBA (To Be Announced)** trading: most liquid agency MBS market

## Asset-Backed Securities (ABS)

Bonds backed by pools of non-mortgage consumer/commercial loans.

### Major ABS Types

| Collateral | Market Size (approx) |
|------------|---------------------|
| Auto loans (prime + subprime) | $250B |
| Credit cards | $150B |
| Student loans | $200B (private) |
| Equipment leases | $50B |
| Aircraft / shipping container | $30B |
| Solar / PACE | growing |
| Whole business (Domino's, Sonic) | $100B+ |

### Tranching

ABS deals are sliced into tranches by seniority:

```
AAA Senior   ──┐
                │ first losses go to bottom
AA Mezzanine   │
A Mezzanine    │
BBB Mezzanine  │
BB Subordinate │
Equity         ──┘ first to lose
```

Senior tranches highly rated; junior absorb defaults first.

## CLOs (Collateralized Loan Obligations)

A type of ABS backed by **leveraged loans**. Market ~$1T globally.

### Structure

- Pool 100–250 leveraged loans
- Issue tranches from AAA to equity
- Equity tranche absorbs first losses but captures excess yield
- Managed by CLO managers (active loan selection)

### Why CLOs Matter

- Largest buyer of leveraged loans (~70% of new issuance)
- Multi-trillion dollar market
- Different from 2008 CDO squared — actual cash-flow CLOs performed well in GFC
- Returns: AAA ~5–6%, equity 15–20%+ when working

### Risk

- Concentration in cyclical loan market
- Worst case: deep recession with high default rate erodes mezz/equity tranches
- COVID 2020 stressed but did not break CLO structures

## CMBS (Commercial Mortgage-Backed Securities)

Bonds backed by commercial real estate loans (office, retail, hotel, multifamily).

- Single-asset / single-borrower (SASB) — one loan, big property
- Conduit — pool of many loans
- Office CMBS has been stressed since 2020 (WFH)

## Emerging Market (EM) Debt

### Categories

- **Sovereign hard currency** (USD-denominated): JPMorgan EMBI index
- **Sovereign local currency** (in EM currency): JPMorgan GBI-EM index
- **EM corporate** (USD or local): CEMBI index

### Considerations

- Higher yields (300–800bp over Treasuries)
- Currency risk on local-currency bonds
- Political/sovereign risk (Argentina, Venezuela defaults)
- Correlation to commodities, USD cycle
- Diversification benefit for global portfolios

## Municipal Bonds

State/local government debt; interest typically federal tax-exempt.

- General Obligation (GO) vs. Revenue bonds
- Insured vs. uninsured
- Taxable equivalent yield calculation:

```python
def taxable_equivalent_yield(muni_yield, marginal_tax_rate, state_tax=0):
    """
    For high earners, munis can beat taxable bonds after-tax.
    """
    effective_tax = 1 - (1 - marginal_tax_rate) * (1 - state_tax)
    return muni_yield / (1 - effective_tax)

# Example: 3.5% muni for someone in 32% federal + 6% state
# TEY = 3.5% / (1 - 0.32) / (1 - 0.06) ≈ 5.5%
```

Best for high-tax-bracket investors in taxable accounts. See [[tax_strategies]].

## Credit Cycle

Credit cycles often differ from business cycles:

```
1. Easy money → spreads tight, leverage rises
2. Excess → low-quality issuance booms (peak in CCC supply)
3. Risk-off catalyst → spreads widen, refi market closes
4. Defaults rise → distressed buyers emerge
5. Survivors recapitalize, cycle restarts
```

### Where We Are in the Cycle

Indicators:
- HY new-issue volume
- CCC share of HY new issuance (>20% = late cycle)
- Cov-lite share of leveraged loans
- Default rate trend (Moody's monthly)
- Distressed ratio (% of HY trading > 1000bp wide)

```python
def credit_cycle_indicator(ccc_pct_of_new_issue, default_rate, hy_oas_bp):
    """Simple heuristic for credit cycle phase."""
    if ccc_pct_of_new_issue > 25 and default_rate < 3 and hy_oas_bp < 400:
        return 'late_cycle (peak euphoria, sell)'
    if default_rate > 6 and hy_oas_bp > 700:
        return 'crisis (often buy)'
    return 'mid_cycle'
```

## Investing Vehicles

### Funds and ETFs

| ETF | Asset | Notes |
|-----|-------|-------|
| AGG / BND | Aggregate bond | Broad IG mix |
| LQD | IG corporate | Most liquid IG ETF |
| HYG / JNK | HY corporate | Liquid HY |
| BKLN | Bank loans | Floating rate |
| MBB | MBS | Agency mortgages |
| EMB | EM USD sovereign | |
| MUB | National municipal | Tax-exempt |
| SHY / IEF / TLT | Treasuries by duration | Rate exposure |
| HYDB | High-yield with duration hedge | Removes rate risk |

### Active Funds

- **PIMCO Income (PIMIX/PONAX)** — multi-sector workhorse
- **DoubleLine Total Return (DBLTX)** — MBS-heavy
- **Loomis Sayles Bond (LSBDX)** — opportunistic
- **TCW Total Return** — MBS specialist
- **BlackRock Strategic Income**

### CEFs (Closed-End Funds)

- Trade at premium/discount to NAV
- Use leverage (10–30%)
- Higher yields, more volatility
- Examples: PDI, EVT, NUV

### Direct Bond Investing

- Treasuries: TreasuryDirect.gov (auctions, T-bills)
- Corporate bonds: through broker (Schwab, Fidelity, IBKR)
- Tax-loss-harvestable; precise duration control
- Spread costs higher than ETFs for retail

## Risks Specific to Credit

1. **Default risk** — issuer can't pay
2. **Downgrade risk** — rating cut causes spread widening
3. **Liquidity risk** — HY can be hard to sell in stress
4. **Call risk** — issuer redeems early when rates fall
5. **Prepayment risk** (MBS) — homeowners refi early
6. **Extension risk** (MBS) — homeowners refi later than expected
7. **Concentration risk** — fund holdings concentrated in distressed sectors
8. **Currency risk** — EM and foreign bonds
9. **Liquidity vs. mark risk** — IG bonds can be hard to sell at quoted levels

## Common Mistakes

1. **Yield chasing** — top yield often = worst credit
2. **Ignoring duration** — long-dated IG bombs in rate spikes
3. **Buying CEFs at premiums** — gives up alpha to NAV
4. **Concentrating in single sector** — energy bonds 2015, retail bonds 2017
5. **Underestimating recovery rates** — HY recovery varies widely
6. **Trusting ratings alone** — Lehman, Bear, Enron were IG before collapse
7. **Frequent trading IG bonds** — high spreads eat returns
8. **Ignoring spread duration** in HY portfolios
9. **Avoiding credit entirely** in retirement portfolios — gives up risk premium
10. **Buying MBS without understanding negative convexity**

## Resources

### Data
- **FRED** — yields, spreads, defaults
- **ICE/BAML indices** — IG OAS, HY OAS (HY0A4, HY0A0)
- **JPMorgan EMBI** — EM spreads
- **Moody's Default Studies** — annual default and recovery data
- **TRACE** (FINRA) — actual bond trade prices

### Books
- *Fixed Income Securities* — Bruce Tuckman
- *The Handbook of Fixed Income Securities* — Frank Fabozzi
- *Distressed Debt Analysis* — Stephen Moyer
- *The Big Short* — Michael Lewis (CDO/MBS pre-2008)
- *The Bond King* — Mary Childs (Bill Gross / PIMCO)

### Sites
- **Reorg Research** — distressed/restructuring (paid)
- **9fin** — leveraged finance news
- **LSTA** — leveraged loan industry stats
- **SIFMA** — bond market statistics

## Key Takeaways

1. **Credit > equity in market size** — and where most institutional capital lives
2. **Spreads compensate for default + risk premium** — and proxy market stress
3. **HY OAS is a top recession/risk indicator** — leads equities at turns
4. **IG and HY behave differently** — IG more rate-sensitive, HY more credit-sensitive
5. **Recovery rates vary**: senior secured loans recover ~65%, unsecured ~40%, sub debt much less
6. **Leveraged loans are floating rate** — minimal duration; good in rising-rate regimes
7. **MBS has negative convexity** — important for rate-bet sizing
8. **CLOs are the dominant leveraged loan buyer** — different from infamous 2008 CDOs
9. **Munis for high tax brackets** — calculate TEY ([[tax_strategies]])
10. **Credit cycle ≠ business cycle** — watch CCC issuance, defaults, distressed share as cycle markers

## Where this connects

- [Bonds](bonds.md) — credit markets include investment-grade and high-yield bonds
- [Interest rates](interest_rates.md) — credit spreads widen when rates rise; rates drive the risk-free base
- [Derivatives](derivatives.md) — credit default swaps (CDS) are the primary credit derivatives
- [Private markets](private_markets.md) — private credit (direct lending) is growing as banks retreat

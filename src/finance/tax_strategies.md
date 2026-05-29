# Tax Strategies for Investors

## Overview

Taxes are the largest controllable drag on long-run investment returns. The IRS doesn't tax your gross return — it taxes your **after-tax** return, and the gap can be 20–40% over a lifetime. Smart tax strategy is the closest thing in investing to a risk-free improvement: tax-loss harvesting, asset location, lot selection, and account sequencing routinely add 0.5–1.5% per year to net returns. None of it requires forecasting markets.

This guide is U.S.-focused. International readers should treat the principles as transferable but specifics as illustrative.

## Capital Gains Basics

### Short-Term vs. Long-Term

| Holding Period | Tax Treatment | 2024 Top Rate |
|---------------|---------------|---------------|
| ≤ 1 year (short-term) | Ordinary income | 37% federal |
| > 1 year (long-term) | LTCG rates | 0% / 15% / 20% |

Holding stocks for **at least one year + one day** can reduce federal tax by ~17 percentage points for high earners. This alone is one of the biggest reasons not to overtrade.

### 2024 Long-Term Capital Gains Brackets (single filer, illustrative)

| LTCG Rate | Taxable Income |
|-----------|----------------|
| 0% | up to $47,025 |
| 15% | $47,026 – $518,900 |
| 20% | above $518,900 |

Add **3.8% Net Investment Income Tax (NIIT)** for high earners (MAGI > $200k single / $250k MFJ).

State taxes stack on top (no preferential LTCG rate in most states — CA, NY tax at ordinary).

### Qualified vs. Ordinary Dividends

- **Qualified dividends** (most U.S. C-corps, holding period > 60 days around ex-date): LTCG rates
- **Ordinary dividends** (REITs, MLPs, BDCs, foreign): ordinary income rates

REIT dividends ([[reits]]) are mostly ordinary — placement matters.

## Tax-Loss Harvesting (TLH)

Selling losing positions to realize capital losses that offset gains and reduce taxable income.

### Mechanics

1. Sell investment at a loss
2. Use losses to offset realized gains (short-term losses offset short-term gains first; same for long-term)
3. Net capital loss: deduct up to $3,000/year against ordinary income
4. Excess losses carry forward indefinitely
5. **Replace position with a similar (not "substantially identical") security** to maintain market exposure

### Wash Sale Rule

The IRS disallows the loss if you buy the **same or "substantially identical"** security within **30 days before or after** the sale.

- 61-day window total (30 days before + sale date + 30 days after)
- Applies across all your accounts (taxable, IRA, spouse's accounts)
- Buying via IRA/Roth permanently destroys the loss (cannot be re-added to basis)
- Disallowed loss is added to the basis of the replacement shares (deferred, not lost — except in IRA case)

```python
def is_wash_sale(sale_date, buy_dates, days_window=30):
    """Returns True if any buy_date is within 30 days of sale_date."""
    from datetime import timedelta
    return any(
        abs((sale_date - bd).days) <= days_window for bd in buy_dates
    )

def wash_sale_disallowed_loss(loss, shares_repurchased, shares_sold):
    """Pro-rata loss disallowance if partial replacement."""
    return loss * min(shares_repurchased, shares_sold) / shares_sold
```

### "Substantially Identical" — Gray Area

- **Clear-cut**: same CUSIP (no go)
- **Safe**: different fund families tracking different indices (VTI ↔ SCHB ↔ ITOT all track total U.S. market via slightly different methodologies)
- **Risky**: VOO ↔ IVV (both S&P 500; IRS has not definitively ruled, but most practitioners consider this risky)
- **Definitely OK**: SPY → IWB (S&P 500 → Russell 1000)

Common TLH pairs:

| Sell (loss) | Buy (replacement) |
|-------------|-------------------|
| VTI | ITOT or SCHB |
| VOO | SPLG or VV |
| QQQ | VGT or QQQM (debated) |
| VTV | VLUE |
| VEA | IEFA |
| VWO | IEMG |
| BND | AGG or SCHZ |
| Individual stock | Sector ETF for 31 days, then back |

### How Much Does TLH Add?

Studies (Vanguard, Wealthfront, Betterment) estimate **0.5–1.0% per year** in after-tax return for active TLH on a taxable portfolio with moderate volatility. Higher early in a portfolio's life (more unrealized losses available), lower as positions appreciate.

### TLH Workflow

1. **Monitor daily** for positions down ≥ 5% from cost basis
2. **Calculate after-tax benefit** vs. transaction friction
3. **Avoid wash sales** — track buys across all accounts
4. **Switch to similar but not identical** asset
5. **Wait 31 days**, then switch back if desired
6. **Track basis carefully** — TLH increases turnover; record-keeping is critical

```python
def tlh_benefit(loss_amount, marginal_tax_rate, transaction_cost=0):
    """Approximate after-tax benefit of harvesting a loss."""
    return loss_amount * marginal_tax_rate - transaction_cost
```

### When NOT to Harvest

- Loss < $200 (friction eats benefit)
- About to receive a dividend on replacement (qualified status broken if not held 60 days)
- Pushed into wash sale by 401k auto-contribution (very common pitfall)
- Loss less than expected re-entry cost (bid-ask, slippage)

## Lot Selection Methods

When selling part of a position, choose which specific shares to sell. Brokers default to **FIFO** (First In, First Out) — usually the worst choice for tax efficiency.

### Available Methods

- **FIFO** — oldest shares first (likely largest gains, highest tax)
- **LIFO** — most recent shares first (often smallest gains/losses)
- **HIFO** (Highest In, First Out) — shares with highest cost basis first (minimizes gains)
- **Specific Identification** — choose exact lots manually (maximum flexibility)
- **Average Cost** — only for mutual funds; locks you in once chosen

### Strategy

**For gains**: HIFO or spec ID to minimize realized gain.
**For losses**: HIFO realizes the largest loss.
**For long-term lots**: prefer selling lots held > 1 year for LTCG rates.
**Charitable donation**: donate highest-basis appreciated shares (best tax outcome).

```python
def optimal_lot_to_sell(lots, target_shares, objective='minimize_gain'):
    """
    lots: list of dicts with 'shares', 'cost_basis', 'purchase_date', 'current_price'
    Returns which lots to sell.
    """
    # Compute gain per share for each lot
    for lot in lots:
        lot['gain_per_share'] = lot['current_price'] - lot['cost_basis']
        lot['is_long_term'] = (today - lot['purchase_date']).days > 365
    
    if objective == 'minimize_gain':
        # Sort: long-term losses first, then long-term lowest gain, then short-term
        lots.sort(key=lambda l: (
            not l['is_long_term'],  # long-term first
            l['gain_per_share']  # lowest gain (or biggest loss) first
        ))
    elif objective == 'maximize_loss':
        lots.sort(key=lambda l: l['gain_per_share'])  # most negative first
    
    selected = []
    remaining = target_shares
    for lot in lots:
        if remaining <= 0:
            break
        take = min(lot['shares'], remaining)
        selected.append({**lot, 'shares_sold': take})
        remaining -= take
    return selected
```

## Asset Location

Place each asset in the account where it's taxed least.

### Account Tax Treatment

| Account | Tax-On-Contribution | Tax-On-Growth | Tax-On-Withdrawal |
|---------|---------------------|---------------|-------------------|
| Taxable brokerage | After-tax | Annual (gains/divs) | LTCG on sale |
| Traditional 401k/IRA | Pre-tax (deduct) | Tax-deferred | Ordinary income |
| Roth 401k/IRA | After-tax | Tax-free | Tax-free |
| HSA | Pre-tax | Tax-free | Tax-free (medical) |

### Placement Rules

| Asset | Best Account | Why |
|-------|--------------|-----|
| Tax-inefficient bonds, REITs ([[reits]]), high-turnover funds | Tax-deferred (Trad IRA/401k) | Avoid ordinary-income taxation annually |
| High-growth equities | Roth | Maximize tax-free compounding |
| Low-turnover index funds (VTI, VOO) | Taxable | Already tax-efficient; LTCG on sale |
| International equities | Taxable | Foreign tax credit only works in taxable |
| Municipal bonds | Taxable | Already tax-exempt federally; wasted in IRA |
| Crypto, MLPs, BDCs | Taxable (with caveats) | UBTI risk in IRAs |

### The "Right" Stack

```
Roth (best long-term)    → highest expected return: small caps, EM, growth
Trad IRA/401k (tax def)  → tax-inefficient: bonds, REITs, active funds
Taxable (already taxed)  → tax-efficient: index funds, munis, international
```

### Example Allocation (60% stocks, 40% bonds, three accounts equal-size)

Bad placement (everything balanced everywhere):
- Each account: 60/40 in same funds
- Bonds in Roth/taxable waste tax-advantaged space

Good placement:
- Trad IRA: 100% bonds
- Roth: 100% small-cap value
- Taxable: large-cap index + international

Net allocation: same 60/40, but bonds shielded from ordinary income tax and Roth holds highest-growth assets.

## Tax-Efficient Fund Selection

For taxable accounts:

| Tax-efficient | Tax-inefficient |
|---------------|----------------|
| Total market index funds (VTI, VOO) | Active mutual funds (high turnover) |
| ETFs over mutual funds | Mutual funds with cap gain distributions |
| Buy-and-hold individual stocks | High-turnover sector funds |
| Municipal bond funds | Taxable bond funds |
| Index funds with in-kind redemption | Actively managed equity funds |

### ETF Tax Advantage

ETFs use the "in-kind creation/redemption" mechanism to flush out low-basis lots without triggering taxable distributions to remaining holders. Mutual funds can't do this — they must sell to meet redemptions, distributing gains to all holders. Result: ETFs distribute ~0–0.5% in annual cap gains vs. 2–5% for active mutual funds.

## Mega Backdoor Roth & Backdoor Roth

### Backdoor Roth

For high earners above Roth IRA income limits ($240k single / $240k MFJ 2024):

1. Contribute non-deductible $7,000 to Traditional IRA
2. Immediately convert to Roth IRA
3. Pay tax only on any earnings (usually $0 if converted right away)

**Pro-rata pitfall**: if you have *any* pre-tax IRA balance, the conversion is taxed proportionally. Plan around this by rolling pre-tax IRAs into 401k first.

### Mega Backdoor Roth

If 401k plan allows after-tax contributions + in-service rollovers:

1. Max regular 401k ($23,000 in 2024 if under 50)
2. Max after-tax contributions (up to total 401k limit of $69,000 minus employer match)
3. Roll after-tax balance to Roth (in-service or in-plan conversion)

Adds $30,000–40,000+/year of Roth space. Not all plans allow this.

## Tax-Advantaged Account Limits (2024)

| Account | Limit | Notes |
|---------|-------|-------|
| 401k employee | $23,000 | +$7,500 catch-up if 50+ |
| 401k total | $69,000 | Inc. employer + after-tax |
| Traditional/Roth IRA | $7,000 | +$1,000 catch-up |
| HSA single / family | $4,150 / $8,300 | +$1,000 catch-up if 55+ |
| 529 (varies by state) | $18,000/year per donor | Gift tax annual exclusion |
| SEP-IRA / Solo 401k | $69,000 | Self-employed |

## Withdrawal Sequencing in Retirement

The order in which you draw down accounts in retirement dramatically affects lifetime taxes. Conventional rule: **taxable → tax-deferred → Roth**. But the optimal can differ.

### Conventional Order

1. **Taxable first** — pay LTCG on appreciation only
2. **Tax-deferred second** — taxed as ordinary income
3. **Roth last** — preserve tax-free growth longest

### Modified for Tax Efficiency

- **Fill low brackets with tax-deferred** in early retirement (before SS, RMDs)
- **Roth conversions in low-income years** (between retirement and age 73)
- **Tax-loss harvest from taxable** to offset Roth conversions
- **Time Social Security** — delay to 70 for guaranteed 8%/year COLA-adjusted increase + bracket management

### RMDs (Required Minimum Distributions)

- Starting age 73 (SECURE 2.0 Act)
- Forces taxable income from Trad IRA/401k
- Avoid pushing into higher brackets — convert to Roth before RMDs hit
- Use **Qualified Charitable Distributions (QCDs)** up to $100k/year from IRA directly to charity — counts for RMD without adding to income

```python
def rmd_amount(account_balance, age, table='uniform'):
    """
    IRS Uniform Lifetime Table (2022+) divisor approximation.
    """
    # Simplified — actual table is from IRS
    divisor_table = {
        73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9,
        78: 22.0, 79: 21.1, 80: 20.2, 85: 16.0, 90: 12.2,
    }
    divisor = divisor_table.get(age, 20.0)  # rough
    return account_balance / divisor
```

## Specific Tax Strategies

### Tax-Gain Harvesting (0% bracket)

For low-income years (early retirement, sabbatical):

- Realize LTCGs while in the 0% LTCG bracket
- Immediately repurchase (no wash sale on gains)
- Resets cost basis higher for free
- Works up to ~$47k taxable income (single)

### Net Unrealized Appreciation (NUA)

For employer stock in 401k:
- Distribute employer stock in-kind to taxable account
- Pay ordinary income tax only on cost basis
- LTCG rate on the appreciation
- Can save 10–20pp tax for highly appreciated stock

### Donor-Advised Funds (DAF)

- Contribute appreciated stock (not cash) to DAF
- Take full FMV deduction in current year
- Avoid LTCG on appreciation
- Distribute to charities over multiple years
- Strategy: "bunching" donations in high-income years; stocks with biggest unrealized gains

### Section 1031 Exchange (real estate only)

- Defer capital gains by reinvesting in like-kind property
- 45 days to identify, 180 days to close
- Only for investment real estate; not stocks

### Section 1244 (small business stock)

- Up to $100k of loss can be ordinary (not capital) if SMB stock
- Helpful if you lose money on small business equity

### QSBS (Section 1202)

- Held > 5 years, original issuance from C-corp with < $50M gross assets at issuance
- Up to $10M or 10x basis of gain **excluded from tax**
- Huge benefit for startup founders/early employees

## Crypto Tax Considerations

- Every trade is a taxable event (including crypto-to-crypto)
- Hard fork → ordinary income at receipt
- Staking rewards → ordinary income
- Mining → ordinary income (self-employment if business)
- No wash sale rule (yet) on crypto — can harvest losses and immediately rebuy (as of 2024; subject to change)
- DeFi transactions create many small taxable events

Wash-sale absence is a notable crypto edge for TLH, though legislation has been proposed repeatedly.

## State Tax Considerations

- **No income tax states**: AK, FL, NV, NH (interest/dividends taxed), SD, TN, TX, WY, WA (cap gains tax for high earners)
- **High income tax states**: CA (13.3% top), NY (10.9%), HI (11%), NJ (10.75%)
- **Municipal bonds**: federal tax-exempt; in-state munis often also state tax-exempt
- **Moving states** mid-year — partial-year residency rules vary

## NIIT and Medicare Surtax

- **3.8% NIIT** on net investment income for MAGI > $200k single / $250k MFJ
- Applies to: interest, dividends, cap gains, rental income, passive business income
- Does not apply to: muni bond interest, qualified retirement distributions
- Often the difference between 20% and 23.8% effective LTCG rate

## Estimated Taxes and Safe Harbors

If you have significant investment income, you may owe quarterly estimated taxes.

**Safe harbor**: pay either
- 100% of last year's total tax (110% if AGI > $150k), or
- 90% of current year's actual tax

Below safe harbor → underpayment penalty (~7% in 2024).

## Common Mistakes

1. **Triggering wash sales via auto-IRA contributions** — biggest TLH bust
2. **Holding REITs in taxable** — ordinary income tax wasted
3. **Selling at 364 days holding** — one more day saves 17pp tax
4. **FIFO default** when HIFO is better
5. **Donating cash instead of appreciated stock**
6. **Roth contributions over income limit** → 6% excise tax until corrected
7. **Forgetting state tax** on Roth conversions
8. **Not tracking basis** on individual stocks across multiple lots
9. **Ignoring NIIT** in tax planning
10. **Cap-gain distributions on mutual funds** in December — buy in January or use ETFs

## Software and Tools

- **TurboTax / H&R Block** — DIY tax filing
- **TaxAct** — cheaper alternative
- **Cointracker / Koinly** — crypto tax tracking
- **Wealthfront / Betterment** — automated TLH on robo platforms
- **Parametric / Aperio** — direct indexing with TLH (HNW)
- **Boldin (formerly NewRetirement)** — retirement planning with tax modeling
- **i-orp.com** — withdrawal sequencing optimizer
- **Pralana Gold** — Monte Carlo with detailed tax

## Resources

### Books
- *The Bogleheads' Guide to Retirement Planning*
- *Tax-Free Wealth* — Tom Wheelwright
- *The Power of Zero* — David McKnight
- *Live Off Dividends* — chapter on tax-efficient income

### Sites
- **Bogleheads Wiki** — tax-efficient fund placement, TLH guides
- **The Finance Buff** — backdoor Roth, MBR walkthroughs
- **Kitces.com** — detailed tax planning articles
- **IRS Publication 550** — investment income/expenses
- **IRS Publication 590-A/B** — IRA contributions/distributions

## Key Takeaways

1. **Hold > 1 year** for LTCG — saves 17pp federal for top earners
2. **TLH adds 0.5–1.0% annual after-tax return** — automate it
3. **Wash sale rule applies across accounts**, including IRAs and spouse
4. **Asset location matters as much as allocation** — bonds in 401k, growth in Roth
5. **Specific-ID over FIFO** for lot selection
6. **Roth conversions in low-income years** — pre-RMD window is gold
7. **ETFs > mutual funds** in taxable for tax efficiency
8. **Backdoor + Mega Backdoor Roth** for high earners
9. **State + NIIT** stack on federal — true marginal rates can hit 50%+
10. **Donate appreciated stock**, not cash, when possible

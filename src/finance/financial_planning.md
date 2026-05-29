# Personal Financial Planning

## Overview

Financial planning is the lifelong process of aligning your money with your goals — budgeting, debt repayment, investing, insurance, tax planning, retirement, and estate. It's less about complex tactics and more about consistent execution of a small number of high-leverage decisions: save aggressively early, invest in diversified low-cost vehicles, eliminate high-interest debt, minimize taxes (see [[tax_strategies]]), and protect against catastrophic risk. Most failures are behavioral, not analytical.

## The Planning Framework

```
1. Goals       →  what are you optimizing for?
2. Snapshot    →  net worth, cash flow
3. Foundation  →  emergency fund, insurance, no high-interest debt
4. Accumulation→  invest by account priority
5. Optimization→  tax, asset location, withdrawal order
6. Protection  →  estate plan, beneficiaries, healthcare
7. Adjust      →  annual review; life events
```

## Step 1: Set Financial Goals

Goals make every other decision concrete. SMART format: **S**pecific, **M**easurable, **A**chievable, **R**elevant, **T**ime-bound.

### Horizon Buckets

**Short-term (< 1 year):**
- Emergency fund
- Pay off credit cards
- Save for vacation / wedding / car

**Medium-term (1–5 years):**
- House down payment
- Pay off student loans
- Education funding for older children

**Long-term (5+ years):**
- Retirement
- Children's college (529s)
- Financial independence
- Estate / legacy goals

### Goal-Account Mapping

| Goal Horizon | Vehicle | Why |
|--------------|---------|-----|
| Cash needs in 1 year | HYSA, MMF, T-bills | Liquidity, no volatility |
| 1–5 years | T-bills, short-duration bonds, CDs | Modest yield, capital preservation |
| 5–10 years | Bond + stock mix (40/60) | Growth + smoothing |
| 10+ years | Equity-heavy index funds | Maximum growth, time absorbs volatility |
| Retirement (very long) | Tax-advantaged accounts first | Compounding without tax drag |

## Step 2: Assess Current Situation

### Net Worth Calculation

```python
def net_worth(assets: dict, liabilities: dict) -> dict:
    """
    assets: {'cash': X, 'investments': Y, 'home_equity': Z, ...}
    liabilities: {'mortgage': A, 'student_loans': B, 'credit_cards': C, ...}
    """
    total_assets = sum(assets.values())
    total_liabilities = sum(liabilities.values())
    return {
        'assets': total_assets,
        'liabilities': total_liabilities,
        'net_worth': total_assets - total_liabilities,
        'liquid_assets': assets.get('cash', 0) + assets.get('investments', 0),
        'debt_to_asset': total_liabilities / total_assets if total_assets else 0,
    }
```

Track quarterly. A flat or rising net-worth trend matters more than absolute level.

### Cash Flow Analysis

```python
def cash_flow_summary(income: float, expenses: dict) -> dict:
    total_exp = sum(expenses.values())
    savings = income - total_exp
    return {
        'income': income,
        'expenses': total_exp,
        'savings': savings,
        'savings_rate': savings / income if income else 0,
        'needs_pct': expenses.get('needs', 0) / income,
        'wants_pct': expenses.get('wants', 0) / income,
    }
```

A 20%+ savings rate is the floor for retirement-on-track-by-65. 40%+ enables FIRE-style early independence.

### Key Ratios

| Ratio | Target | Why |
|-------|--------|-----|
| Savings rate | 20%+ | Standard accumulation pace |
| Housing / gross income | ≤ 28% | Avoids house-poor situation |
| Total debt service / income | ≤ 36% | Standard underwriting limit |
| Emergency fund coverage | 3–6 mo | Job loss / unexpected expense buffer |
| Net worth / annual income | 1× by 30, 3× by 40, 6× by 50 | Industry rule of thumb |
| Liquid / total assets | ≥ 20% | Flexibility |

## Step 3: Create a Budget

### 50/30/20 Rule (Elizabeth Warren)

- 50% Needs (housing, food, utilities, insurance, minimum debt payments)
- 30% Wants (entertainment, dining out, hobbies, discretionary)
- 20% Savings + extra debt payment

### Zero-Based Budget (Dave Ramsey)

Income minus every expense = $0. Every dollar assigned a job. Higher friction; works for those who need explicit allocation.

### Pay-Yourself-First

- Auto-transfer savings the day paycheck arrives
- Live on what's left
- Removes willpower from the equation
- The single highest-leverage budgeting technique

### Anti-Budget

For high-income, low-discipline cases: automate savings + max accounts; spend the rest guilt-free. Works only when savings rate is already very high.

### Lifestyle Inflation

Each raise: route at least 50% to savings. Maintain prior lifestyle until raises significantly outpace inflation.

```python
def lifestyle_inflation_check(income_growth_pct, expense_growth_pct):
    """A healthy financial life has expense growth < 50% of income growth."""
    if expense_growth_pct < 0.5 * income_growth_pct:
        return 'good (banking the raise)'
    if expense_growth_pct < income_growth_pct:
        return 'caution (lifestyle inflation creeping)'
    return 'alarm (lifestyle outpacing income)'
```

## Step 4: Emergency Fund

The first investment is *not* losing money to a catastrophic event you couldn't fund.

### How Much

| Situation | Months of Expenses |
|-----------|-------------------|
| Dual income, stable jobs | 3 months |
| Single income, stable | 6 months |
| Self-employed / commission | 12 months |
| Pre-retirement / fixed cost-of-living | 24 months (in cash + ST bonds) |
| FIRE'd early retiree | 2–5 years (bond/cash bucket) |

### Where to Keep It

- **HYSA** (online banks at 4–5% APY in current environment)
- **Money market funds** (SGOV, VUSXX, SPRXX — yield Treasury rates, near-cash liquidity)
- **T-bills** (4-week, 8-week) for slightly better after-state-tax yield
- **CD ladders** for slightly higher yield with monthly liquidity

Not: stocks, long bonds, anything that can drop 20% the day you need it.

## Step 5: Debt Management

### Payoff Strategies

**Debt Avalanche (Mathematical optimum)**
- Pay minimums on all
- Apply extra to highest-rate debt first
- Save the most interest

**Debt Snowball (Behavioral)**
- Pay minimums on all
- Apply extra to smallest balance first
- Quick wins build momentum
- Costs slightly more in interest but completion rate is higher

```python
def debt_payoff_avalanche(debts, monthly_extra):
    """
    debts: list of dicts with 'balance', 'rate', 'min_payment'
    Returns months to payoff and total interest paid.
    """
    debts = sorted([dict(d) for d in debts], key=lambda d: -d['rate'])
    month = 0
    total_interest = 0
    while any(d['balance'] > 0 for d in debts):
        month += 1
        extra = monthly_extra
        for d in debts:
            if d['balance'] <= 0:
                continue
            interest = d['balance'] * d['rate'] / 12
            payment = d['min_payment']
            d['balance'] = max(0, d['balance'] + interest - payment)
            total_interest += interest
        # Allocate extra to highest-rate unpaid debt
        for d in debts:
            if d['balance'] > 0:
                d['balance'] = max(0, d['balance'] - extra)
                break
    return {'months': month, 'total_interest': total_interest}
```

### Good Debt vs. Bad Debt

| Type | Why Categorized | Action |
|------|----------------|--------|
| Mortgage (3–7%) | Asset appreciation, tax deduction | Don't rush payoff if rate < 5% |
| Student loans (federal, 3–7%) | Income-driven repayment options | Pay minimum, invest excess |
| Student loans (private, 8%+) | High rate | Refi if possible; aggressive payoff |
| Auto loans (4–10%) | Depreciating asset | Aggressive payoff if > 6% |
| Credit cards (18–28%) | Highest-rate consumption | Pay off before investing |
| Personal loans (10–25%) | Unsecured consumption | Pay off ASAP |
| HELOC (variable, currently 8–10%) | Asset-backed but variable | Refi or pay off if rates rising |
| Margin debt (broker call rate + spread) | Investment leverage | Tax-deductible if used for investing; risky |

**Rule**: if guaranteed after-tax interest rate > expected after-tax return on investment, pay off the debt. For most situations, that breakpoint is around 6–7%.

## Step 6: Retirement Planning

### How Much to Save

**Target nest egg via 4% Rule (Bengen, 1994)**:
```
Annual retirement spend = X
Target portfolio = X × 25
```

Withdraw 4% in year 1, inflation-adjust thereafter. 30-year historical success rate ~95% for 50/50 to 75/25 stock/bond mixes.

```python
def retirement_target(annual_spending, withdrawal_rate=0.04):
    return annual_spending / withdrawal_rate

def years_to_target(current_savings, monthly_contribution, target,
                   expected_real_return=0.05):
    """Future value formula solved for time."""
    import math
    monthly_r = expected_real_return / 12
    if monthly_contribution == 0:
        return math.log(target / current_savings) / math.log(1 + monthly_r)
    fv_factor = (target * monthly_r + monthly_contribution) / \
                 (current_savings * monthly_r + monthly_contribution)
    return math.log(fv_factor) / math.log(1 + monthly_r) / 12  # years
```

### Savings Rate vs. Years to Retirement

| Savings Rate | Years to FI (at 5% real return) |
|--------------|--------------------------------|
| 10% | 51 |
| 25% | 32 |
| 50% | 17 |
| 65% | 10.5 |
| 75% | 7 |

The math comes from the relationship between savings rate and required portfolio multiple — savings rate matters far more than investment returns.

### U.S. Retirement Accounts (2024 limits)

| Account | Limit | Notes |
|---------|-------|-------|
| 401(k) employee | $23,000 (+$7,500 50+) | Pre-tax or Roth |
| 401(k) total (with employer + after-tax) | $69,000 | Mega-backdoor Roth opportunity |
| Traditional / Roth IRA | $7,000 (+$1,000 50+) | Income-limited for Roth direct |
| HSA single / family | $4,150 / $8,300 (+$1,000 55+) | Triple-tax-advantaged |
| 457(b) | $23,000 | Government workers |
| SEP-IRA / Solo 401(k) | $69,000 | Self-employed |
| 529 (annual gift limit) | $18,000 / donor | $90k 5-year front-load allowed |

See [[tax_strategies]] for asset location and the backdoor / mega-backdoor Roth playbook.

### Roth vs. Traditional Decision

**Choose Roth if:**
- Currently in lower tax bracket than expected in retirement
- Younger (long time for tax-free growth)
- Already have large traditional balance
- Want estate-planning flexibility (no RMDs on Roth IRA)

**Choose Traditional if:**
- Currently in high bracket (24%+) and expect lower in retirement
- Need the deduction to maximize savings now
- State plans to leave (low- to no-tax state for retirement)

**The optimal**: usually a mix. Diversify tax exposure across pre-tax, Roth, taxable.

### Contribution Priority

1. **401(k) to employer match** — instant 50–100% return; never skip
2. **HSA max** (if HDHP eligible) — triple-tax advantage; pay medical from cash, save receipts
3. **Roth IRA** (or backdoor Roth) — flexibility, tax-free growth
4. **Remaining 401(k)** — max if income allows
5. **Mega-backdoor Roth** — if plan supports
6. **529 plans** — if college savings is a goal
7. **Taxable brokerage** — index funds, tax-efficient

### Catch-Up Contributions (50+)

- 401(k): +$7,500
- IRA: +$1,000
- HSA: +$1,000
- SECURE 2.0 introduced super catch-up at 60–63 ($11,250 starting 2025)

### Retirement Withdrawal Strategies

**Static 4% Rule (Bengen)**
- 4% year 1, CPI-adjusted thereafter
- 95%+ success rate historically over 30 years
- Conservative; often leaves large estate

**Guardrails (Guyton-Klinger)**
- Adjust withdrawals based on portfolio value vs. starting trajectory
- Cut spending after big drawdowns; raise after gains
- Supports higher initial withdrawal (~5%)

**Dynamic Spending (Vanguard)**
- Floor/ceiling: max ±5% adjustment per year
- Balance spending stability with portfolio sustainability

**Bucket Strategy**
- Bucket 1: 1–2 years cash (HYSA, T-bills)
- Bucket 2: 5–10 years bonds + bond funds
- Bucket 3: 10+ years equities
- Refill from each bucket to the next as needed
- Psychologically reassuring during bear markets

**Withdrawal Sequencing (Tax-Optimal)**

Conventional: taxable → tax-deferred → Roth. Optimized: fill low brackets with tax-deferred + Roth conversions in early retirement, before SS and RMDs kick in.

See [[tax_strategies]] for the detailed sequencing playbook.

### Social Security Strategy

- **Earliest claim**: age 62 (with 25–30% reduction)
- **Full retirement age (FRA)**: 66–67
- **Delayed credit**: +8%/year between FRA and 70 — guaranteed, CPI-adjusted
- **Spousal**: lower-earning spouse can take 50% of higher earner's benefit
- **Survivor**: surviving spouse gets higher of the two

**Default optimal**: delay to 70 if life expectancy allows (breakeven ~age 80). Run [opensocialsecurity.com](https://opensocialsecurity.com) for personalized.

### Required Minimum Distributions (RMDs)

- Start age 73 (SECURE 2.0 — 75 for those born 1960+)
- Traditional 401(k)/IRA only (not Roth IRA; Roth 401(k) RMD eliminated 2024+)
- Penalty for missing: 25% of missed amount (reduced from 50%)
- Convert to Roth in pre-RMD years to manage future tax exposure ([[tax_strategies]])

### Medicare and Healthcare

- **Age 65**: Medicare eligibility
- **IRMAA surcharges**: high income triggers premium increases (lookback: 2 years prior MAGI)
- **Bridge to 65**: ACA marketplace, COBRA, employer retiree benefits
- **Long-term care**: insurance vs. self-fund decision

## Step 7: Insurance Planning

Insurance handles low-probability, high-severity events. Don't insure what you can absorb.

### Essential

**Health insurance**
- Mandatory under most circumstances
- HDHP + HSA combo for tax efficiency
- Out-of-pocket max + emergency fund = real worst case
- Choose plan based on expected utilization, not just premium

**Term life insurance**
- Need if dependents rely on your income
- Amount: 10–15× annual income (DIME method: Debt + Income × Multiplier + Mortgage + Education)
- Term length: until dependents independent / retirement age
- Type: term, not whole life

```python
def life_insurance_dime(debt, annual_income, income_years, mortgage_balance,
                        education_costs):
    """DIME method for life insurance need."""
    return debt + annual_income * income_years + mortgage_balance + education_costs
```

**Disability insurance**
- Statistically more likely than dying during working years
- Long-term disability through employer often inadequate (taxed if employer-paid)
- Own-occupation policies for high earners
- Replace ~60–70% of income

**Auto / homeowners / renters**
- Required by law / lender
- Raise deductibles to reduce premiums (with emergency fund coverage)
- Consider replacement-cost (not actual cash value) coverage on home

**Umbrella liability**
- $1M–$5M coverage; very cheap (~$300–500/year)
- Required if net worth > coverage limits of underlying policies
- Litigation-prone professions (doctors, executives, landlords)

### Long-Term Care Insurance

- Probability of LTC need: ~70% for 65-year-olds at some point
- Cost: $80–120k/year (nursing home), $50–70k (in-home)
- Options: traditional LTC, hybrid life/LTC, self-insure
- Buy in 50s while still healthy and underwriting favorable

### Insurance to Skip

- Whole / universal life (high fees; bundle better unbundled)
- Mortgage insurance (term life is cheaper for same coverage)
- Cancer / specific disease policies (covered by health insurance)
- Credit card balance insurance
- Extended warranties on appliances/electronics
- Air travel insurance (rarely covered events)

## Step 8: Tax Planning

The deepest treatment lives in [[tax_strategies]]. Key concepts:

- **Marginal vs. effective rate**: marginal = rate on next dollar; effective = average across all income
- **Tax-advantaged contributions** reduce current taxable income
- **Tax-loss harvesting** in taxable accounts (0.5–1% annual after-tax boost)
- **Asset location** — tax-inefficient assets in tax-advantaged accounts
- **Roth conversions** in low-income gap years
- **QCDs** (Qualified Charitable Distributions) for RMDs at 70½+
- **Donor-advised funds** for bunching charitable giving

### 2024 Federal Brackets (Single Filer)

| Rate | Income |
|------|--------|
| 10% | $0 – $11,600 |
| 12% | $11,600 – $47,150 |
| 22% | $47,150 – $100,525 |
| 24% | $100,525 – $191,950 |
| 32% | $191,950 – $243,725 |
| 35% | $243,725 – $609,350 |
| 37% | $609,350+ |

Plus 3.8% NIIT, 0.9% Medicare surtax, state income tax. Top effective rate in CA exceeds 50%.

## Step 9: College Savings

### 529 Plans

- Tax-free growth for qualified education expenses
- State tax deduction in many states (varies)
- $18k/donor annual contribution gift limit (or $90k 5-year front-load)
- Beneficiary can be changed (siblings, cousins, even self)
- SECURE 2.0: up to $35k lifetime can roll to Roth IRA for beneficiary (after 15+ years)

### Alternatives

- **Coverdell ESA**: $2k/yr limit; K–12 eligible; investment flexibility
- **UTMA/UGMA**: custodial; not just education; becomes child's at age of majority (no tax benefit)
- **Roth IRA**: contributions can be withdrawn for any reason; earnings for qualified ed without penalty (taxed if not 59½)

### Principle

"You can borrow for college, not for retirement." Prioritize retirement first. A wealthy retired parent helps adult children more than a broke retiree with degree-holding kids.

## Step 10: Estate Planning

### Essential Documents

**Will**
- Distribute assets, name executor, guardianship for minor kids
- Update on major life events (marriage, divorce, kids, death of beneficiary)
- Without one: state intestacy law applies (rarely what you'd want)

**Powers of Attorney**
- **Financial POA**: manage finances if incapacitated
- **Healthcare POA**: make medical decisions

**Living Will / Advance Directive**
- End-of-life care preferences
- DNR / DNI specifications

**Beneficiary Designations**
- Retirement accounts, life insurance, TOD on brokerage, POD on bank
- Beneficiary designations OVERRIDE wills
- Check annually; update on life events

**HIPAA Authorization**
- Allows family access to medical info

### Trusts

| Type | Use |
|------|-----|
| Revocable Living Trust | Avoid probate, smooth asset transfer |
| Irrevocable Trust | Estate tax planning, asset protection |
| Special Needs Trust | Provide for disabled beneficiary without disqualifying benefits |
| Charitable Remainder Trust | Income for life + remainder to charity, tax deduction |
| ILIT (Irrevocable Life Insurance Trust) | Keep life insurance out of estate |
| QPRT (Qualified Personal Residence Trust) | Transfer home at reduced gift value |

### Estate Tax (2024)

- **Federal exemption**: $13.61M / person ($27.22M MFJ) — set to halve in 2026 absent legislation
- **Rate above exemption**: 40%
- **Annual gift exclusion**: $18,000 / donor / recipient (no lifetime cap effect)
- **State estate taxes**: 12 states + DC (varies: OR/WA exempt $1M, CT/NY $13M+)
- **Step-up in basis at death**: assets get FMV basis (eliminates LTCG on heirs)

### Estate-Planning Strategies

- **Annual gifting** to family ($18k/donor/recipient) — drains future estate over time
- **529 superfunding**: 5 years of gift exclusion to grandchild's 529 ($90k) in one year
- **Roth conversions** during life: shifts tax burden off heirs (no RMDs for surviving spouse on inherited Roth IRA; non-spouse 10-year rule applies)
- **Charitable giving via DAF**: bunched in high-income years
- **Step-up basis advantage**: hold highly appreciated assets to death rather than sell during life
- **Spousal portability**: surviving spouse can use both exemptions ($27.22M combined)

## Financial Independence / FIRE

### Definitions

- **Lean FIRE**: $25–40k/year, ~$625k–1M portfolio
- **Regular FIRE**: $40–80k/year, ~$1M–2M portfolio
- **Fat FIRE**: $100k+/year, ~$2.5M+ portfolio
- **Coast FIRE**: enough saved that growth alone (no further contributions) reaches retirement target by 65
- **Barista FIRE**: portfolio + part-time work covers expenses
- **Geo-arbitrage FIRE**: earn in HCOL, retire in LCOL

### Coast FIRE Math

```python
def coast_fire_number(current_age, retire_age, annual_expense,
                      real_return=0.05, withdrawal_rate=0.04):
    """How much do you need TODAY for it to grow to retirement target?"""
    target = annual_expense / withdrawal_rate
    years = retire_age - current_age
    return target / (1 + real_return) ** years
```

E.g., to retire at 65 needing $80k/year, with 5% real return:
- Target nest egg: $2M
- Coast number at age 30: $363k — invest no more, and you're set by 65
- Coast number at age 40: $592k

### Common FIRE Pitfalls

- Underestimating healthcare costs pre-65 (ACA subsidies depend on AGI)
- Sequence-of-returns risk in early retirement years
- Lifestyle inflation creeping back
- Boredom / loss of identity
- Tax inefficient early withdrawals (use Roth contribution withdrawals, 72(t) SEPP, or Roth conversion ladder)

### Roth Conversion Ladder

Classic early-retirement tactic:
- Year N: convert tradiitional to Roth (taxable in year of conversion)
- Year N+5: withdraw converted amount tax + penalty-free
- Repeat annually to build a 5-year-out pipeline

## Sequence of Returns Risk

The order of returns matters when you're contributing or withdrawing. A bear market early in retirement is far more damaging than late, even with identical average returns.

### Mitigation

- Larger cash buffer entering retirement (2+ years)
- Bond tent: gradually increase bonds approaching retirement, then de-risk after the first 5–10 years
- Don't sell equities in bear markets — spend from bonds/cash
- Variable withdrawal (cut spending after big drops)
- Delay SS to reduce portfolio reliance

## Financial Milestones by Age (Rough Guide)

| Age | Net Worth (× salary) | Key Actions |
|-----|---------------------|-------------|
| 25 | 0.5× | Emergency fund, start 401(k), pay off CC debt |
| 30 | 1× | Max 401(k) to match, max Roth IRA |
| 35 | 2× | Increase savings rate, consider home, term life if dependents |
| 40 | 3× | Max 401(k), backdoor Roth, college savings |
| 45 | 4–5× | Disability + umbrella insurance, mid-career planning |
| 50 | 6–7× | Catch-up contributions, LTC insurance evaluation |
| 55 | 8–9× | Estate plan refresh, retirement budget modeling |
| 60 | 10–11× | Bond tent build, asset location optimization |
| 65 | 12–14× | Medicare, SS decision, withdrawal strategy |
| 70 | 14× + | SS at 70, RMD planning, Roth conversion completion |

These are aggressive; ~50th percentile retirees have far less, but those are the people often working into 70s.

## Common Mistakes

1. **No emergency fund** — first setback turns into debt spiral
2. **High-interest debt** — paying 20% credit card interest while investing
3. **Lifestyle inflation** — raises silently absorbed into spending
4. **Not starting early** — every decade lost halves the compounding
5. **Ignoring employer match** — leaving free money on the table
6. **Whole life insurance** — sold by commissioned agents; rarely optimal
7. **AUM-fee advisors** — 1%/year compounds to 28% of portfolio over 30 years
8. **Timing the market** — missing the 10 best days drops 30-year returns by ~50%
9. **Tax-inefficient choices** — bonds in taxable, REITs in taxable, no TLH
10. **No estate plan** — dying intestate creates lengthy expensive probate
11. **Sequence risk denial** — retiring at peak with no bond buffer
12. **Insurance-as-investment confusion** — bundling rarely beats unbundled
13. **No tax planning around Roth conversions** in low-income gap years
14. **Children's college over retirement** — borrowing impossible for retirement
15. **Concentration risk in employer stock** — Enron / Lehman / SVB lessons

## Best Practices

1. **Pay yourself first** — automate before you see it
2. **Live below your means** — savings rate is the prime mover
3. **Maximize tax-advantaged accounts** — match → HSA → Roth → 401(k)
4. **Keep investing simple** — three-fund portfolio beats 95% of professionals
5. **Bank the raise** — 50%+ of raises to savings
6. **Annual review** — life changes; plan adjusts
7. **Avoid fees ruthlessly** — every basis point compounds
8. **Stay educated** — but ignore most financial media
9. **Don't compare** — different starting points, different goals
10. **Balance** — money is a means; enjoy life along the way

## Tools and Software

| Need | Tool |
|------|------|
| Budgeting | YNAB, Monarch, Copilot, Tiller |
| Net worth tracking | Empower (Personal Capital), Monarch |
| Retirement projection | Boldin (NewRetirement), Pralana, ProjectionLab |
| Investment platform | Fidelity, Schwab, Vanguard, IBKR |
| Tax filing | TurboTax, H&R Block, FreeTaxUSA |
| SS optimization | OpenSocialSecurity.com (free) |
| Withdrawal sequencing | i-orp.com, Boldin |

## Resources

### Books
- *The Simple Path to Wealth* — JL Collins
- *Your Money or Your Life* — Vicki Robin, Joe Dominguez
- *The Millionaire Next Door* — Stanley & Danko
- *I Will Teach You to Be Rich* — Ramit Sethi
- *The Psychology of Money* — Morgan Housel
- *Die With Zero* — Bill Perkins
- *The Bogleheads' Guide to Investing* / *Retirement Planning*
- *How Much Money Do I Need to Retire?* — Todd Tresidder
- *The Wealthy Barber* — David Chilton

### Communities
- **Bogleheads.org wiki and forum** — gold standard for U.S. investing
- **r/personalfinance** — beginner Q&A
- **r/financialindependence** — FIRE strategy
- **r/Bogleheads** — index investing discussion
- **r/tax** — tax planning Q&A
- **Mr. Money Mustache blog** — frugal-aggressive FIRE

### Newsletters / Sites
- **Kitces.com** — practitioner-level advanced planning
- **The Finance Buff** — Backdoor / Mega Backdoor / Roth conversion specifics
- **Of Dollars and Data** (Nick Maggiulli)
- **Morningstar Retirement** — withdrawal strategies, fund research

## Key Takeaways

1. **Start now** — time is the most expensive thing to waste
2. **Savings rate dominates returns** — for the first 10–20 years of accumulation
3. **Emergency fund is the foundation** — investing comes after, not before
4. **Pay off high-interest debt first** — 18% guaranteed return is rare
5. **Employer match + HSA + Roth IRA** — fill these before other investing
6. **Automate everything** — willpower is finite
7. **Three-fund portfolio is sufficient** — complexity rarely improves outcomes
8. **Tax planning compounds** — see [[tax_strategies]] for the asymmetric wins
9. **Insurance is risk transfer, not investing** — separate the two
10. **Estate planning is for your family, not you** — keep it current

See also: [[portfolio_management]], [[risk_management]], [[tax_strategies]], [[interest_rates]], [[private_markets]] (for accredited investors), [[reits]]

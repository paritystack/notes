# Derivatives

## Overview

A derivative is a financial contract whose value is *derived* from an underlying — a stock, bond, rate, currency, commodity, index, or even another derivative. Derivatives are among the most powerful instruments in finance: they let you hedge precise risks, gain leveraged exposure, replicate payoffs that don't exist as cash instruments, and arbitrage between markets. They also concentrate risk in ways that have produced the largest blowups in history (LTCM 1998, AIG 2008, Archegos 2021).

The global notional value of outstanding OTC derivatives is over $700 trillion. Most of this is interest rate swaps. Options and futures are covered in dedicated guides ([[options]], [[futures]]); this file focuses on swaps, forwards, exotics, and the structural concepts that tie all derivatives together.

## The Two Big Categories

| Category | Examples | Venue | Counterparty Risk |
|----------|----------|-------|-------------------|
| **Exchange-traded** | Futures, listed options | CME, ICE, CBOE | Cleared by exchange |
| **OTC (Over-the-Counter)** | Swaps, forwards, exotics | Bilateral | Direct (unless cleared) |

Post-2008 regulation (Dodd-Frank, EMIR) pushed most standardized OTC derivatives into central clearing. Bespoke and exotic deals remain bilateral.

## Forwards

### What They Are

A forward is a private contract to buy or sell an asset at a future date for a price agreed today. The simplest derivative.

- **Buyer (long)**: agrees to buy at strike K
- **Seller (short)**: agrees to sell at strike K
- **Payoff at maturity** = S_T − K (long) or K − S_T (short)
- No cash changes hands at inception (usually)
- Settled at maturity (physical delivery or cash)

### Forward vs. Futures

| Feature | Forward | Future |
|---------|---------|--------|
| Trading venue | OTC | Exchange |
| Customization | Full | Standardized |
| Settlement | Maturity only | Daily mark-to-market |
| Counterparty | Bilateral | Clearinghouse |
| Margin | Negotiated/none | Required, daily |
| Liquidity | Lower | Higher |

For deep coverage of futures, see [[futures]].

### Forward Pricing (Cost-of-Carry)

The no-arbitrage forward price for an asset paying no income:

```
F = S × e^(r × T)
```

For an asset with continuous yield `q` (e.g., dividend yield, foreign rate):

```
F = S × e^((r − q) × T)
```

For an asset with storage costs `u` minus convenience yield `y`:

```
F = S × e^((r + u − y) × T)
```

```python
import math

def forward_price(spot, risk_free_rate, time_to_maturity, dividend_yield=0,
                  storage_cost=0, convenience_yield=0):
    """
    Generic cost-of-carry forward price (continuous compounding).
    """
    carry = risk_free_rate - dividend_yield + storage_cost - convenience_yield
    return spot * math.exp(carry * time_to_maturity)

def forward_value(current_forward_price, original_forward_price,
                  risk_free_rate, time_to_maturity):
    """PV of an existing forward contract (long position)."""
    return (current_forward_price - original_forward_price) * \
           math.exp(-risk_free_rate * time_to_maturity)
```

### FX Forwards (Covered Interest Parity)

For FX, the forward price is set by interest-rate differentials:

```
F = S × (1 + r_domestic × T) / (1 + r_foreign × T)
```

This is **covered interest parity (CIP)**. When CIP breaks (it has, notably 2008 and 2020), arbitrageurs profit until balance restores. See [[forex]] for currency mechanics.

```python
def fx_forward(spot_fx, r_domestic, r_foreign, time_to_maturity):
    """FX forward via covered interest parity (annualized rates)."""
    return spot_fx * (1 + r_domestic * time_to_maturity) / \
                     (1 + r_foreign * time_to_maturity)
```

### Forward Rate Agreements (FRAs)

A forward on an interest rate. Buyer locks a future borrowing rate; seller locks a lending rate.

- Notional: $10M
- Period: 3-month rate starting in 6 months ("6×9 FRA")
- Strike: 4.5%
- Reference rate at settlement: SOFR fixing

If actual 3-month rate at settlement is 5.0%:
- Buyer (long, locked in 4.5%) wins: receives 50bp × $10M × 90/360 = $12,500

FRAs are settled at the **start** of the period (discounted for the term).

```python
def fra_payoff(notional, fixed_rate, floating_rate, day_count_frac):
    """Settled at start of period, present-valued back."""
    interest_diff = (floating_rate - fixed_rate) * day_count_frac * notional
    pv_factor = 1 / (1 + floating_rate * day_count_frac)
    return interest_diff * pv_factor
```

## Swaps

A swap is a series of exchanges of cash flows between two parties over time. Mechanically, a strip of forwards.

### Interest Rate Swap (IRS)

The most important derivative on Earth — $300T+ notional outstanding.

**Plain vanilla IRS**:
- Party A pays fixed rate (e.g., 4.5%)
- Party B pays floating rate (e.g., SOFR + 10bp)
- Notional never exchanged — just net cash flows
- Standard tenors: 2, 5, 10, 30 years

#### Mechanics

```
Notional: $100M
Tenor: 5 years
Fixed leg: 4.50% semi-annual
Floating leg: SOFR quarterly

Every 6 months:
  Fixed pays: $100M × 4.50% / 2 = $2.25M
Every 3 months:
  Float pays: $100M × SOFR × 0.25
  
Net = exchange (netted to single payment)
```

#### Why Use IRS

- **Liability management**: corporation issued fixed bonds, wants floating exposure → "pay floating / receive fixed"
- **Asset management**: bank receiving floating loan income, wants stable revenue → "pay floating / receive fixed"
- **Speculation**: betting rates rise → "pay fixed / receive floating"
- **Hedge bond portfolio duration**: shorter or longer duration via swaps without selling bonds

#### Pricing

The fair swap rate makes the PV of fixed leg equal PV of expected floating leg. Built from the swap curve (bootstrapped from observed swap rates).

```python
def swap_pv_fixed_leg(notional, fixed_rate, year_fractions, discount_factors):
    """PV of fixed leg cash flows."""
    return sum(notional * fixed_rate * yf * df
               for yf, df in zip(year_fractions, discount_factors))

def swap_par_rate(notional, year_fractions, discount_factors,
                   forward_rates):
    """
    Par swap rate: makes both legs PV equal.
    """
    pv_float = sum(notional * fr * yf * df
                   for fr, yf, df in zip(forward_rates, year_fractions,
                                          discount_factors))
    annuity = sum(yf * df for yf, df in zip(year_fractions, discount_factors))
    return pv_float / (notional * annuity)
```

#### Risks

- **Rate risk** — duration-equivalent exposure
- **Counterparty risk** — historically the major concern; now mitigated by clearing
- **Basis risk** — when floating reference doesn't match underlying exposure
- **Curve risk** — different points of curve move differently

### Currency Swap (XCS / Cross-Currency Swap)

Exchange principal + interest in different currencies. Unlike IRS, principal IS exchanged at start and end.

```
Day 0: 
  Party A gives $100M USD to Party B
  Party B gives €90M EUR to Party A
  
Quarterly:
  A pays B EUR interest on €90M
  B pays A USD interest on $100M
  
Maturity:
  A returns €90M to B
  B returns $100M to A
```

#### Use Cases

- Foreign company issues USD bonds → swap proceeds + payments back to home currency
- Bank wants synthetic foreign-currency funding
- Hedging multi-year FX exposure

### Equity Swap / Total Return Swap (TRS)

Exchange equity return for a financing rate.

```
Party A pays SOFR + spread
Party B pays total return on equity (price + dividends)
```

#### Why Used

- **Hidden long exposure** — don't appear on 13F (Archegos)
- **Tax-efficient ownership** transfer
- **Avoid disclosure thresholds** (HSR, 5% owner)
- **Leverage** — much higher than prime brokerage margin
- **Short access** — synthetic shorts on hard-to-borrow names

#### Archegos Blowup (2021)

Bill Hwang's family office held ~$100B notional through TRS at multiple prime brokers. None saw the aggregate exposure. When ViacomCBS, Discovery, and Chinese tech stocks fell, margin calls cascaded. Credit Suisse alone lost $5.5B; Nomura ~$3B. Largest single-counterparty loss for a prime broker since LTCM.

### Commodity Swap

Exchange floating commodity price for fixed price, cash-settled.

```
Airline hedges jet fuel:
  Pays fixed $90/bbl
  Receives floating spot price
  
If spot averages $110: airline receives $20/bbl × notional
```

Used heavily by airlines, utilities, shipping, energy producers.

### Variance Swap and Volatility Swap

Pure exposure to realized volatility.

```
Variance swap payoff = (Realized Variance − Strike Variance) × Vega Notional

Volatility swap payoff = (Realized Vol − Strike Vol) × Vega Notional
```

Used by:
- Hedging vol exposure precisely
- Speculating on volatility levels
- Trading the variance risk premium (see [[volatility_trading]])

Variance swaps were popular pre-2008 but never recovered to their previous size. VIX futures filled much of the demand.

### Credit Default Swap (CDS)

Insurance against default. Covered in depth in [[credit_markets]].

- **Buyer of protection**: pays premium, gets paid if default
- **Seller of protection**: collects premium, pays out on default
- **Reference entity**: company/sovereign being insured
- **Trigger events**: bankruptcy, failure to pay, restructuring

CDS spreads are often the cleanest market signal of credit risk — more liquid than the underlying bonds for many large issuers.

### Inflation Swap

Exchange fixed rate for realized inflation (CPI year-over-year). Used by pension funds, insurers, sovereigns to hedge real liabilities.

The market-implied breakeven inflation = fixed rate side. Track this for inflation expectations — equivalent to the TIPS breakeven, but cleaner. See [[interest_rates]].

## Exotic Options

Beyond plain vanilla calls and puts ([[options]]), exotic options have non-standard payoffs.

### Barrier Options

Activated or deactivated when underlying crosses a barrier level.

| Type | Behavior |
|------|----------|
| **Up-and-in** | Call/put that activates when price rises above barrier |
| **Up-and-out** | Knocks out (becomes worthless) when price rises above barrier |
| **Down-and-in** | Activates on price falling below barrier |
| **Down-and-out** | Knocks out when price falls below barrier |

Cheaper than vanilla options (give up some payoff). Used by structurers and hedgers.

```python
def knock_out_check(price_path, barrier, direction='down'):
    """Returns True if barrier was breached during the path."""
    if direction == 'down':
        return any(p <= barrier for p in price_path)
    return any(p >= barrier for p in price_path)
```

### Asian Options

Payoff based on **average price** over the observation period, not just final price.

- Average price call: max(0, Avg(S) − K)
- Average strike call: max(0, S_T − Avg(S))

Cheaper than vanilla (less volatility in average vs. spot). Common in commodities and FX hedging.

### Binary / Digital Options

All-or-nothing payoff:
- Pays fixed amount if condition met (e.g., S_T > K)
- Pays zero otherwise

Used in structured products, FX options, and as building blocks. Retail "binary options" platforms (often offshore, unregulated) have been associated with widespread fraud.

### Lookback Options

Payoff based on extreme prices during the option's life:
- Floating-strike lookback call: max(0, S_T − min(S))
- Floating-strike lookback put: max(0, max(S) − S_T)

Always end up in the money (if any movement). Expensive premium.

### Cliquet (Ratchet) Options

Series of forward-starting options. Each period locks in the gain and resets the strike. Used in equity-linked structured products.

### Rainbow Options

Payoff depends on multiple underlyings:
- "Best of": pays the best-performing asset
- "Worst of": pays the worst-performing asset
- "Spread": pays difference between two assets

Common in structured notes and currency strategies.

### Quanto Options

Payoff in one currency based on underlying in another, with FX risk hedged out at a fixed exchange rate. Used by foreign investors wanting clean exposure without FX risk.

### Compound Options

Options on options. Pay premium today for the right to buy an option later. Convex pricing.

## Structured Products

Pre-packaged combinations of bonds + derivatives, sold to retail and institutions.

### Principal-Protected Notes (PPN)

- Zero-coupon bond + call option
- Protects principal at maturity
- Capped upside
- Lookbacks like 5-10 years
- **High fees, often poor value vs. DIY** (buy zero + LEAPS yourself)

### Autocallable Notes

- Periodic check (e.g., quarterly): if underlying ≥ initial, redeem early with coupon
- If never redeemed, principal at risk based on underlying performance
- High coupons (8–12%) until autocalled
- Sold in $billions in Europe and Asia
- Risk: tail event triggers loss of principal

### Reverse Convertibles

- Bond paying high coupon
- Buyer is short an OTM put on the underlying
- If put expires OTM: get principal back
- If ITM: receive shares (usually at a loss)
- Effectively selling vol for high yield ([[volatility_trading]])

### Equity-Linked Notes (ELN)

- Return tied to an equity index
- Many flavors: capped, floored, leveraged, basket

### Snowballs and Phoenix Autocallables

Complex Asia-prevalent structures. Snowball notes wiped out billions in China in 2024 when CSI 500 fell sharply.

## Pricing Principles

### No Arbitrage

If two portfolios produce identical cash flows in all states, they must cost the same. Violation = riskless profit (arbitrage), which gets traded away.

### Replication

Most derivative pricing comes from constructing a replicating portfolio of simpler instruments:
- Forward = spot + cash position
- European call = (BSM hedging strategy with stock + cash)
- Swap = strip of forwards

### Risk-Neutral Pricing

Discount expected payoffs at the risk-free rate under a synthetic "risk-neutral" probability measure. The math is convenient — though it's not how real probabilities work.

### Black-Scholes for Options

Vanilla European option pricing covered in [[options]]. Foundation for most exotic pricing too.

### Monte Carlo Simulation

For path-dependent or multi-asset derivatives, simulate thousands of price paths and average payoffs.

```python
import numpy as np

def monte_carlo_european_call(spot, strike, risk_free, vol, time, n_paths=100_000):
    """Monte Carlo price of European call (sanity check vs. Black-Scholes)."""
    z = np.random.standard_normal(n_paths)
    s_t = spot * np.exp((risk_free - 0.5 * vol**2) * time + vol * np.sqrt(time) * z)
    payoffs = np.maximum(s_t - strike, 0)
    return np.exp(-risk_free * time) * payoffs.mean()

def monte_carlo_barrier_call(spot, strike, barrier, risk_free, vol, time,
                              n_paths=100_000, n_steps=252, barrier_type='up_out'):
    """Monte Carlo price of up-and-out call (path-dependent)."""
    dt = time / n_steps
    z = np.random.standard_normal((n_paths, n_steps))
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = spot
    for t in range(n_steps):
        paths[:, t+1] = paths[:, t] * np.exp(
            (risk_free - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z[:, t]
        )
    # Knock-out condition
    knocked_out = (paths.max(axis=1) >= barrier)
    payoffs = np.where(knocked_out, 0, np.maximum(paths[:, -1] - strike, 0))
    return np.exp(-risk_free * time) * payoffs.mean()
```

## Risks of Derivatives

### Market Risk

Same as the underlying, but often levered. A futures position with 10x leverage moves your P&L 10x faster.

### Counterparty Risk (OTC)

If the other side defaults, you may not get paid. Mitigations:
- **Central clearing** (CCPs like LCH, CME, ICE)
- **Collateral / variation margin** posted daily
- **Initial margin** (independent collateral)
- **ISDA Master Agreements** with CSA (Credit Support Annex)

### Basis Risk

When your hedge instrument doesn't perfectly match your exposure. Examples:
- Airline hedges jet fuel with heating oil futures
- Bank hedges FRN with swap of slightly different reference rate
- Equity index hedge for portfolio that's not exactly index-weighted

### Liquidity Risk

OTC contracts can be hard to unwind. Even cleared products can gap in stress. The bid-ask widens dramatically in exotic instruments.

### Model Risk

Exotic pricing depends on models. If your model assumptions break (e.g., constant vol assumption in BSM), prices may be wrong. Famous example: LTCM's risk models assumed normal distributions; tail events killed them.

### Operational and Legal Risk

Complex documentation, settlement chains, regulatory compliance, novation issues. Big OTC books require entire operations teams.

### Wrong-Way Risk

Counterparty credit risk and exposure correlate. E.g., AIG was selling massive CDS protection on subprime MBS — exactly when subprime defaulted (creating exposure), AIG itself was insolvent (counterparty failure).

## Regulatory Landscape (Post-2008)

### Dodd-Frank (U.S.) and EMIR (EU)

Major changes:
- **Central clearing mandate** for standardized OTC derivatives (most swaps)
- **Trade reporting** to swap data repositories
- **Margin requirements** for non-cleared trades
- **Push-out rule** for certain bank derivatives (later softened)
- **Volcker Rule** restricts proprietary trading

### Central Counterparties (CCPs)

LCH (UK), CME (U.S.), Eurex (Germany), ICE Clear — interpose themselves between counterparties. Risk concentrated in CCPs is now itself systemic.

### ISDA Documentation

The International Swaps and Derivatives Association master agreement is the legal backbone of OTC derivatives. Specifies events of default, close-out netting, governing law. Every serious derivatives counterparty has ISDA agreements.

## Notable Derivatives Blowups

| Year | Entity | Loss | Cause |
|------|--------|------|-------|
| 1994 | Procter & Gamble | $157M | Exotic leveraged swaps mis-sold by Bankers Trust |
| 1995 | Barings Bank | £827M | Nick Leeson Nikkei futures, $1.3B loss → bankruptcy |
| 1998 | LTCM | $4.6B | Convergence trades on massive leverage |
| 2001 | Enron | bankruptcy | Hidden derivatives liabilities |
| 2008 | AIG | $182B bailout | CDS protection sold without capital |
| 2008 | Lehman | bankruptcy | Counterparty defaults cascade |
| 2012 | JPM London Whale | $6.2B | Outsized credit index positions |
| 2015 | Swiss National Bank | major losses by clients | Removed EUR/CHF floor; $20B in retail FX losses |
| 2018 | XIV | wiped out overnight | Short vol ETN |
| 2021 | Archegos | $10B+ across banks | Hidden equity TRS exposure |
| 2024 | China Snowballs | tens of billions | Autocallable note triggers |

Theme: most blowups involved hidden leverage, model failures, or counterparty cascades.

## Hedging Use Cases

### Corporate Hedging

- **Airline**: hedge jet fuel via commodity swap
- **Exporter**: hedge foreign revenue via FX forwards
- **REIT** ([[reits]]): hedge floating-rate debt via IRS (pay fixed)
- **Bank**: hedge loan portfolio duration via swaps

### Institutional Hedging

- **Pension**: LDI (Liability-Driven Investing) — match duration with long swaps
- **Insurance**: hedge variable annuity guarantees with options
- **Endowment**: hedge equity exposure with index puts

### Investor Hedging

- **Portfolio insurance**: index puts on equity exposure
- **Currency hedging**: forwards on international stocks
- **Interest rate hedge**: TLT puts when long bonds
- **Tail hedge**: OTM puts for crash protection (see [[volatility_trading]])

## Speculation Use Cases

Beyond hedging, derivatives concentrate directional bets:

- **Leveraged directional**: futures (10–50x leverage on margin)
- **Volatility trading**: variance swaps, straddles ([[volatility_trading]])
- **Curve trades**: swap spread, butterflies on rates ([[interest_rates]])
- **Spread trades**: pairs via TRS ([[pairs_mean_reversion]])
- **Credit spread bets**: CDS, index CDX

## Practical Considerations

### When to Use What

| Need | Best Tool |
|------|-----------|
| Hedge an existing exposure | Forward, swap, option |
| Leveraged directional bet | Future, option |
| Tail protection | OTM put, VIX call |
| Steady income from vol | Short strangle/condor |
| Custom payoff | Exotic option, structured note |
| Lock in future borrowing rate | FRA, swap |
| Synthetic short hard-to-borrow | TRS |

### DIY vs. Structured Products

Most structured products can be replicated with bonds + listed options at lower cost. Always check: would buying the components yourself cost less? Usually yes (the fee differential is 2–5% upfront on retail-sold structures).

## Common Mistakes

1. **Using exotics when vanilla works** — added complexity rarely justifies the cost
2. **Ignoring counterparty risk** — bilateral OTC means you can be left holding the bag
3. **Underestimating leverage** — derivatives amplify both ways
4. **Naked short options on tail risk** — many small wins, occasional catastrophic loss
5. **Trusting model prices in stressed markets** — Gaussian assumptions break in crises
6. **Forgetting margin requirements** — can balloon overnight (Volmageddon 2018)
7. **Concentrating across counterparties** — diversify even when cleared
8. **Buying retail structured products** at high markups
9. **Not understanding the embedded derivatives** in your portfolio (bond calls, autocalls)
10. **Hedging the wrong risk** — basis risk can leave you exposed despite the hedge

## Resources

### Books
- *Options, Futures, and Other Derivatives* — John Hull (the standard text)
- *Dynamic Hedging* — Nassim Taleb (practitioner perspective)
- *The Volatility Smile* — Emanuel Derman & Michael Miller
- *FX Options and Smile Risk* — Antonio Castagna
- *Interest Rate Models* — Damiano Brigo, Fabio Mercurio
- *When Genius Failed* — Roger Lowenstein (LTCM)
- *The Big Short* — Michael Lewis (CDS/CDO)
- *Liar's Poker* — Michael Lewis (mortgage derivatives)

### Sites and Data
- **ISDA** — documentation, market stats
- **DTCC** — swap data repository
- **BIS triennial survey** — global OTC derivatives statistics
- **CME, ICE, Eurex** — exchange-traded data
- **OpenGamma**, **QuantLib** — open-source pricing libraries

## Key Takeaways

1. **Derivatives are tools, not strategies** — what matters is what you do with them
2. **No-arbitrage and replication** underpin all pricing
3. **Forwards are the building block** — swaps are strips, options add convexity
4. **Interest rate swaps dominate** — $300T+ notional; the largest derivative market
5. **Central clearing reduced counterparty risk** post-2008 but concentrated it in CCPs
6. **Exotics cost more in fees + model risk** — usually replicable with vanillas
7. **Structured products often poor value** for retail — DIY replication beats the fee
8. **TRS hide exposure** — Archegos showed why broker aggregation matters
9. **Hedging requires basis discipline** — perfect hedges are rare
10. **Tail risk asymmetric** — selling tail premium is profitable until it isn't

## See Also

- [[options]] — vanilla call/put pricing and strategies
- [[futures]] — exchange-traded forwards
- [[interest_rates]] — yield curve, Fed, duration
- [[credit_markets]] — CDS in depth, structured credit
- [[volatility_trading]] — variance swaps, vol selling
- [[forex]] — FX forwards and swaps
- [[commodities]] — commodity swaps and futures
- [[risk_management]] — hedging mechanics, portfolio risk
- [[private_markets]] — TRS in private fund structures

## Where this connects

- [Options](options.md) — options are the most common derivative for retail and institutional traders
- [Volatility trading](volatility_trading.md) — derivative volatility surface drives vol trading
- [Interest rates](interest_rates.md) — interest rate derivatives (swaps, caps, floors) hedge rate risk
- [Credit markets](credit_markets.md) — credit default swaps (CDS) are credit derivatives

# Options

## Overview

An option is a contract that gives the buyer the **right, but not the obligation**, to buy
(a **call**) or sell (a **put**) an underlying asset at a fixed **strike price** before or
at expiration. That asymmetry is the whole point: unlike [futures](futures.md), where both
sides are obligated and the payoff is linear, an option buyer's loss is capped at the
**premium** paid while the upside stays open — the payoff is *kinked* at the strike. Options
are a category of [derivatives](derivatives.md) and the primary instrument for trading
[volatility](volatility_trading.md), hedging [portfolios](portfolio_management.md), and
shaping the risk of a [stock](stocks.md) position.

```
   LONG CALL payoff at expiry           LONG PUT payoff at expiry
 P&L                                   P&L
   │            ___                       │___
   │          _/                          │   \__
 0 │────────/──────── S                  0│──────\________ S
   │   −premium  K                        │   K   −premium
   │  (loss capped)                       │  (loss capped)
   right to BUY at K                      right to SELL at K
```

This page covers option pricing (Black-Scholes, the Greeks), the mechanics of premium
(intrinsic vs time value, moneyness, put-call parity), volatility (the one input you can't
observe), the common strategies, and the advanced topics — American exercise, dividends,
exotics, and assignment.

## Payoffs at expiration

Every option position is built from four primitives. The buyer pays premium for a capped
loss and an open-ended (or large) gain; the seller collects premium and takes the mirror
risk.

```
   LONG CALL              SHORT CALL            LONG PUT              SHORT PUT
 +│      /              +│___                 +│\                  +│      ___
  │____ /                │    \                 │ \                  │     /
 0│────/──── S          0│─────\──── S        0│──\─── S           0│────/──── S
  │   K                  │      \  (unbounded   │   K               │   /
 −│  premium paid       −│       \   loss)     −│  premium paid    −│  /  premium got
  capped loss            premium income         capped loss          (large downside)
```

- **Long call** — max loss = premium; gain unbounded as $S$ rises above $K$.
- **Short call** — max gain = premium; loss **unbounded** if naked (must deliver shares at
  $K$ no matter how high $S$ goes).
- **Long put** — max loss = premium; gain large (up to $K$ − premium) as $S$ falls to zero.
- **Short put** — max gain = premium; loss large if $S$ collapses (obligated to buy at $K$).

A contract is **100 shares**, so a $2.50 premium quote costs $250 and a one-point move in
the underlying is $100 per contract.

## Black-Scholes Model

The Black-Scholes model prices European options under a no-arbitrage argument. Developed by
Fischer Black and Myron Scholes and published in 1973, it assumes the underlying follows a
geometric Brownian motion and derives a price by constructing a continuously rebalanced
risk-free hedge.

### Formulas

**Call Option:**
$$C = S_0 N(d_1) - K e^{-rT} N(d_2)$$

**Put Option:**
$$P = K e^{-rT} N(-d_2) - S_0 N(-d_1)$$

where:

$$d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$$

$$d_2 = d_1 - \sigma\sqrt{T}$$

**Variables:**
- $C$, $P$ = call / put price
- $S_0$ = current price of the underlying
- $K$ = strike price
- $T$ = time to expiration (years)
- $r$ = risk-free interest rate (annualized)
- $\sigma$ = volatility (annualized standard deviation of returns)
- $N(x)$ = cumulative standard-normal distribution function

### Assumptions

The model's elegance comes at the cost of assumptions that don't all hold in practice:

1. **European exercise** — exercisable only at expiration.
2. **No dividends** during the option's life (relaxed by the $q$ adjustment below).
3. **Efficient, arbitrage-free markets.**
4. **Constant volatility** — the assumption the [volatility surface](#volatility) violates.
5. **Constant, known risk-free rate.**
6. **Log-normal prices** (returns normally distributed).
7. **No transaction costs, taxes, or margin.**
8. **Continuous trading** — the underlying can be traded and the hedge rebalanced freely.

### Two derivations

1. **PDE / delta hedging** — a risk-free portfolio of the option and a delta-weighted
   position in the underlying must earn the risk-free rate, giving the Black-Scholes PDE:

   $$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$$

   where $V$ is the option value.

2. **Risk-neutral valuation** — the price is the discounted expected payoff under the
   risk-neutral measure $\mathbb{Q}$:

   $$C = e^{-rT} \mathbb{E}^Q[\max(S_T - K, 0)]$$

## Greeks

The Greeks measure an option's price sensitivity to each input. They are the working
vocabulary of risk: traders hedge and adjust positions in terms of them.

| Greek | Symbol | Measures sensitivity to | Sign / shape |
|-------|--------|-------------------------|--------------|
| Delta | $\Delta$ | underlying price (per \$1) | call 0→1, put −1→0; ≈ hedge ratio |
| Gamma | $\Gamma$ | rate of change of delta | peaks ATM; same sign long calls & puts |
| Theta | $\Theta$ | passage of time (decay) | usually negative for long options |
| Vega  | $\nu$ | implied volatility (per 1%) | peaks ATM, larger for long-dated |
| Rho   | $\rho$ | interest rates (per 1%) | call +, put − |

- **Delta** doubles as the approximate hedge ratio (shares per option to be delta-neutral)
  and as a rough risk-neutral probability of finishing ITM.
- **Gamma** is highest at-the-money: it tells you how fast delta — and therefore your hedge
  — drifts as the underlying moves. High gamma means a static hedge decays quickly.
- **Theta** is the price you pay for holding optionality; it accelerates into expiration and
  is the income an option *seller* harvests.
- **Vega** is the exposure to a *change in implied volatility* — the dominant risk in
  earnings plays and long straddles.
- **Rho** matters mostly for long-dated (LEAPS) options; it's negligible for short tenors.

In practice: hedge directional risk with delta, watch gamma to know how often you must
re-hedge, manage theta when you're short premium, and trade vega when your view is about
volatility rather than direction.

## Pricing mechanics

An option's premium splits into two parts:

$$\text{Premium} = \underbrace{\text{Intrinsic value}}_{\max(S-K,0)\ \text{call},\ \max(K-S,0)\ \text{put}} + \underbrace{\text{Time value}}_{\text{everything else}}$$

**Intrinsic value** is the immediate exercise value. **Time value** (extrinsic value) is the
premium above it that buyers pay for the chance of a favorable move; it grows with time to
expiration and volatility, and decays to zero at expiry.

### Moneyness

```
                 CALL                    PUT
   ITM        S > K  (intrinsic)      S < K  (intrinsic)
   ATM        S ≈ K  (all time val,   S ≈ K  (highest gamma
              highest gamma/vega)              & vega)
   OTM        S < K  (time val only)  S > K  (time val only)
```

### Factors that move option prices

| Factor | Call | Put |
|--------|------|-----|
| Underlying price $S$ ↑ | ↑ | ↓ |
| Strike $K$ ↑ | ↓ | ↑ |
| Time to expiry $T$ ↑ | ↑ | ↑ (usually) |
| Volatility $\sigma$ ↑ | ↑ | ↑ |
| Risk-free rate $r$ ↑ | ↑ | ↓ |

### Put-call parity

For European options with the same strike and expiry:

$$C - P = S_0 - K e^{-rT} \qquad\Longleftrightarrow\qquad C + K e^{-rT} = P + S_0$$

A long call plus short put replicates a leveraged long position in the stock; violations are
arbitrage. **Example** — stock at \$100, 1-year call \$10, put \$5, strike \$100, $r=5\%$:

$$C - P = 10 - 5 = 5$$
$$S_0 - K e^{-rT} = 100 - 100\,e^{-0.05} = 100 - 95.12 = 4.88$$

The small gap is consistent with bid-ask spreads or minor mispricing.

## Volatility

Volatility is the only Black-Scholes input you can't directly observe, and it has the
largest impact on price — which is why options are really *volatility instruments*.

**Historical (realized) volatility** is the annualized standard deviation of past returns:

$$\sigma_{\text{HV}} = \sqrt{\frac{252}{n-1} \sum_{i=1}^{n} (r_i - \bar{r})^2}$$

where $r_i$ are daily log returns, $\bar{r}$ their mean, and 252 the trading days per year.
It is backward-looking.

**Implied volatility (IV)** is the $\sigma$ that makes the model price equal the *market*
price — forward-looking, the market's expectation. When IV ≫ HV, options look rich (favoring
selling); when IV ≪ HV, they look cheap (favoring buying).

### The volatility surface

IV is not a single number. Plotting it against strike and expiry gives the **volatility
surface**, whose existence is itself proof that the constant-volatility assumption fails.

```
   IV │                          equity index SKEW
      │ \__                      (reverse skew / smirk)
      │    \___                  OTM puts  bid up  (crash hedging)
      │        \____             OTM calls cheaper
      │             \______
      └───────────────────── strike
       low K (puts)   high K (calls)
```

- **Smile** — both wings (ITM & OTM) carry higher IV than ATM; common in FX/commodities.
- **Skew / smirk** — in equities, lower strikes (OTM puts) trade at higher IV, reflecting
  crash risk, demand for portfolio protection, fat tails, and the leverage effect.
- **Term structure** — ATM IV across expiries: upward-sloping (contango) in calm markets,
  downward-sloping (backwardation) under stress, humped around known events (earnings).

### VIX

The **VIX** ("fear gauge") is the market's expected 30-day S&P 500 volatility, computed from
a weighted strip of SPX option prices. Rough regimes: **< 15** complacent, **15–20** normal,
**20–30** elevated, **> 30** stress. VIX is strongly mean-reverting, which is the basis for
volatility timing and tail hedging — see [volatility trading](volatility_trading.md).

## Strategies

Options combine into positions tuned to a directional, income, volatility, or hedging view.
The marquee payoffs:

```
   COVERED CALL              VERTICAL (bull call) SPREAD
 +│      ____  capped       +│        ____ capped gain
  │     /                    │       /
 0│────/──── S              0│──────/──── S
  │   /  (long stock         │  ___/   (defined-risk
 −│  /    + short call)     −│ /        directional)

   LONG STRADDLE             IRON CONDOR
 +│\        /               +│   ______   credit, profits
  │ \      /                 │  /      \   in a range
 0│──\____/── S             0│_/        \_ S
  │   ATM  (profits          │            (short premium,
 −│  on a big move)         −│  defined risk both wings)
```

| Group | Strategy | Construction & purpose |
|-------|----------|------------------------|
| **Income** | Covered call | Long stock + short call — premium income, caps upside. |
| | Short put | Sell put — income if price holds; obligated to buy on a drop. |
| | Short call | Sell call (naked) — income, **unbounded** risk. |
| **Hedging** | Protective put | Long stock + long put — downside insurance. |
| | Collar | Long stock + long put + short call — bounded both ways, low/zero cost. |
| **Directional** | Long call / long put | Leveraged bullish / bearish, loss capped at premium. |
| | Bull call spread | Buy lower-$K$ call, sell higher-$K$ — capped, cheaper bull. |
| | Bear put spread | Buy higher-$K$ put, sell lower-$K$ — capped bear. |
| **Volatility** | Straddle | Buy ATM call + put — profits on a large move either way. |
| | Strangle | Buy OTM call + put — cheaper, needs a bigger move. |
| | Iron condor | Sell OTM call & put spreads — profits in a range (short vol). |
| | Butterfly / iron butterfly | Pin near a strike — profits on low volatility. |
| **Spreads** | Calendar / diagonal | Same (or different) strike, different expiries — trade time & vol. |
| | Ratio spread | Unequal long/short counts — finance a directional view. |
| **Synthetic / arb** | Synthetic long/short stock | +call / −put (or reverse) replicates the stock. |
| | Box spread | Bull call + bear put at the same strikes — locks a near risk-free rate. |

Long single options (long call / long put) keep maximum loss at the premium with large
upside; their short counterparts invert that profile.

## Advanced topics

### American vs European exercise

Black-Scholes prices **European** options (exercise only at expiry). Most US single-stock
options are **American** (exercise any time), so they are worth at least as much:

$$V_{\text{American}} \geq V_{\text{European}}$$

The gap is the **early-exercise premium**, priced with numerical methods (binomial trees,
finite differences). For calls on **non-dividend** stocks it is essentially zero — early
exercise throws away time value and the time value of the strike, so it's rarely optimal.
Early exercise *can* be optimal for:

- **Calls** — when an upcoming dividend exceeds the call's remaining time value.
- **Puts** — when deep ITM, where the interest earned on the strike received now beats the
  small remaining time value.

### Dividends

Dividends drop the stock on the ex-date, lowering calls and raising puts. The continuous
dividend-yield $q$ enters the call as:

$$C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

For discrete dividends, subtract their present value from the spot: $S_{\text{adj}} = S_0 -
PV(\text{Dividends})$. Dividends are the main reason American calls get exercised early.

### Binary and exotic options

**Binary (digital)** options pay a fixed amount if a condition is met. A cash-or-nothing
call paying $Q$ if $S_T > K$ is worth:

$$C_{\text{binary}} = Q e^{-rT} N(d_2)$$

since $N(d_2)$ is the risk-neutral probability of finishing ITM. Other **exotics**:

- **Asian** — payoff on the *average* price, $\max\!\big(\tfrac{1}{n}\sum_i S_{t_i} - K, 0\big)$; less manipulable, lower vol.
- **Barrier** — knock-in / knock-out when $S$ crosses a level $H$; cheaper than vanilla.
- **Lookback** — payoff on the path max/min (perfect hindsight); expensive.
- **Chooser** — pick call or put later; **compound** — options on options; **rainbow** —
  payoff on several underlyings (best-of, worst-of, spread).

### Assignment and settlement mechanics

- **Exercise** — the holder uses the right (call: buy at $K$; put: sell at $K$), typically
  by notifying the broker before the deadline (~5:30 PM ET on expiration day).
- **Assignment** — the writer's obligation when a holder exercises; the OCC assigns it
  randomly.
- **Auto-exercise** — ITM options (by ≥ \$0.01, broker-dependent) are exercised at expiry
  unless a "do not exercise" instruction is filed.
- **Settlement** — equity options settle *physically* (100 shares); index options settle in
  *cash*.
- **Pin risk** — the underlying closing right at the strike leaves assignment uncertain and
  can hand you an unintended stock position; close ITM shorts before expiry to avoid it.

## Where this connects

- [Derivatives](derivatives.md) — options are the canonical *non-linear* derivative; the
  page covers the broader family (forwards, swaps, structured products).
- [Futures](futures.md) — the linear, both-sides-obligated counterpart; the futures-vs-
  options contrast frames why options cost premium.
- [Volatility trading](volatility_trading.md) — options are the primary vehicle for trading
  implied vs realized volatility (VIX, skew, the variance risk premium).
- [Portfolio management](portfolio_management.md) — collars, protective puts, and covered
  calls are core hedging and income overlays.
- [Risk management](risk_management.md) — the Greeks are the language of position risk;
  sizing short-premium trades is where accounts blow up.
- [Stocks](stocks.md) — single-stock options reshape the risk of an equity holding.
- [Event-driven](event_driven.md) — earnings and catalysts drive IV crush and event-vol
  straddle/spread setups.

## Pitfalls

- **Theta bleed** — long options lose value every day the underlying sits still; being
  *right on direction but slow* still loses if time value decays first.
- **Naked short calls are unbounded** — selling calls without owning the stock exposes you
  to theoretically unlimited loss; a covered call or call spread defines the risk.
- **Assignment & pin risk** — short ITM options can be assigned (early, for American), and a
  pin at the strike can leave a surprise stock position over the weekend.
- **IV crush** — buying options into earnings or a known event pays inflated IV that
  collapses afterward; the stock can move your way and the option still loses.
- **Liquidity and spreads** — far-OTM and far-dated strikes have wide bid-ask spreads; the
  round-trip cost can dwarf the edge.
- **Early-exercise surprise** — short calls on dividend-paying stocks get exercised the day
  before ex-dividend; you can be assigned and lose the dividend.
- **Paying up on vega** — buying when IV is already high (the surface is rich) means you
  need a *bigger* realized move just to break even.

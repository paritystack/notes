# Factor Investing

## Overview

Factor investing says that the return of almost any stock is mostly *not* about
the company — it's about a handful of shared, systematic traits the stock has in
common with thousands of others: how small it is, how cheap, whether it's been
rising, how profitable. These traits are **factors**, and historically a few of
them have earned a persistent premium. The job is to harvest that premium
cheaply rather than to pick winners.

This is the theory layer beneath several sibling pages. [Portfolio
management](portfolio_management.md) allocates capital but treats assets as
opaque return streams; this page *decomposes* those returns into exposures.
[ETFs](etfs.md) sell factors as "smart-beta" products; this is what's inside the
wrapper. [Momentum & trend](momentum_trend.md) and [pairs & mean
reversion](pairs_mean_reversion.md) are factor strategies in the wild, and
[fundamental analysis](fundamental_analysis.md) supplies the value/quality
metrics factors are sorted on. The pricing machinery sits in [quantitative
finance](quant_finance.md).

```
        decomposing a single stock's return
   total return
       │
       ├─ market beta × market return     ← the tide (CAPM)
       ├─ size      × SMB                  ┐
       ├─ value     × HML                  ├─ factor exposures × factor premia
       ├─ momentum  × WML                  ┘   (systematic, harvestable, cheap)
       └─ alpha (+ noise)                  ← what's actually skill / luck
```

The punchline of the last 30 years of empirical finance: most of what used to be
called "alpha" was really **factor beta** in disguise — explainable, replicable,
and far cheaper to buy than a hedge fund's fee.

## From CAPM to multi-factor models

The **Capital Asset Pricing Model (CAPM)** said there is exactly *one* factor:
the market. A stock's expected excess return is just its **beta** (sensitivity to
the market) times the market premium. Everything else is idiosyncratic and
diversifiable.

```
CAPM:   E[Rᵢ] − R_f = βᵢ · (E[R_m] − R_f)        one source of priced risk
```

The data disagreed. Small stocks and cheap ("value") stocks beat what their CAPM
beta predicted, year after year. **Arbitrage Pricing Theory (APT)** generalized
the idea to *many* priced risks, and **Fama & French (1992)** made it concrete
with a **three-factor model**: market, **size** (SMB, "small minus big"), and
**value** (HML, "high minus low" book-to-market). Carhart added **momentum**
(WML/UMD) in 1997, and Fama-French extended to **five factors** (2015) by adding
**profitability** (RMW) and **investment** (CMA).

```
FF3:   Rᵢ − R_f = αᵢ + βᵢ(R_m−R_f) + sᵢ·SMB + hᵢ·HML + εᵢ
                          ▲ market    ▲ size    ▲ value
```

Running this regression on a fund's returns is the standard test: if the loadings
explain the performance and **α ≈ 0**, the manager delivered factor exposure, not
skill. (See beta and the market model in [risk management](risk_management.md).)

## The canonical factors

Each factor is a *recipe*: rank stocks on a metric, go long the favorable end,
short the unfavorable end. The classic ones:

| Factor | Sorted on (proxy) | Long / short | Thesis |
|--------|-------------------|--------------|--------|
| **Market (MKT)** | beta to the market | long equities vs. cash | equity risk premium — the tide |
| **Size (SMB)** | market capitalization | small − big | small caps are riskier / under-covered |
| **Value (HML)** | book/price, E/P, CF/P | cheap − expensive | cheap stocks compensate for distress risk or mispricing |
| **Momentum (WML)** | trailing 12-1 month return | winners − losers | underreaction to news; trends persist |
| **Profitability (RMW)** | gross profit / ROE | robust − weak | profitable firms outperform per dollar of book |
| **Investment (CMA)** | asset growth | conservative − aggressive | empire-builders underperform |
| **Quality** | margins, leverage, accruals | high − low quality | durable, well-run firms |
| **Low volatility** | realized/idiosyncratic vol | low − high vol | the "low-vol anomaly" — risk underpriced at the top end |

Two structural notes. **Momentum and value are negatively correlated** — they
tend to win in different regimes, which is exactly why combining them diversifies.
And **low-vol breaks CAPM head-on**: it says *less* risk earned *more* return, the
opposite of what a single-beta world predicts.

## How a factor portfolio is built

A factor is a portfolio, not a stock. The canonical construction is a
**long-short** built from quantile sorts:

```python
def long_short_factor(scores, returns, quantile=0.2):
    """Top-minus-bottom factor return. `scores`: higher = more exposed.
    `returns`: next-period stock returns aligned to scores."""
    n = len(scores)
    ranked = scores.sort_values(ascending=False)
    k = int(n * quantile)
    longs  = ranked.index[:k]      # most exposed
    shorts = ranked.index[-k:]     # least exposed
    return returns[longs].mean() - returns[shorts].mean()   # dollar-neutral
```

Real construction adds: **periodic rebalancing** (factors drift as prices move —
momentum needs monthly refresh, value tolerates annual), **breakpoints** (FF uses
NYSE medians/terciles to avoid micro-caps dominating), and **weighting** (cap- vs
equal-weight changes the size tilt). Most *investable* products are **long-only
tilts** — you overweight the favorable end without shorting, capturing part of the
premium with none of the borrow cost or short squeeze risk. The pure
**market-neutral** long-short version isolates the factor but lives in the
shorting/financing world of [stat arb](pairs_mean_reversion.md).

## Why factors earn a premium

There are two camps, and the honest answer is "some of both, depends on the
factor."

- **Risk-based (rational).** The premium is *compensation for bearing bad-times
  risk*. Value and small stocks tank hardest in recessions exactly when you can
  least afford it, so they must offer a higher long-run return to get held. Under
  this view the premium is real and durable — it's a risk you're paid to take.
- **Behavioral (mispricing).** The premium comes from *systematic investor
  errors* that arbitrage can't fully erase. Momentum reflects underreaction to
  news; value reflects overreaction to bad headlines; low-vol reflects a
  preference for lottery-like stocks. These *can* be arbitraged away once
  published — which is the worry of the next section.

The distinction matters: a risk premium should persist after publication; a
behavioral anomaly may decay as crowds pile in.

## Smart beta & factor products

"Smart beta" is the retail packaging of factor investing: rules-based index ETFs
that weight by a factor instead of by market cap — a value ETF, a momentum ETF, a
multi-factor blend. They sit between cheap passive indexing and expensive active
management on both fee and tracking-error. The product mechanics, weighting
schemes, and concrete tickers live on the [ETFs](etfs.md) page; this page is the
*why* behind them.

## Crowding, decay & factor timing

Factors are not free money:

- **Post-publication decay.** Documented factors tend to weaken *after* the paper
  comes out — studies find roughly a third of the premium erodes once a signal is
  public and capital chases it. Some of the "zoo" never existed outside the
  backtest.
- **Crowding.** A popular factor becomes a [crowded
  trade](pairs_mean_reversion.md): valuations on the long leg stretch, and a
  forced-deleveraging unwind (the **August 2007 "quant quake"**) can hit everyone
  running the same sort simultaneously.
- **Cyclicality & deep drawdowns.** Factors underperform for *years*. Value
  endured a brutal ~2010s drawdown that made many declare it dead — right before
  it rebounded in 2021. The premium is a long-run average, not a yearly paycheck.
- **Timing is hard.** Tactically rotating between factors is seductive and mostly
  unrewarding — factor valuations and momentum give weak timing signals, and the
  turnover costs eat the edge. Most evidence favors *diversifying across* factors
  over *timing* them.

## Where this connects

- [Portfolio management](portfolio_management.md) — factors are the modern lens on
  diversification and risk budgeting; allocate across factor premia, not just
  asset classes.
- [ETFs](etfs.md) — smart-beta products are the investable wrapper for everything
  here.
- [Momentum & trend](momentum_trend.md) — the momentum factor as a standalone
  strategy.
- [Pairs & mean reversion](pairs_mean_reversion.md) — market-neutral long-short
  construction and crowding share the same machinery.
- [Quantitative finance](quant_finance.md) — the pricing theory (APT, risk-neutral
  vs. real-world premia) underneath factor models.
- [Risk management](risk_management.md) — beta, the market model, and factor
  exposure as a risk-decomposition tool.
- [Behavioral finance](behavioral_finance.md) — the biases that the behavioral
  explanation of factor premia rests on.

## Pitfalls

- **The "factor zoo."** Hundreds of published factors exist; most are
  data-mined noise. With enough tries *something* looks significant — demand an
  economic story and out-of-sample/global evidence before believing a factor.
- **Multiple testing / p-hacking.** A t-stat of 2 is far too lax when you tested
  300 signals; the bar for a *new* factor should be much higher (3+).
- **Ignoring transaction costs.** Momentum's high turnover and small-cap factors'
  thin liquidity can erase the paper premium once [trading
  costs](market_microstructure.md) and capacity limits are charged.
- **Confusing factor beta with alpha.** A "great" manager whose returns vanish in
  a Fama-French regression sold you factor exposure at active-management prices.
- **Timing factors.** Switching factors based on recent performance usually buys
  high and sells low; the diversification benefit comes from *holding* the set.
- **Backtest overfitting.** Survivorship bias, look-ahead in the sort metric, and
  ignoring the short-leg borrow cost all flatter a historical factor that won't
  repeat live.

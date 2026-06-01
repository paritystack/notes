# Futures

## Overview

A future is a standardised, exchange-traded contract to buy or sell an asset at a fixed
price on a fixed date — the simplest leveraged [derivative](derivatives.md). Unlike
[options](options.md), both sides are *obligated*, so the payoff is linear and symmetric.
Futures are the price-discovery and hedging venue for nearly every asset class in this
book: [commodities](commodities.md) (oil, gold, grain), [forex](forex.md) (currency
futures), equity indices ([stocks](stocks.md)), and rates ([interest rates](interest_rates.md)).
Because they are levered, sizing and stops are governed by [risk management](risk_management.md).

This page covers contract mechanics, the margin/mark-to-market system that makes daily
settlement work, the term structure (contango vs backwardation) and roll yield, and how
futures differ from options.

## Contract mechanics

A futures contract standardises everything except price: the underlying, contract size,
tick value, delivery months, and settlement method (physical delivery vs cash). That
standardisation is what makes them fungible and liquid on an exchange.

```
        WHEAT FUTURES (CBOT)
  ┌───────────────────────────────────┐
  │ underlying   : soft red winter wheat
  │ contract size: 5,000 bushels
  │ price quote  : cents/bushel
  │ tick         : 1/4 cent = $12.50
  │ months       : Mar May Jul Sep Dec
  │ settlement   : physical delivery
  └───────────────────────────────────┘
   long  ── obligated to BUY  at expiry
   short ── obligated to SELL at expiry
```

Most participants never take delivery: they close (offset) the position before expiry, or
roll it to a later month. Hedgers use futures to lock a price — a farmer **shorts** wheat
to fix their selling price; an airline **longs** crude to fix fuel cost — while
speculators provide the liquidity on the other side.

## Margin and mark-to-market

This is the part the old "how to trade" guides skip, and it's the whole point. You don't
pay the contract's notional value; you post **margin** (a good-faith deposit, often 5–15%
of notional), which is what creates the leverage. The exchange then settles gains and
losses **every day** via mark-to-market, so credit risk never accumulates.

```
  Initial margin ──────────────────────────── you may trade
        │
  price moves against you each day
        │   daily P&L swept from/to your account (mark-to-market)
        ▼
  Maintenance margin ── balance dips below this line
        │
        ▼
  MARGIN CALL ── top up to initial margin by deadline…
        │            └─ or the broker liquidates the position
```

- **Initial margin** — required to open.
- **Maintenance margin** — the floor; fall below it and you get a **margin call**.
- **Mark-to-market** — each day's profit is credited (and can be withdrawn); each day's
  loss is debited immediately. Leverage cuts both ways: a small adverse move can wipe the
  margin and force liquidation at the worst time.

## Term structure: contango and backwardation

Different expiry months trade at different prices. The curve of price vs expiry is the
**term structure**, and its slope drives the return of anyone holding through time.

```
 price                         price
   │      contango              │   backwardation
   │        ___-- far months    │ \__
   │    _--                     │    \___
   │ _-- spot                   │  spot  \___ far months
   └────────────── expiry       └────────────── expiry
  far months EXPENSIVE          far months CHEAP
  (storage/carry > yield)       (scarcity / convenience yield)
```

- **Contango** — far months priced above spot (normal for storable commodities: you pay
  carry/storage). Holding a long and rolling it forward **loses** value as each expensive
  contract converges down to spot — negative **roll yield**.
- **Backwardation** — far months below spot (tight supply, high convenience yield).
  Rolling a long here **earns** positive roll yield.
- **Basis** — the gap between spot and futures price; it converges to zero at expiry. The
  basis is what a hedger is really exposed to once the outright price is locked.

Roll yield is why a commodity [ETF](etfs.md) that holds front-month futures can bleed
relative to spot in persistent contango, even when the spot price is flat.

## Futures vs options

Both are leveraged [derivatives](derivatives.md), but the obligation differs, and that
changes the entire risk profile.

```
                 FUTURES                    OPTIONS (long)
  obligation   both sides must transact   buyer has the RIGHT, not duty
  upfront cost margin (a deposit)         premium (a sunk cost)
  buyer loss   unbounded                  capped at premium paid
  buyer gain   unbounded                  unbounded (calls)
  seller       symmetric, unbounded       premium income, unbounded risk
  payoff       linear                     non-linear (kinked at strike)
```

A gold **future** obligates you to the full up-and-down P&L from the entry price. A gold
**call option** caps your downside at the premium but you pay that premium whether or not
the trade works. Futures are cheaper to carry and cleaner to hedge with; options buy you
asymmetry. See [options](options.md) for pricing (Black-Scholes, the Greeks).

## Where this connects

- [Derivatives](derivatives.md) — futures are the canonical linear derivative; the page
  covers forwards, swaps, and the broader family.
- [Commodities](commodities.md) — physical-delivery futures are how commodity markets set
  price; contango/backwardation matters most here.
- [Forex](forex.md) / [Interest Rates](interest_rates.md) — currency and rate futures hedge
  the same exposures discussed there.
- [ETFs](etfs.md) — futures-based ETFs inherit roll yield, so contango erodes returns.
- [Risk Management](risk_management.md) — daily mark-to-market and leverage make position
  sizing and stop discipline essential.

## Pitfalls

- **Leverage cuts both ways** — margin is a fraction of notional, so a small adverse move
  can exceed your deposit and force liquidation.
- **Margin calls hit at the worst time** — volatility spikes raise margin requirements
  exactly when you're losing; under-funded accounts get closed out at the bottom.
- **Roll/contango bleed** — holding long futures through contango loses value on every
  roll regardless of spot; a flat spot price can still mean a losing position.
- **Expiry and delivery** — forget to offset a physically-settled contract and you can be
  obligated to take/make delivery.
- **Basis risk** — a hedge using a related-but-different contract leaves you exposed to
  the basis even after the outright price is locked.
- **Limit moves** — exchanges halt trading at daily price limits, which can trap a
  position you cannot exit.

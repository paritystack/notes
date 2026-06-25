# Market Microstructure

## Overview

Market microstructure is the *plumbing* under every trading page: how an order actually
becomes a fill, who provides the liquidity, and what it really costs. The strategy pages
assume execution is free and instant — [algorithmic trading](algorithmic_trading.md) lists
"market making" as a strategy, [building trading bots](trading_bots.md) sends orders through
a broker API, [pairs trading](pairs_mean_reversion.md) and [momentum](momentum_trend.md)
assume you get filled at the price you modeled. This page is where that assumption gets paid
for, in spread, slippage, and market impact.

The core object is the **limit order book**, and the core question for any trader is
**transaction cost** — the gap between the decision price and the realized fill.

```
        the lifecycle of an order
  decision price
       │   spread + fees + slippage + impact
       ▼   ◄──────── transaction cost ─────────►
   route → match (price-time priority) → fill → clear → settle (T+1)
```

## The limit order book

An exchange matches buyers and sellers through a **central limit order book (CLOB)** — a
sorted list of resting limit orders on each side:

```
            ASKS (sellers)            depth
   100.04  ████ 1,200                  ▲
   100.03  ██ 600                      │  "offer" side
   100.02  █████████ 2,500   ◄ best ask│
  ─────────────────────────  spread = 0.02
   100.00  ████████ 2,100    ◄ best bid│
    99.99  ███ 800                     │  "bid" side
    99.98  ██████ 1,500                ▼
            BIDS (buyers)
```

- **Best bid / best ask (the "top of book")** — highest price a buyer will pay, lowest a
  seller will accept. The midpoint is the **mid price**.
- **Bid-ask spread** — `ask − bid`. The implicit cost of an immediate round trip; the market
  maker's compensation. Tighter in liquid names (SPY: ~1 cent), wider in illiquid ones.
- **Depth / liquidity** — the size resting at each level. Deep books absorb large orders with
  little price movement; thin books move on small size.

**Order types** (the [bot page](trading_bots.md) tabulates broker usage): a **market** order
*takes* liquidity — crosses the spread for an immediate fill at whatever price the book
offers. A **limit** order *makes* (provides) liquidity — rests in the book until matched, with
price control but no fill guarantee. This **maker/taker** distinction drives both fees and
strategy.

## Matching: price-time priority

When a marketable order arrives, the engine fills it against the book by **price-time
priority**: best price first, and within a price level, the order that arrived earliest fills
first (FIFO). A large market buy "walks the book," consuming successive ask levels at
worsening prices — the mechanical source of **slippage**.

```
Market BUY 4,000 shares hits the asks above:
  2,500 @ 100.02  ┐
  600   @ 100.03  ├─ avg fill ≈ 100.029, not 100.02
  900   @ 100.04  ┘   → slipped 0.9¢ past the quote
```

This is why *displayed* price ≠ *achieved* price for anything bigger than top-of-book size.
(Some venues use pro-rata matching instead of time priority, common in futures.)

## Liquidity providers: market makers & HFT

**Market makers** quote a two-sided market (a bid and an ask) and earn the spread for
supplying immediacy. Their risk is **adverse selection** — getting filled right before the
price moves against them (you tend to buy from them just before a rally). They manage it by
widening spreads in volatile or informed markets and by hedging **inventory**.

**High-frequency trading (HFT)** is the modern, latency-sensitive form: co-located servers,
microsecond reaction, strategies like passive market making, latency arbitrage (exploiting
stale quotes across venues), and statistical arb. Net effect is contested — generally tighter
spreads and more displayed liquidity in calm markets, but liquidity that can *evaporate* in
stress (see the 2010 Flash Crash).

## Transaction costs

What trading actually costs, beyond commissions:

- **Spread cost** — half the bid-ask spread per side, paid for immediacy.
- **Slippage** — the difference between expected and executed price from walking the book or
  the market moving between decision and fill.
- **Market impact** — *your own* order moving the price. **Temporary impact** reverts after
  you stop trading; **permanent impact** is the information your trade reveals. Impact grows
  roughly with the **square root** of order size relative to volume — the key reason large
  orders must be sliced.
- **Implementation shortfall** — the all-in benchmark: `decision price − final average fill`,
  including the opportunity cost of any unfilled portion. The honest measure of execution
  quality.

**Transaction cost analysis (TCA)** measures realized execution against benchmarks (arrival
price, VWAP, close) to evaluate brokers and algos. These costs are exactly what backtests
overstate returns by ignoring — see the backtesting caveats in
[algorithmic trading](algorithmic_trading.md).

## Execution algorithms

To move size without paying full impact, brokers slice a "parent" order into "child" orders
over time:

| Algo | Logic | Best for |
|------|-------|----------|
| **TWAP** | Even slices over a fixed time window | Simple, predictable schedule |
| **VWAP** | Match the day's volume profile (trade more when volume is high) | Benchmarking to VWAP |
| **POV** (% of volume) | Stay a fixed fraction of live volume | Adapting to actual activity |
| **Implementation Shortfall** | Front-load to minimize slippage-vs-impact tradeoff | Urgency / alpha decay |
| **Iceberg** | Show a small slice, hide the rest | Concealing size in the book |

The universal tradeoff: **trade fast** → high market impact; **trade slow** → high
*timing/opportunity risk* (the price drifts away). IS algos solve exactly this optimization
(the Almgren–Chriss framework formalizes it).

## Venues, dark pools & order routing

Modern equity markets are **fragmented** across many venues:

- **Lit markets** — exchanges with a public, displayed order book (NYSE, Nasdaq).
- **Dark pools** — private venues that don't display quotes; let institutions trade large
  blocks without revealing intent (less impact) at the cost of less transparency.
- **Smart order routers (SOR)** — algorithms that split an order across venues to find the
  best price and liquidity, the glue holding a fragmented market together.

**Payment for order flow (PFOF)** — retail brokers route orders to wholesalers who pay for
the flow and fill it (often at a slight price improvement vs. the lit quote). It funds
"commission-free" trading but raises best-execution and conflict-of-interest questions. This
is the machinery behind the broker APIs the [bot page](trading_bots.md) connects to.

## Clearing & settlement

A fill is a *trade*, not yet a *transfer*. After matching:

- **Clearing** — a central counterparty (CCP) steps between buyer and seller, nets offsetting
  positions, and guarantees the trade (removing counterparty risk).
- **Settlement** — actual exchange of cash for securities. U.S. equities moved to **T+1** (one
  business day after trade) in May 2024; FX is typically T+2.

Settlement timing drives margin, the cost of leverage, and why short selling needs a
**locate** (a borrowable share). The CCP/clearing concept connects to the post-2008
derivatives plumbing in [derivatives](derivatives.md).

## Microstructure across asset classes

- [Forex](forex.md) — decentralized, dealer/ECN-based rather than a single CLOB; "last look"
  lets dealers reject quotes.
- [Crypto](crypto.md) — 24/7 CLOBs on centralized exchanges plus on-chain **AMMs** (automated
  market makers) where a pricing curve replaces the order book.
- [ETFs](etfs.md) — the creation/redemption arbitrage by authorized participants is what keeps
  price near NAV; on-screen spread hides deeper underlying liquidity.
- [Futures](futures.md) — exchange CLOBs, often pro-rata matching, with the CCP central to the
  daily mark-to-market.

## Where this connects

- [Building trading bots](trading_bots.md) — order types, routing, and fills are the API
  surface this page explains underneath.
- [Algorithmic trading](algorithmic_trading.md) — market making and the execution costs that
  realistic backtests must model.
- [Pairs trading](pairs_mean_reversion.md) & [momentum](momentum_trend.md) — strategies whose
  edge can be entirely eaten by spread and impact.
- [Volatility trading](volatility_trading.md) — wide option spreads make execution a
  first-order cost.
- [Forex](forex.md), [Crypto](crypto.md), [ETFs](etfs.md), [Futures](futures.md) — the
  asset-class variations of these mechanics.
- [Risk management](risk_management.md) — liquidity risk and the cost of forced exits.

## Pitfalls

- **Quoting the spread you can't get** — the displayed top-of-book size is tiny; anything
  larger fills worse, so backtests priced at the mid or the quote overstate returns.
- **Ignoring impact in sizing** — a strategy that "works" on paper dies once your own order
  moves the price; impact scales with size, not linearly.
- **Market orders in thin books** — crossing a wide or shallow spread (illiquid names, off
  hours, news) can slip painfully; use limits when you can wait.
- **Confusing volume with liquidity** — high volume can still be illiquid if the book is thin
  and fast; depth and resilience matter more than turnover.
- **Stops are market orders** — a triggered stop can gap far through your level in a fast move;
  the "protection" isn't a guaranteed price.
- **Assuming all venues are equal** — fragmentation, PFOF, and dark fills mean the price you
  see isn't always the price you get; routing matters.

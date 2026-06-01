# Technical Analysis

## Overview

Technical analysis (TA) reads price and volume history to estimate the *probability* of
future moves, rather than valuing the underlying business — that's the job of
[fundamental analysis](fundamental_analysis.md). It assumes price already discounts known
information and that supply/demand leaves repeatable footprints in charts. TA supplies the
entry/exit signals behind systematic strategies elsewhere in this book:
[momentum & trend](momentum_trend.md), [pairs & mean reversion](pairs_mean_reversion.md),
[volatility trading](volatility_trading.md), and most [algorithmic](algorithmic_trading.md)
[trading bots](trading_bots.md).

Treat everything below as probabilistic and easy to fool yourself with — see
[Pitfalls](#pitfalls). This page covers the price/volume primitives, the core indicators
(moving averages, RSI, MACD, Bollinger Bands), and chart patterns.

## Price, trend, support and resistance

Everything in TA builds on the trend (the direction of higher-highs/higher-lows or the
reverse) and on **support/resistance** — price levels where buying or selling has
repeatedly absorbed the move.

```
 price
   │            ┌── resistance (supply) ──┐
   │      /\    ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
   │     /  \      /\          /\
   │    /    \    /  \   /\   /  \      uptrend:
   │   /      \  /    \ /  \ /    \     higher highs +
   │  /        \/      V    V      \    higher lows
   │ /   ____________________________  support (demand)
   └────────────────────────────────────▶ time
```

A level flips role once broken: prior resistance often becomes new support. Volume
confirms — a breakout on heavy volume is more trustworthy than one on thin volume.

## Indicators

Indicators are math transforms of price/volume. They lag (they're built from past data),
so they confirm and filter rather than predict. The four most common:

**Moving averages (SMA/EMA)** smooth price to expose trend. SMA is a flat average over N
periods; EMA weights recent prices more, so it turns faster. Crossovers (e.g. price above
the 200-day, or a 50/200-day "golden/death cross") are classic trend filters, and MAs act
as dynamic support/resistance.

```
 RSI (0–100)                         MACD
 100 ┤                               ┌ MACD line  = EMA12 − EMA26
  70 ┤···· overbought ····           ├ signal     = EMA9 of MACD
     │   \    /\                      └ histogram  = MACD − signal
  50 ┤    \  /  \   /
  30 ┤·····\/····\·/···· oversold     bullish: MACD crosses ABOVE signal
   0 ┤                                 bearish: MACD crosses BELOW signal
```

**RSI** is a momentum oscillator: >70 flags overbought, <30 oversold. In a strong trend
it can stay pinned at an extreme, so it's most useful for divergences (price makes a new
high, RSI doesn't).

**MACD** measures the gap between a fast and slow EMA; the signal-line crossover and the
histogram flipping sign are the trade triggers, combining trend and momentum in one tool.

**Bollinger Bands** plot an SMA with bands at ±2 standard deviations. The bands widen with
volatility and narrow ("squeeze") before expansion; touches of the upper/lower band flag
stretched conditions but are not standalone reversal signals.

## Chart patterns

Patterns are recurring shapes traders read as continuation (trend resumes) or reversal
(trend turns). They're informative but subjective — the same chart looks different to two
analysts.

```
  HEAD & SHOULDERS (reversal)        TRIANGLES (usually continuation)
            head                      ascending      descending
            /\                        ‾‾‾‾‾‾‾‾        \
     shldr /  \ shldr                 /                \
      /\  /    \  /\                 /                  \____
  ___/__\/______\/__\__ neckline    /                    ‾‾‾‾
     break neckline → target        flat top,        flat bottom,
                                     rising lows      falling highs
```

- **Continuation:** triangles (ascending/descending/symmetrical), flags and pennants,
  rectangles — brief consolidation before the prior trend resumes, ideally on a
  volume-confirmed breakout.
- **Reversal:** head-and-shoulders (and its inverse), double/triple tops and bottoms —
  the trend exhausts and turns; confirmation is the break of the neckline or support level.
- A measured-move target is often the pattern's height projected from the breakout point.

## Where this connects

- [Fundamental Analysis](fundamental_analysis.md) — the complement: *what* to own (value)
  vs *when* to act (timing). Many combine both.
- [Momentum & Trend](momentum_trend.md) / [Pairs & Mean Reversion](pairs_mean_reversion.md)
  — these strategies are TA signals made systematic.
- [Volatility Trading](volatility_trading.md) — Bollinger squeezes and range analysis feed
  volatility-based entries.
- [Algorithmic Trading](algorithmic_trading.md) / [Trading Bots](trading_bots.md) —
  indicators and patterns are the rules most bots encode and backtest.

## Pitfalls

- **Lagging signals** — every indicator is built from past prices; crossovers fire after
  the move has begun and whipsaw in sideways markets.
- **Overfitting** — tune enough indicators/parameters to past data and you'll "find" a
  system that fits noise and fails live; insist on out-of-sample testing.
- **Look-ahead bias** — backtests that use a bar's close to act *within* that bar (or that
  reference future data) inflate results; align signals to information actually available.
- **Confirmation bias / subjectivity** — patterns are easy to see after the fact;
  pre-commit to rules rather than redrawing trendlines to fit your view.
- **No edge by itself** — TA's efficacy is debated; it's a probability tool, not a
  prediction. Pair it with risk limits and position sizing from
  [risk management](risk_management.md).
- **Self-defeating popularity** — when a level/pattern is widely watched, stops cluster
  there and get hunted, so the textbook reaction may not hold.

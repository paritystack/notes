# Building Trading Bots

## Overview

This is the hands-on, code-first companion to [Algorithmic Trading](algorithmic_trading.md). That guide covers *what* to trade and *why* (strategies, backtesting, metrics, risk theory). This guide covers *how to actually wire code to a broker and trade*: authenticating, pulling live market data, placing and tracking orders, structuring a bot, and running it safely from paper trading to a live account.

The strategy logic here is deliberately trivial (a moving-average crossover, borrowed from the theory doc). The point is the **plumbing** around it — the data → signal → order → risk loop — which is where most real-world bot effort goes.

> All code uses Python and the **Alpaca** API as the concrete example (free paper-trading accounts, commission-free US stocks + crypto, clean REST + websocket API). The patterns are wrapped behind a broker-agnostic interface so Interactive Brokers, CCXT/crypto exchanges, or others can be swapped in.

## Prerequisites

```bash
pip install alpaca-py pandas python-dotenv
```

- A free Alpaca account: https://alpaca.markets → generate **paper trading** API keys.
- Python 3.10+.
- Comfort with pandas (see [Technical Analysis](technical_analysis.md) for the indicator math).

---

## 1. Setup, Auth & Market Data

### Storing credentials (never hardcode keys)

Put secrets in a `.env` file that is **git-ignored**. Hardcoded keys leak through commits, screenshots, and logs.

```bash
# .env  (add to .gitignore!)
ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxx
ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ALPACA_PAPER=true
```

```python
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY    = os.environ["ALPACA_API_KEY"]
SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
PAPER      = os.environ.get("ALPACA_PAPER", "true").lower() == "true"
```

### Connecting

```python
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient

trading = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
data    = StockHistoricalDataClient(API_KEY, SECRET_KEY)

account = trading.get_account()
print(f"Status: {account.status}")
print(f"Buying power: ${float(account.buying_power):,.2f}")
print(f"Equity: ${float(account.equity):,.2f}")
```

### Fetching historical bars (for backtests & warming up indicators)

```python
import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

def get_bars(symbol: str, days: int = 365, timeframe=TimeFrame.Day) -> pd.DataFrame:
    """Return a clean OHLCV DataFrame indexed by timestamp."""
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=datetime.now() - timedelta(days=days),
    )
    bars = data.get_stock_bars(request).df
    if bars.empty:
        raise ValueError(f"No data returned for {symbol}")
    # Multi-index (symbol, timestamp) -> single symbol frame
    return bars.loc[symbol]

df = get_bars("AAPL", days=300)
print(df[["open", "high", "low", "close", "volume"]].tail())
```

### Live data: REST polling vs. websocket streaming

There are two ways to get current prices. Choose based on how often your strategy reacts.

**REST polling** — simplest. Ask for the latest quote on a schedule. Good for strategies that act on bar closes (e.g. once a minute or once a day).

```python
from alpaca.data.requests import StockLatestQuoteRequest

def latest_price(symbol: str) -> float:
    req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
    quote = data.get_stock_latest_quote(req)[symbol]
    return (quote.bid_price + quote.ask_price) / 2  # mid price
```

**Websocket streaming** — push-based, low latency. The server sends you every trade/quote/bar as it happens. Use for reactive or high-frequency strategies. You get a persistent connection and register async handlers.

```python
from alpaca.data.live import StockDataStream

stream = StockDataStream(API_KEY, SECRET_KEY)

async def on_bar(bar):
    print(f"{bar.symbol} {bar.timestamp} close={bar.close}")
    # feed bar into your strategy here

stream.subscribe_bars(on_bar, "AAPL")
# stream.run()  # blocks; runs the asyncio event loop
```

> **Rule of thumb:** poll for daily/minute-close strategies (less to break); stream when latency matters or you watch many symbols. Streaming adds reconnection logic you must handle (see §4).

---

## 2. Order Execution & Position Management

### A broker-agnostic interface

Wrap every broker behind one interface. Your strategy talks to `Broker`, never to Alpaca directly — so you can swap brokers, or drop in a fake broker for testing, without touching strategy code.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class Position:
    symbol: str
    qty: float
    avg_price: float
    market_value: float
    unrealized_pl: float

class Broker(ABC):
    @abstractmethod
    def buy(self, symbol: str, qty: float, limit: float | None = None): ...
    @abstractmethod
    def sell(self, symbol: str, qty: float, limit: float | None = None): ...
    @abstractmethod
    def position(self, symbol: str) -> Position | None: ...
    @abstractmethod
    def cash(self) -> float: ...
```

### Alpaca implementation

```python
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

class AlpacaBroker(Broker):
    def __init__(self, client: TradingClient):
        self.client = client

    def _submit(self, symbol, qty, side, limit):
        if limit is None:
            order = MarketOrderRequest(
                symbol=symbol, qty=qty, side=side,
                time_in_force=TimeInForce.DAY,
            )
        else:
            order = LimitOrderRequest(
                symbol=symbol, qty=qty, side=side, limit_price=limit,
                time_in_force=TimeInForce.DAY,
            )
        return self.client.submit_order(order)

    def buy(self, symbol, qty, limit=None):
        return self._submit(symbol, qty, OrderSide.BUY, limit)

    def sell(self, symbol, qty, limit=None):
        return self._submit(symbol, qty, OrderSide.SELL, limit)

    def position(self, symbol):
        try:
            p = self.client.get_open_position(symbol)
        except Exception:
            return None  # no position
        return Position(
            symbol=p.symbol, qty=float(p.qty),
            avg_price=float(p.avg_entry_price),
            market_value=float(p.market_value),
            unrealized_pl=float(p.unrealized_pl),
        )

    def cash(self):
        return float(self.client.get_account().cash)
```

### Order types (and when to use them)

| Type | Use when | Risk |
|------|----------|------|
| **Market** | You must fill *now* and accept any price | Slippage, esp. illiquid/fast markets |
| **Limit** | You want price control, can wait | May never fill |
| **Stop** | Auto-exit on adverse move | Gaps through your stop |
| **Stop-limit** | Stop + price floor | May not fill in a fast drop |

See [General Finance → Market Orders](index.html) for the full taxonomy and durations (DAY, GTC, IOC, FOK).

### Tracking fills & reconciliation

Submitting an order does **not** mean it filled. Orders sit in `new` → `accepted` → `partially_filled` → `filled` (or `canceled`/`rejected`). Always **reconcile against the broker's reported position** — never assume your local state matches reality.

```python
import time

def wait_for_fill(client, order_id, timeout=30):
    """Poll an order until terminal state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        o = client.get_order_by_id(order_id)
        if o.status in ("filled", "canceled", "rejected"):
            return o
        time.sleep(1)
    client.cancel_order_by_id(order_id)  # don't leave it hanging
    return client.get_order_by_id(order_id)
```

> **Golden rule:** the broker is the source of truth for positions and cash. On startup and after every order, re-sync from `broker.position()` rather than trusting an in-memory counter.

---

## 3. The Full Bot Skeleton

This ties §1 and §2 together: fetch data → compute signal → size position → place order → manage risk → loop. The signal is the MA crossover from [Algorithmic Trading](algorithmic_trading.md); everything around it is the bot.

```python
"""
Minimal but complete trading bot.
Strategy: long when short MA > long MA, flat otherwise (MA crossover).
Run against PAPER trading only until proven.
"""
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("bot")


def compute_signal(df, short=50, long=200) -> int:
    """Return target position: 1 = long, 0 = flat."""
    short_ma = df["close"].rolling(short).mean().iloc[-1]
    long_ma  = df["close"].rolling(long).mean().iloc[-1]
    return 1 if short_ma > long_ma else 0


def position_size(broker: Broker, price: float, risk_pct=0.95) -> int:
    """Whole shares we can afford with `risk_pct` of cash."""
    budget = broker.cash() * risk_pct
    return int(budget // price)


def run_once(broker: Broker, symbol: str):
    """One decision cycle. Idempotent: safe to call repeatedly."""
    df = get_bars(symbol, days=300)
    target   = compute_signal(df)
    price    = latest_price(symbol)
    current  = broker.position(symbol)
    held     = current.qty if current else 0

    log.info(f"{symbol} target={target} held={held} price={price:.2f}")

    if target == 1 and held == 0:
        qty = position_size(broker, price)
        if qty > 0:
            log.info(f"BUY {qty} {symbol}")
            broker.buy(symbol, qty)
    elif target == 0 and held > 0:
        log.info(f"SELL {held} {symbol}")
        broker.sell(symbol, held)
    else:
        log.info("No action (already aligned with target)")


def main():
    broker = AlpacaBroker(trading)
    symbol = "AAPL"
    POLL_SECONDS = 60 * 60  # hourly; align to your timeframe

    log.info("Bot starting (paper=%s)", PAPER)
    while True:
        try:
            run_once(broker, symbol)
        except Exception as e:
            log.exception("Cycle failed: %s", e)  # never let one error kill the loop
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
```

**Why it's shaped this way:**

- **`run_once` is idempotent** — it compares *target* vs *current* and only trades the difference. If the bot restarts mid-day, it picks up correctly instead of double-buying.
- **The broker is queried for `held`**, not a local variable — reconciliation by construction (§2).
- **The loop swallows exceptions per-cycle** — a transient API error shouldn't crash the whole bot (§4).

---

## 4. Safety, Errors & Deployment

This section is what separates a toy script from something you'd trust with money.

### Paper → live workflow

1. **Paper trade the exact code** for weeks. The only change to go live should be the API keys and `paper=False` — nothing else.
2. **Live with tiny size first.** Trade 1 share. You are testing *infrastructure*, not the strategy, at this stage.
3. **Scale only after** the live P&L roughly tracks what paper/backtest predicted (it rarely matches exactly — slippage, fees, fills).

```python
# Make the paper/live switch loud and explicit
if not PAPER:
    confirm = input("LIVE trading with REAL money. Type 'LIVE' to continue: ")
    if confirm != "LIVE":
        raise SystemExit("Aborted.")
```

### Rate limits & retries

APIs throttle you (Alpaca: ~200 req/min). Back off and retry transient failures; never hammer.

```python
import time, functools

def retry(times=3, base_delay=2):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            for attempt in range(times):
                try:
                    return fn(*a, **k)
                except Exception as e:
                    if attempt == times - 1:
                        raise
                    delay = base_delay * (2 ** attempt)  # exponential backoff
                    log.warning("Retry %d after error: %s (%ss)", attempt + 1, e, delay)
                    time.sleep(delay)
        return wrapper
    return deco
```

### Websocket reconnection

Streaming connections drop. The `alpaca-py` stream auto-reconnects, but if you build your own, wrap `stream.run()` in a supervisor that restarts on disconnect with backoff — and **re-sync positions from the broker on every reconnect**, since you may have missed fills while disconnected.

### Kill switches & circuit breakers

Hard limits that halt trading regardless of what the strategy "wants."

```python
class CircuitBreaker:
    def __init__(self, max_daily_loss: float, max_position_value: float):
        self.max_daily_loss = max_daily_loss
        self.max_position_value = max_position_value
        self.start_equity = None

    def check(self, broker, account):
        equity = float(account.equity)
        if self.start_equity is None:
            self.start_equity = equity
        loss = self.start_equity - equity
        if loss >= self.max_daily_loss:
            raise SystemExit(f"KILL SWITCH: daily loss ${loss:.0f} exceeded limit")
```

Call `breaker.check(...)` at the top of every cycle. Pair it with position-size and portfolio-heat limits from [Risk Management](risk_management.md).

### Logging & observability

- Log **every decision and order** (you did this above) — you cannot debug a bot you can't replay.
- Persist trades to a file/DB for a paper trail and later analysis.
- Send a **heartbeat / alert** (email, Telegram, Slack) when the bot starts, stops, or trips a breaker. Silent death is the worst failure mode.

### Running it: where bots live

| Option | Good for | Notes |
|--------|----------|-------|
| **cron** | Daily/periodic strategies | Schedule `run_once`; no long-running process. Simple, robust. |
| **systemd service** | Always-on loop on a Linux box/VPS | Auto-restart on crash (`Restart=always`); logs to journald. |
| **Docker** | Reproducible deploys | Pin deps; same image paper→live; easy to ship to any host. |
| **Cloud (EC2/Fly/Railway)** | 24/7 reliability, crypto | Co-locate near the exchange for latency; mind egress/uptime costs. |

A minimal `systemd` unit:

```ini
# /etc/systemd/system/tradingbot.service
[Unit]
Description=Trading Bot
After=network-online.target

[Service]
WorkingDirectory=/home/you/bot
EnvironmentFile=/home/you/bot/.env
ExecStart=/home/you/bot/venv/bin/python bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now tradingbot
journalctl -u tradingbot -f   # tail logs
```

> For purely scheduled (e.g. once-a-day-at-close) strategies, prefer **cron** over an always-on loop — fewer moving parts, nothing to leak memory or hang. Use systemd/cloud when you need streaming or intraday reactivity.

---

## Common Pitfalls (Engineering Edition)

These are bugs in the *plumbing*, distinct from the strategy pitfalls in [Algorithmic Trading](algorithmic_trading.md):

1. **Trusting local state over the broker** — assuming a position exists because you "sent a buy" that never filled.
2. **No idempotency** — restarting the bot double-submits orders.
3. **Look-ahead via the latest bar** — using a still-forming (incomplete) current bar as if it were closed; act on *closed* bars.
4. **Unhandled disconnects** — websocket drops, you miss the exit signal, position runs unmanaged.
5. **Hardcoded / committed API keys** — instant account compromise.
6. **No rate-limit handling** — getting throttled or banned mid-trade.
7. **Going live before paper-proving the exact code path.**
8. **No kill switch** — a logic bug trades your account to zero overnight while you sleep.
9. **Timezones** — market hours, bar timestamps, and your server clock must agree (use the exchange's calendar; check `trading.get_clock()`).
10. **Ignoring partial fills** — selling `held` when only part of your buy filled.

## Key Takeaways

1. **The broker is the source of truth** — always reconcile positions and cash from it.
2. **Make every cycle idempotent** — compare target vs. current, trade only the delta.
3. **Wrap the broker behind an interface** — swap brokers and test with a fake one.
4. **Paper-prove the exact code** — flip only the keys to go live, then start at 1 share.
5. **Errors are normal** — retries, backoff, reconnects, and per-cycle exception isolation are mandatory, not optional.
6. **Kill switches before strategy** — cap daily loss and position size at the infrastructure layer.
7. **Log everything, alert on death** — a silent bot is a dangerous bot.
8. **Match deployment to cadence** — cron for scheduled, systemd/cloud for streaming.
9. **The plumbing is the hard part** — the signal is 20 lines; the safe execution of it is the rest.

## Resources

- **Alpaca docs**: https://docs.alpaca.markets — REST/websocket reference, paper trading.
- **alpaca-py** (official SDK), **ib_insync** (Interactive Brokers), **CCXT** (100+ crypto exchanges, one API).
- **Backtesting** before live: Backtrader, VectorBT (see [Algorithmic Trading](algorithmic_trading.md)).
- **Cloud/ops**: Docker, systemd, any cheap VPS or Fly.io/Railway for always-on bots.

## Important Disclaimer

Educational content only — **not financial advice**. ~90%+ of retail algo traders lose money. Live trading risks real capital. Always paper trade first, start small, use kill switches, and never trade money you can't afford to lose. Bugs in trading code lose money in real time.

---

See also: [Algorithmic Trading](algorithmic_trading.md) (strategy & backtesting theory), [Technical Analysis](technical_analysis.md) (indicators), [Risk Management](risk_management.md) (position sizing & limits), [Crypto](crypto.md), [Stocks](stocks.md).

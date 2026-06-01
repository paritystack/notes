# Crypto

## Overview

Cryptocurrencies are bearer digital assets settled on public blockchains rather than
through banks or clearing houses. As an asset class they behave like high-beta,
24/7-traded instruments, so the trading mechanics overlap heavily with the rest of this
book: leverage and perpetuals echo [derivatives](derivatives.md) and
[futures](futures.md), strategy design borrows from [algorithmic trading](algorithmic_trading.md)
and [trading bots](trading_bots.md), and position sizing is governed by the same
[risk management](risk_management.md) rules. Their value, unlike equities, rests on
network security and token supply schedules rather than cash flows, which ties price
dynamics to [macroeconomics](macroeconomics.md) (liquidity, real rates) more than to
fundamentals.

This page covers how chains reach consensus, the Solana architecture as a
high-throughput example, token economics, market structure, and where the real risks
live.

## Consensus: how a chain agrees on state

A blockchain is a replicated ledger with no central operator, so the core problem is
agreeing on transaction ordering among mutually distrusting nodes (Byzantine fault
tolerance). The two dominant answers are **Proof of Work** and **Proof of Stake**, which
differ in what resource you must commit to earn the right to append a block.

```
Proof of Work (Bitcoin)            Proof of Stake (Ethereum)
-----------------------            -------------------------
scarce resource : hardware + power  scarce resource : staked capital
"who appends?"  : first to find a   "who appends?"  : pseudo-randomly
                  hash < target                       chosen, weighted by stake
attack cost     : >50% of hashrate  attack cost     : >33%/>50% of stake
                  (rent/buy ASICs)                    (and it gets slashed)
finality        : probabilistic     finality        : economic / checkpointed
                  (~6 confirmations)                  (slashing makes reverts costly)
energy          : very high         energy          : ~99.9% lower
```

**Proof of Work.** Miners repeatedly hash `block data + nonce` (Bitcoin uses SHA-256)
searching for an output below a difficulty target. Finding one is hard; verifying it is
trivial. The network retargets difficulty to hold block time near a constant (~10 min
for Bitcoin). Security is physical: rewriting history means out-hashing the rest of the
network, which is the **51% attack** threshold. The block reward halves on a fixed
schedule ("halvings"), tapering new issuance toward Bitcoin's 21M cap.

**Proof of Stake.** Validators lock collateral and are pseudo-randomly selected to
propose/attest blocks, with selection weighted by stake. Honesty is enforced
economically: provably bad behaviour (double-signing, equivocation) triggers
**slashing**, burning part of the stake. PoS drops the energy cost of PoW and lowers the
hardware barrier, but concentrates influence by wealth and depends on a fair initial
distribution. **Delegated PoS** has holders vote for a small validator set for higher
throughput at the cost of more centralisation.

## Solana: a high-throughput architecture

Solana's bet is that you can scale throughput on a single chain by removing bottlenecks
rather than sharding. Its pieces form a pipeline rather than a single consensus step.

```
                 ┌─────────────────────────────────────────────┐
   txns ─────────▶  Gulf Stream   push txns to next leader early │
                 │      │          (mempool-less forwarding)      │
                 │      ▼                                          │
                 │  Proof of History   verifiable clock — hash    │
                 │      │              chain timestamps ordering   │
                 │      ▼                                          │
                 │  Tower BFT      consensus that *reads the PoH   │
                 │      │           clock* instead of voting on it │
                 │      ▼                                          │
                 │  Sealevel       parallel execution across cores │
                 │      │           (txns that don't touch same    │
                 │      ▼            state run concurrently)        │
                 │  Turbine        block propagation as small      │
                 │                 packets, BitTorrent-style        │
                 └─────────────────────────────────────────────┘
   Cloudbreak: horizontally-scaled accounts DB underneath it all
```

The key idea is **Proof of History**: a pre-consensus, verifiable delay function that
stamps a cryptographic ordering on events *before* validators vote. Because everyone
already agrees on time/order, consensus (**Tower BFT**) becomes cheap. **Sealevel**
exploits transactions declaring which accounts they touch, so non-conflicting ones
execute in parallel. The result is high throughput and sub-second finality; the cost is
high validator hardware requirements and a history of liveness outages.

## Token economics

A token's price floor and dilution are set by its issuance and sink mechanics, not by
earnings. The levers to read on any token:

- **Supply schedule** — fixed cap (BTC), disinflationary (SOL targets ~1.5% terminal
  inflation), or uncapped. New issuance is paid out as staking rewards, so non-stakers
  are diluted.
- **Sinks / burns** — fees (or a share of them) burned remove supply. Ethereum's EIP-1559
  base-fee burn can make net issuance negative under load; Solana burns part of each fee.
- **Staking yield** — nominal yield ≈ issuance / staked ratio. High headline yields are
  often just inflation you must stake to stand still against.
- **Float vs FDV** — circulating supply versus fully-diluted valuation. Low float + high
  FDV means large future unlocks (team/VC vesting) overhanging the price.

## Market structure

```
        CEX (Coinbase, Binance)         DEX (Uniswap, Jupiter)
        -----------------------         ----------------------
order book, off-chain matching     AMM pools (x*y=k) or on-chain books
custodian holds your keys          self-custody; you sign every trade
KYC, fiat on/off ramps             permissionless, pseudonymous
counterparty = the exchange        counterparty = a smart contract
```

- **Custody.** "Not your keys, not your coins": assets on a CEX are an IOU against that
  exchange (FTX showed the tail risk). Self-custody moves the risk to key management.
- **Stablecoins.** Dollar pegs are either fiat-collateralised (USDC, USDT), over-
  collateralised on-chain (DAI), or algorithmic (UST — which de-pegged to zero). They are
  the settlement layer for most crypto trading.
- **Derivatives.** Perpetual futures ("perps") dominate volume: no expiry, held in line
  with spot by a periodic **funding rate** paid between longs and shorts — see
  [derivatives](derivatives.md) and [futures](futures.md) for the contract mechanics.

## Where this connects

- [Derivatives](derivatives.md) / [Futures](futures.md) — perps, funding rates, and
  leverage are the same machinery applied to tokens.
- [Algorithmic Trading](algorithmic_trading.md) / [Trading Bots](trading_bots.md) — 24/7
  markets and open APIs make crypto a common venue for automated strategies.
- [Risk Management](risk_management.md) — extreme volatility makes position sizing and
  liquidation-price awareness non-optional.
- [Macroeconomics](macroeconomics.md) — crypto trades as a long-duration liquidity asset,
  sensitive to real rates and risk appetite.
- [Volatility Trading](volatility_trading.md) — realised vol dwarfs equities; options and
  vol surfaces behave differently here.

## Pitfalls

- **51% / long-range attacks** — small-cap PoW chains can be rented and reorged; PoS adds
  long-range and weak-subjectivity concerns.
- **Slashing** — validators (and delegators) can lose stake to downtime or
  double-signing, not just to malice.
- **Bridge risk** — cross-chain bridges are the single largest source of hacks; wrapped
  assets are only as safe as the bridge holding the collateral.
- **Impermanent loss** — providing liquidity to an AMM underperforms holding when prices
  diverge; "yield" can mask it.
- **Stablecoin de-peg** — algorithmic and under-collateralised pegs can fail abruptly;
  even fiat-backed pegs carry issuer/banking risk.
- **Custody & counterparty** — exchange insolvency, lost keys, and phishing/approvals
  drain wallets; on-chain transactions are irreversible.
- **Leverage liquidation** — perp funding plus thin liquidity produces cascading
  liquidations; know your liquidation price before sizing.

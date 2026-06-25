# Corporate Finance

## Overview

Corporate finance is the *issuer side* of the market: the decisions a company's CFO makes
about where to get money and what to do with it. The rest of the finance section mostly looks
at firms from the *outside* — [valuation](valuation.md) prices them, [fundamental
analysis](fundamental_analysis.md) reads their statements, [stocks](stocks.md) trades their
shares, [event-driven](event_driven.md) arbitrages their M&A, and [private
markets](private_markets.md) runs LBOs on them. This page is the view from *inside* the
boardroom.

Every corporate-finance question reduces to three decisions:

```
        ┌─────────────────────────────────────────────┐
        │              The firm's cash                 │
        └─────────────────────────────────────────────┘
           ▲                  │                  │
   FINANCING            INVESTMENT            PAYOUT
   how to raise $       which projects        return $ to
   (debt vs equity)     to fund (NPV)         owners (div/buyback)
        │                    │                    │
   capital structure   capital budgeting    dividend policy
```

The unifying objective is **maximizing firm value**, and the unifying tool is the **cost of
capital** ([WACC](valuation.md), derived there as a discount rate) — invest only above it,
finance to minimize it, and return cash when you can't beat it.

## Capital structure

The mix of **debt vs. equity** used to fund the firm. The central question: does the mix
*change firm value*, or just slice the same pie differently?

- **Modigliani–Miller (MM), no taxes** — in a frictionless world, capital structure is
  *irrelevant*; value comes from the assets, not the financing. The baseline that tells you
  *which frictions* actually matter.
- **MM with taxes** — interest is tax-deductible, creating a **tax shield** (`= tax rate ×
  debt`). This pushes toward *more* debt.
- **Trade-off theory** — the tax-shield benefit is offset by **costs of financial distress**
  (bankruptcy, fire sales, lost customers/employees, debt overhang). The optimum is where the
  marginal tax benefit equals the marginal distress cost.
- **Pecking-order theory** — because of information asymmetry (issuing equity signals "we
  think we're overvalued"), firms prefer **internal funds → debt → equity** in that order.

```
firm value
   │        trade-off optimum
   │            ╭─╮
   │         ╭──╯ ╰──╮         ◄─ distress costs bite past here
   │      ╭──╯       ╰───╮
   │   ╭──╯ tax shield    ╰──
   └───┴────────────────────────► leverage (D/E)
```

Higher leverage raises **financial risk** and the cost of equity (levered beta), even as it
adds a cheap tax-advantaged layer — see [credit markets](credit_markets.md) for the lender's
view and [risk management](risk_management.md) for distress dynamics.

## Capital budgeting

Deciding *which projects* to fund. Discount each project's free cash flows at the firm's cost
of capital and rank by value created:

- **NPV (net present value)** — the gold standard. Take every project with `NPV > 0`; it adds
  exactly that much value today. Uses the same DCF machinery as [valuation](valuation.md).
- **IRR (internal rate of return)** — the discount rate where `NPV = 0`. Intuitive ("this
  earns 18%") but breaks with non-conventional cash flows (multiple IRRs) and can rank
  mutually exclusive projects wrong. When NPV and IRR disagree, **trust NPV**.
- **Payback period** — years to recoup the outlay. Simple, ignores time value and everything
  after payback; a crude liquidity screen, not a value test.
- **Profitability index** — `PV of inflows / initial outlay`; useful for ranking under a
  capital constraint.

```python
def npv(rate, cashflows):
    """cashflows[0] is the (negative) initial outlay at t=0."""
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows))

# Project: -1000 now, then +400/yr for 3 yrs, at a 10% cost of capital
print(round(npv(0.10, [-1000, 400, 400, 400]), 2))   # -5.26  -> reject
```

## Payout policy: dividends vs. buybacks

Once a firm generates more cash than it has `NPV > 0` projects, it returns the rest to
shareholders. Two channels (the *investor's* view of these is in [stocks](stocks.md); this is
the *issuer's choice*):

- **Dividends** — regular cash payouts. Sticky (cutting them is punished), signal confidence,
  attract income-focused holders (**clientele effect**).
- **Buybacks** — repurchasing shares. Flexible, tax-efficient (defers gains vs. taxable
  dividends — see [tax strategies](tax_strategies.md)), boost EPS by shrinking the share
  count, and signal management thinks the stock is *cheap*.

**MM dividend irrelevance**: in a frictionless world, payout *form* doesn't matter — a
shareholder can make their own dividend by selling shares. In reality, taxes, signaling, and
agency costs make the choice meaningful. The risk: buying back overvalued stock or borrowing
to fund payouts *destroys* value (a recurring critique of debt-funded buybacks).

## Raising capital

How firms actually source the financing decided above:

- **IPO (initial public offering)** — first sale of shares to the public. An underwriting
  syndicate sets the price, takes a spread, and stabilizes the aftermarket; **underpricing**
  ("the IPO pop") is a persistent cost to the issuer. The investor/primary-vs-secondary view
  is in [stocks](stocks.md).
- **SPAC** — a blank-check shell that IPOs, then merges with a private target (a "de-SPAC").
  Faster and with negotiable terms vs. a traditional IPO, but sponsor promote and redemptions
  often dilute outside holders.
- **Follow-on / secondary offerings & rights issues** — raising more equity post-IPO;
  dilutive, and the announcement often signals overvaluation (pecking order again).
- **Debt issuance** — bonds and loans; cheaper and tax-advantaged but adds fixed claims and
  covenants. See [bonds](bonds.md) and [credit markets](credit_markets.md).
- **Convertibles** — debt that converts to equity; lowers the coupon in exchange for embedded
  optionality (priced with the tools in [quantitative finance](quant_finance.md)).

## M&A mechanics

The *strategic/corporate* side of mergers and acquisitions (the arbitrageur's view is in
[event-driven](event_driven.md)):

- **Rationale & synergies** — cost synergies (overhead, scale) and revenue synergies
  (cross-sell); the latter rarely materialize as promised. Overpaying for synergy is the
  classic value destroyer (**winner's curse**).
- **Consideration** — cash vs. stock vs. mixed. Cash deals signal confidence; all-stock deals
  share both upside *and* the risk that the acquirer's shares are overvalued.
- **Accretion / dilution** — does the deal raise or lower the acquirer's EPS? A quick
  first-pass screen (accretive if the target's earnings yield exceeds the after-tax cost of
  the financing), *not* a value test — NPV still rules.
- **Deal structures** — stock vs. asset purchase, tender offers, mergers; tax and liability
  treatment differ. LBOs (debt-funded buyouts) are detailed in [private
  markets](private_markets.md) and the [LBO method in valuation](valuation.md).

```
Accretion/dilution quick check (all-cash deal):
  pro-forma EPS = (acquirer NI + target NI − after-tax interest on new debt)
                  ─────────────────────────────────────────────────────────
                              acquirer shares (unchanged in cash deal)
  > standalone EPS  →  accretive
```

## Working capital & cash management

The day-to-day side: financing the operating cycle. The **cash conversion cycle** measures
how long cash is tied up:

```
CCC = DIO + DSO − DPO
      (inventory days) + (receivable days) − (payable days)
```

A *shorter* (even negative) CCC frees cash — the Dell/Amazon model: get paid by customers
*before* paying suppliers. Tightly linked to the liquidity ratios in [fundamental
analysis](fundamental_analysis.md).

## Agency & governance

The owners (shareholders) and the managers running the firm aren't the same people —
**agency costs**. Managers may empire-build, hoard cash, or chase pet projects instead of
returning value. Mitigants: equity-linked **compensation**, **board** oversight, debt's
disciplining effect (interest forces cash discipline), and the market for corporate control
(takeover threat). Activist investors exploit the gap — see [event-driven](event_driven.md).

## Where this connects

- [Valuation](valuation.md) — supplies WACC, CAPM, DCF, and the LBO model this page's
  decisions feed into.
- [Fundamental analysis](fundamental_analysis.md) — the financial statements and ratios that
  reveal capital structure, payout, and working-capital health.
- [Stocks](stocks.md) — corporate actions (splits, dividends, buybacks) from the investor's
  side; primary vs. secondary markets.
- [Event-driven](event_driven.md) — the trading/arbitrage view of the M&A, buybacks, and
  spin-offs decided here.
- [Private markets](private_markets.md) — LBOs and the GP's view of capital structure and
  leverage.
- [Credit markets](credit_markets.md) & [Bonds](bonds.md) — the lender's side of debt
  financing, covenants, and distress.
- [Quantitative finance](quant_finance.md) — pricing convertibles and the real options
  embedded in investment decisions.

## Pitfalls

- **IRR over NPV** — IRR can rank projects wrong, multiply with unconventional cash flows, and
  flatter small projects; NPV measures value created, IRR doesn't.
- **EPS accretion ≠ value creation** — a deal or buyback can lift EPS while *destroying* value
  (e.g. cheap debt funding an overpriced acquisition).
- **Debt-funded buybacks at the top** — repurchasing overvalued shares with borrowed money
  raises leverage *and* burns cash; great for short-term EPS, bad for resilience.
- **Synergy optimism** — revenue synergies are routinely overestimated; the premium gets paid
  upfront, the synergies (maybe) arrive later.
- **Ignoring distress costs** — the tax shield is real, but so are bankruptcy, covenant, and
  customer-confidence costs that compound exactly when cash is scarce.
- **Treating cash as free** — idle cash has an opportunity cost; hoarding it is an agency
  symptom, not prudence.

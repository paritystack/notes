# Company Valuation

## Overview

Valuation is the process of determining the intrinsic value of a company or asset. This guide covers various valuation methodologies used by investors and analysts.

## Valuation Methods

### 1. Discounted Cash Flow (DCF)

Present value of future cash flows.

**Steps:**
1. Project future cash flows (5-10 years)
2. Calculate terminal value
3. Discount to present value using WACC
4. Sum all present values

**Formula:**
```
Enterprise Value = Σ(FCF_t / (1 + WACC)^t) + Terminal Value / (1 + WACC)^n
Equity Value = Enterprise Value - Net Debt
Price Per Share = Equity Value / Shares Outstanding
```

**Pros:** Intrinsic value based on fundamentals
**Cons:** Sensitive to assumptions (growth, discount rate)

### 2. Relative Valuation (Multiples)

Compare to similar companies.

**Common Multiples:**
- **P/E Ratio** = Price / Earnings Per Share
- **P/B Ratio** = Price / Book Value Per Share
- **EV/EBITDA** = Enterprise Value / EBITDA
- **P/S Ratio** = Price / Sales Per Share
- **PEG Ratio** = P/E / Growth Rate

**Usage:**
- Compare to industry average
- Compare to historical range
- Find undervalued stocks

**Example:**
```
Company A: P/E = 15, Growth = 20%
Company B: P/E = 25, Growth = 15%

PEG A = 15/20 = 0.75 (undervalued)
PEG B = 25/15 = 1.67 (overvalued)
```

### 3. Asset-Based Valuation

Value company's net assets.

**Methods:**
- **Book Value**: Assets - Liabilities
- **Liquidation Value**: What assets would fetch if sold
- **Replacement Cost**: Cost to rebuild the company

**Best For:** Asset-heavy companies (real estate, manufacturing)
**Poor For:** Tech companies with intangible assets

### 4. Comparable Company Analysis (Comps)

Compare to similar public companies.

**Steps:**
1. Select comparable companies (same industry, size, growth)
2. Calculate valuation multiples
3. Apply average/median multiple to target company

**Example:**
```
Comparable Companies P/E Ratios: 18, 20, 22, 19
Average P/E: 19.75

Target Company EPS: $5.00
Implied Value: $5.00 × 19.75 = $99.75/share
```

### 5. Precedent Transaction Analysis

Value based on M&A transaction prices.

**Steps:**
1. Find recent acquisitions of similar companies
2. Calculate multiples paid
3. Apply to target company

**Note:** Includes control premium (typically 20-40%)

## Key Valuation Metrics

### Free Cash Flow (FCF)

```
FCF = Operating Cash Flow - Capital Expenditures

or

FCF = EBIT(1 - Tax Rate) + Depreciation - CapEx - Change in NWC
```

### Weighted Average Cost of Capital (WACC)

```
WACC = (E/V × Re) + (D/V × Rd × (1 - Tc))

Where:
E = Market value of equity
D = Market value of debt
V = E + D
Re = Cost of equity (CAPM)
Rd = Cost of debt
Tc = Corporate tax rate
```

### Cost of Equity (CAPM)

```
Re = Rf + β(Rm - Rf)

Where:
Rf = Risk-free rate
β = Beta
Rm = Market return
(Rm - Rf) = Equity risk premium
```

### Terminal Value

**Perpetuity Growth Method:**
```
Terminal Value = FCF_final × (1 + g) / (WACC - g)

Where g = perpetual growth rate (typically 2-3%)
```

**Exit Multiple Method:**
```
Terminal Value = EBITDA_final × Exit Multiple
```

## Valuation by Industry

### Technology Companies
- High growth, low/no profits initially
- Use P/S ratio, EV/Revenue
- DCF with high growth rates
- Consider network effects, TAM

### Financial Companies (Banks)
- Use P/B ratio (Price-to-Book)
- ROE important metric
- Dividend Discount Model
- Asset quality matters

### Real Estate (REITs)
- FFO (Funds From Operations)
- P/FFO ratio
- NAV (Net Asset Value)
- Cap rates on properties

### Retailers
- Same-store sales growth
- EV/EBITDA
- P/E ratio
- Inventory turnover

### Cyclical Companies
- Use normalized earnings (average through cycle)
- EV/EBITDA less affected by leverage
- P/B ratio for trough valuation

## Common Mistakes

1. **Garbage In, Garbage Out**: Bad assumptions = bad valuation
2. **Over-Precision**: "Worth $47.23" vs "Worth $40-50"
3. **Ignoring Qualitative Factors**: Management, moat, industry trends
4. **Using Wrong Method**: Each method fits different situations
5. **Circular References**: Using current price in valuation
6. **Ignoring Debt**: Enterprise value vs. equity value confusion
7. **Not Adjusting for One-Time Items**: Normalize earnings

## Valuation Pitfalls

### DCF Pitfalls
- **Terminal value dominates** (often 60-80% of value)
- **Growth rate sensitivity**: Small changes = huge value changes
- **WACC estimation**: Difficult to get right
- **Working capital changes**: Often overlooked

### Multiples Pitfalls
- **Not truly comparable**: Different business models
- **Peak earnings**: Using unsustainable earnings
- **Accounting differences**: GAAP vs. adjusted metrics
- **Ignoring growth**: High P/E may be justified

## Rules of Thumb

### Quick Valuation Checks
- **Rule of 72**: Years to double = 72 / growth rate
- **PEG < 1**: Potentially undervalued
- **P/B < 1**: Trading below book value
- **EV/EBITDA < 8**: Potentially cheap (vary by industry)
- **Dividend Yield > 10-Year Treasury**: Income opportunity

### Sanity Checks
- Compare to historical ranges
- Compare to industry peers
- Check if implied growth is realistic
- Reverse-engineer market expectations

## Best Practices

1. **Use Multiple Methods**: Triangulate value
2. **Conservative Assumptions**: Margin of safety
3. **Scenario Analysis**: Base, bull, bear cases
4. **Sensitivity Analysis**: Test key assumptions
5. **Focus on Drivers**: Growth, margins, returns
6. **Quality Matters**: Good business > cheap price
7. **Margin of Safety**: Buy below intrinsic value (20-30% discount)

## Resources

- **Damodaran's Valuation**: Free resources at pages.stern.nyu.edu
- **"Valuation" by McKinsey**: Comprehensive textbook
- **"The Little Book of Valuation"**: Accessible introduction
- **SEC Filings**: 10-K, 10-Q for actual data
- **Bloomberg/CapIQ**: Professional valuation tools

## Key Takeaways

1. **Valuation is an art AND science** - Not precise
2. **Multiple methods provide range** - Not single number
3. **Assumptions are critical** - Understand sensitivities
4. **Margin of safety required** - Don't pay fair value
5. **Qualitative factors matter** - Not just numbers
6. **Industry context essential** - Different methods for different sectors
7. **Compare to alternatives** - Opportunity cost
8. **Stay conservative** - Better to miss opportunity than lose money

See also: [Fundamental Analysis](fundamental_analysis.md), [Stocks](stocks.md)

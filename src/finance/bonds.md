# Bonds and Fixed Income

## Overview

Bonds are debt securities where investors loan money to an issuer (government, municipality, or corporation) in exchange for periodic interest payments and the return of principal at maturity. Bonds are a cornerstone of fixed income investing and portfolio diversification.

## What is a Bond?

A bond is essentially an IOU - a loan made by an investor to a borrower. The borrower promises to:
- Pay interest (coupon payments) at regular intervals
- Return the principal (face value) at a specified maturity date

**Key Components:**
- **Face Value (Par Value)**: Amount paid at maturity (typically $1,000)
- **Coupon Rate**: Annual interest rate paid on face value
- **Maturity Date**: When principal is repaid
- **Issuer**: Entity borrowing the money
- **Price**: Current market value (can differ from face value)

## Types of Bonds

### Government Bonds

**U.S. Treasury Securities:**
- **T-Bills**: Maturity < 1 year, sold at discount
- **T-Notes**: Maturity 2-10 years, semi-annual coupons
- **T-Bonds**: Maturity 20-30 years, semi-annual coupons
- **TIPS**: Treasury Inflation-Protected Securities

**Characteristics:**
- Backed by U.S. government (lowest credit risk)
- Exempt from state and local taxes
- Benchmark for other interest rates

### Corporate Bonds

Issued by companies to raise capital.

**Grades:**
- **Investment Grade** (BBB- or higher): Lower risk, lower yield
- **High Yield (Junk Bonds)** (BB+ or lower): Higher risk, higher yield

**Types:**
- **Secured Bonds**: Backed by collateral
- **Unsecured Bonds (Debentures)**: Backed only by creditworthiness
- **Convertible Bonds**: Can convert to stock
- **Callable Bonds**: Issuer can redeem early

### Municipal Bonds

Issued by state/local governments.

**Types:**
- **General Obligation (GO)**: Backed by taxing power
- **Revenue Bonds**: Backed by specific revenue source

**Tax Benefits:**
- Interest often exempt from federal taxes
- May be exempt from state/local taxes

### International Bonds

- **Sovereign Bonds**: Issued by foreign governments
- **Eurobonds**: Issued in currency different from issuer's home currency
- **Emerging Market Bonds**: Higher risk, higher potential return

## Bond Pricing

### Price vs. Yield Relationship

**Inverse Relationship:**
- When interest rates rise → bond prices fall
- When interest rates fall → bond prices rise

### Present Value Calculation

```python
def bond_price(face_value, coupon_rate, market_rate, years_to_maturity, payments_per_year=2):
    """
    Calculate bond price using present value

    Args:
        face_value: Par value of bond
        coupon_rate: Annual coupon rate (as decimal)
        market_rate: Current market interest rate (as decimal)
        years_to_maturity: Years until maturity
        payments_per_year: Coupon frequency (2 = semi-annual)
    """
    n_periods = int(years_to_maturity * payments_per_year)
    coupon_payment = (face_value * coupon_rate) / payments_per_year
    period_rate = market_rate / payments_per_year

    # Present value of coupon payments
    pv_coupons = sum([coupon_payment / (1 + period_rate)**i
                      for i in range(1, n_periods + 1)])

    # Present value of face value
    pv_face = face_value / (1 + period_rate)**n_periods

    return pv_coupons + pv_face

# Example: 5% coupon bond, 10 years to maturity, market rate 4%
price = bond_price(1000, 0.05, 0.04, 10, 2)
print(f"Bond Price: ${price:.2f}")
```

### Premium, Par, and Discount

```python
def bond_price_classification(coupon_rate, market_rate):
    """Determine if bond trades at premium, par, or discount"""
    if coupon_rate > market_rate:
        return "Premium (price > $1,000)"
    elif coupon_rate == market_rate:
        return "Par (price = $1,000)"
    else:
        return "Discount (price < $1,000)"

# Coupon rate 6%, market rate 5%
print(bond_price_classification(0.06, 0.05))  # Premium
```

## Bond Yields

### Current Yield

```python
def current_yield(annual_coupon, current_price):
    """
    Current Yield = Annual Coupon Payment / Current Price
    Simple measure of return
    """
    return (annual_coupon / current_price) * 100

# Example
annual_coupon = 50  # $1,000 face × 5% coupon
current_price = 950
cy = current_yield(annual_coupon, current_price)
print(f"Current Yield: {cy:.2f}%")
```

### Yield to Maturity (YTM)

The total return if bond is held to maturity.

```python
def ytm_approximation(face_value, current_price, annual_coupon, years_to_maturity):
    """
    Approximate YTM calculation
    More accurate methods use iterative solutions
    """
    numerator = annual_coupon + ((face_value - current_price) / years_to_maturity)
    denominator = (face_value + current_price) / 2
    return (numerator / denominator) * 100

# Example
ytm = ytm_approximation(1000, 950, 50, 10)
print(f"Approximate YTM: {ytm:.2f}%")
```

### Yield to Call (YTC)

For callable bonds, yield if called at first call date.

```python
def yield_to_call(call_price, current_price, annual_coupon, years_to_call):
    """Calculate approximate yield to call"""
    numerator = annual_coupon + ((call_price - current_price) / years_to_call)
    denominator = (call_price + current_price) / 2
    return (numerator / denominator) * 100
```

## Duration and Convexity

### Macaulay Duration

Weighted average time to receive bond cash flows.

```python
def macaulay_duration(face_value, coupon_rate, market_rate, years_to_maturity,
                      payments_per_year=2):
    """
    Calculate Macaulay Duration
    Measures bond price sensitivity to interest rate changes
    """
    n_periods = int(years_to_maturity * payments_per_year)
    coupon_payment = (face_value * coupon_rate) / payments_per_year
    period_rate = market_rate / payments_per_year

    # Calculate bond price
    price = bond_price(face_value, coupon_rate, market_rate,
                      years_to_maturity, payments_per_year)

    # Calculate weighted present values
    weighted_pv = 0
    for t in range(1, n_periods + 1):
        pv = coupon_payment / (1 + period_rate)**t
        weighted_pv += (t * pv)

    # Add principal payment
    weighted_pv += (n_periods * face_value / (1 + period_rate)**n_periods)

    # Duration in periods
    duration_periods = weighted_pv / price
    # Convert to years
    return duration_periods / payments_per_year

# Example
duration = macaulay_duration(1000, 0.05, 0.04, 10, 2)
print(f"Macaulay Duration: {duration:.2f} years")
```

### Modified Duration

Measures percentage price change for 1% interest rate change.

```python
def modified_duration(macaulay_duration, market_rate, payments_per_year=2):
    """
    Modified Duration = Macaulay Duration / (1 + YTM/n)
    where n is number of payment periods per year
    """
    return macaulay_duration / (1 + market_rate / payments_per_year)

def estimate_price_change(modified_duration, yield_change):
    """
    Estimate percentage price change from yield change
    """
    return -modified_duration * yield_change * 100

# Example: If rates increase by 1%
mod_duration = modified_duration(8.5, 0.04, 2)
price_change = estimate_price_change(mod_duration, 0.01)
print(f"Modified Duration: {mod_duration:.2f}")
print(f"Estimated Price Change: {price_change:.2f}%")
```

### Convexity

Measures curvature in price-yield relationship.

```python
def convexity(face_value, coupon_rate, market_rate, years_to_maturity,
              payments_per_year=2):
    """
    Calculate bond convexity
    Improves price change estimate for large yield changes
    """
    n_periods = int(years_to_maturity * payments_per_year)
    coupon_payment = (face_value * coupon_rate) / payments_per_year
    period_rate = market_rate / payments_per_year

    # Calculate bond price
    price = bond_price(face_value, coupon_rate, market_rate,
                      years_to_maturity, payments_per_year)

    # Calculate convexity
    convexity_sum = 0
    for t in range(1, n_periods + 1):
        pv = coupon_payment / (1 + period_rate)**t
        convexity_sum += t * (t + 1) * pv

    # Add principal
    t = n_periods
    pv_face = face_value / (1 + period_rate)**t
    convexity_sum += t * (t + 1) * pv_face

    # Convexity formula
    convexity_value = convexity_sum / (price * (1 + period_rate)**2)
    return convexity_value / (payments_per_year**2)

def price_change_with_convexity(modified_duration, convexity, yield_change):
    """
    More accurate price change estimate using convexity
    """
    duration_effect = -modified_duration * yield_change
    convexity_effect = 0.5 * convexity * (yield_change**2)
    return (duration_effect + convexity_effect) * 100

# Example
conv = convexity(1000, 0.05, 0.04, 10, 2)
price_chg = price_change_with_convexity(8.4, conv, 0.02)
print(f"Convexity: {conv:.2f}")
print(f"Price Change (with convexity): {price_chg:.2f}%")
```

## Credit Risk

### Credit Ratings

**Major Rating Agencies:**
- Moody's: Aaa, Aa, A, Baa, Ba, B, Caa, Ca, C
- S&P/Fitch: AAA, AA, A, BBB, BB, B, CCC, CC, C, D

**Investment Grade:** BBB-/Baa3 or higher
**High Yield:** BB+/Ba1 or lower

### Credit Spread

```python
def credit_spread(corporate_yield, treasury_yield):
    """
    Credit Spread = Corporate Yield - Treasury Yield
    Compensation for credit risk
    """
    return (corporate_yield - treasury_yield) * 100  # in basis points

# Example
corp_yield = 0.055  # 5.5%
treas_yield = 0.035  # 3.5%
spread = credit_spread(corp_yield, treas_yield)
print(f"Credit Spread: {spread:.0f} basis points")
```

### Default Probability

```python
def probability_of_default(credit_spread, recovery_rate=0.40):
    """
    Estimate default probability from credit spread
    Assumes constant default risk
    """
    return credit_spread / (1 - recovery_rate)

# Example
spread = 0.02  # 2% spread
prob = probability_of_default(spread, 0.40)
print(f"Implied Default Probability: {prob*100:.2f}%")
```

## Yield Curve

### Yield Curve Shapes

**Normal (Upward Sloping):**
- Long-term yields > Short-term yields
- Healthy economy expected

**Inverted (Downward Sloping):**
- Short-term yields > Long-term yields
- Recession warning signal

**Flat:**
- Similar yields across maturities
- Economic uncertainty

### Yield Curve Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_yield_curve(maturities, yields, title="Yield Curve"):
    """
    Plot yield curve

    Args:
        maturities: List of years to maturity
        yields: List of corresponding yields (as percentages)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(maturities, yields, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Years to Maturity')
    plt.ylabel('Yield (%)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

# Example: Normal yield curve
maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
yields = [2.5, 2.7, 3.0, 3.2, 3.4, 3.7, 3.9, 4.0, 4.2, 4.3]

# plot_yield_curve(maturities, yields, "Normal Yield Curve")
```

## Bond Strategies

### Laddering

Spread investments across different maturities.

```python
def bond_ladder(total_investment, maturities):
    """
    Create bond ladder strategy
    Equal amounts in bonds of different maturities
    """
    allocation = total_investment / len(maturities)
    ladder = {f"{mat} year": allocation for mat in maturities}
    return ladder

# Example: $100,000 ladder across 5 years
investment = 100000
mats = [1, 2, 3, 4, 5]
ladder = bond_ladder(investment, mats)
print("Bond Ladder:")
for maturity, amount in ladder.items():
    print(f"  {maturity}: ${amount:,.2f}")
```

### Barbell Strategy

Invest in short and long-term bonds, avoid intermediate.

```python
def barbell_strategy(total_investment, short_weight=0.5):
    """
    Barbell: Combine short and long-term bonds

    Args:
        short_weight: Percentage in short-term (default 50%)
    """
    short_term = total_investment * short_weight
    long_term = total_investment * (1 - short_weight)
    return {
        "short_term (1-3 years)": short_term,
        "long_term (20-30 years)": long_term
    }

barbell = barbell_strategy(100000, 0.5)
print("Barbell Strategy:")
for term, amount in barbell.items():
    print(f"  {term}: ${amount:,.2f}")
```

### Bullet Strategy

Concentrate investments around single maturity date.

```python
def bullet_strategy(total_investment, target_maturity, maturities_around=1):
    """
    Bullet: Concentrate around target maturity

    Args:
        target_maturity: Target maturity year
        maturities_around: Years on either side
    """
    maturities = range(target_maturity - maturities_around,
                      target_maturity + maturities_around + 1)
    allocation = total_investment / len(list(maturities))
    return {f"{mat} years": allocation for mat in maturities}

bullet = bullet_strategy(100000, 10, 1)
print("Bullet Strategy (target 10 years):")
for mat, amount in bullet.items():
    print(f"  {mat}: ${amount:,.2f}")
```

## Bond Risks

### Interest Rate Risk

Risk that rates will rise, decreasing bond value.

**Mitigation:**
- Shorter duration bonds
- Floating rate bonds
- Interest rate hedging

### Credit Risk

Risk of issuer default.

**Mitigation:**
- Diversification
- Investment grade bonds
- Credit analysis

### Inflation Risk

Risk that inflation erodes purchasing power.

**Mitigation:**
- TIPS (Treasury Inflation-Protected Securities)
- Floating rate bonds
- Shorter maturities

### Reinvestment Risk

Risk of reinvesting coupons at lower rates.

**Mitigation:**
- Zero-coupon bonds
- Longer maturities
- Laddering strategy

### Call Risk

Risk that issuer calls bond when rates fall.

**Mitigation:**
- Non-callable bonds
- Call protection periods
- Higher yield compensation

### Liquidity Risk

Risk of not being able to sell quickly at fair price.

**Mitigation:**
- Stick to liquid issues
- Government bonds
- Avoid small corporate issues

## Bond Math Formulas

### Key Formulas

```python
class BondCalculations:
    """Collection of bond calculation formulas"""

    @staticmethod
    def accrued_interest(coupon_rate, face_value, days_since_last_coupon,
                        days_in_period):
        """Calculate accrued interest"""
        annual_coupon = coupon_rate * face_value
        daily_interest = annual_coupon / 365
        return daily_interest * days_since_last_coupon

    @staticmethod
    def dirty_price(clean_price, accrued_interest):
        """Price including accrued interest"""
        return clean_price + accrued_interest

    @staticmethod
    def after_tax_yield(yield_rate, tax_rate):
        """After-tax yield"""
        return yield_rate * (1 - tax_rate)

    @staticmethod
    def tax_equivalent_yield(municipal_yield, tax_rate):
        """Convert tax-free muni yield to taxable equivalent"""
        return municipal_yield / (1 - tax_rate)

    @staticmethod
    def real_yield(nominal_yield, inflation_rate):
        """Inflation-adjusted yield (Fisher equation)"""
        return ((1 + nominal_yield) / (1 + inflation_rate)) - 1

# Examples
calc = BondCalculations()

# Accrued interest
accrued = calc.accrued_interest(0.05, 1000, 45, 180)
print(f"Accrued Interest: ${accrued:.2f}")

# Tax equivalent yield
muni_yield = 0.04  # 4% tax-free
tax_rate = 0.30  # 30% tax bracket
equivalent = calc.tax_equivalent_yield(muni_yield, tax_rate)
print(f"Tax Equivalent Yield: {equivalent*100:.2f}%")

# Real yield
real = calc.real_yield(0.05, 0.02)
print(f"Real Yield: {real*100:.2f}%")
```

## Treasury Inflation-Protected Securities (TIPS)

### How TIPS Work

```python
def tips_value(original_principal, cpi_start, cpi_current, coupon_rate):
    """
    Calculate TIPS adjusted principal and interest

    Principal adjusts with CPI
    Coupon rate applied to adjusted principal
    """
    inflation_ratio = cpi_current / cpi_start
    adjusted_principal = original_principal * inflation_ratio
    semi_annual_interest = (adjusted_principal * coupon_rate) / 2

    return {
        "adjusted_principal": adjusted_principal,
        "semi_annual_interest": semi_annual_interest,
        "annual_interest": semi_annual_interest * 2
    }

# Example
tips = tips_value(1000, 250, 265, 0.02)
print(f"Original Principal: $1,000")
print(f"Adjusted Principal: ${tips['adjusted_principal']:.2f}")
print(f"Annual Interest: ${tips['annual_interest']:.2f}")
```

## Bond Portfolio Metrics

### Portfolio Duration

```python
def portfolio_duration(bonds_data):
    """
    Calculate weighted average duration of bond portfolio

    bonds_data: List of dicts with 'value' and 'duration'
    """
    total_value = sum(bond['value'] for bond in bonds_data)
    weighted_duration = sum(bond['value'] * bond['duration']
                           for bond in bonds_data)
    return weighted_duration / total_value

# Example portfolio
portfolio = [
    {'value': 30000, 'duration': 5.2},  # Bond 1
    {'value': 40000, 'duration': 7.8},  # Bond 2
    {'value': 30000, 'duration': 3.1},  # Bond 3
]

port_duration = portfolio_duration(portfolio)
print(f"Portfolio Duration: {port_duration:.2f} years")
```

### Portfolio Yield

```python
def portfolio_yield(bonds_data):
    """
    Calculate weighted average yield

    bonds_data: List of dicts with 'value' and 'yield'
    """
    total_value = sum(bond['value'] for bond in bonds_data)
    weighted_yield = sum(bond['value'] * bond['yield']
                        for bond in bonds_data)
    return (weighted_yield / total_value) * 100

# Example
portfolio = [
    {'value': 30000, 'yield': 0.04},
    {'value': 40000, 'yield': 0.055},
    {'value': 30000, 'yield': 0.035},
]

port_yield = portfolio_yield(portfolio)
print(f"Portfolio Yield: {port_yield:.2f}%")
```

## Best Practices

### Bond Selection Criteria

1. **Credit Quality**: Check ratings from multiple agencies
2. **Yield**: Compare to benchmarks and similar bonds
3. **Duration**: Match to investment timeline
4. **Liquidity**: Ensure adequate trading volume
5. **Call Features**: Understand call provisions
6. **Tax Status**: Consider tax implications

### Risk Management

1. **Diversify**: Across issuers, sectors, maturities
2. **Monitor**: Track credit ratings and market conditions
3. **Rebalance**: Adjust as rates and goals change
4. **Understand**: Know what you own and why

### Common Mistakes to Avoid

1. Chasing yield without considering risk
2. Ignoring interest rate sensitivity (duration)
3. Over-concentrating in single issuer
4. Neglecting tax implications
5. Not understanding call features
6. Buying illiquid bonds
7. Ignoring inflation impact

## Resources

### Data Sources

- **U.S. Treasury**: TreasuryDirect.gov
- **FINRA**: Bond pricing and trade data
- **Bloomberg**: Professional bond data
- **WSJ**: Bond market overview

### Bond Indices

- Bloomberg Barclays U.S. Aggregate Bond Index
- ICE BofA U.S. Corporate Index
- S&P U.S. Treasury Bond Index

### Educational Resources

- SIFMA (Securities Industry and Financial Markets Association)
- Investopedia Bond Section
- CFA Institute Fixed Income Resources

## Key Takeaways

1. **Bonds provide income** through regular coupon payments
2. **Inverse relationship** between bond prices and interest rates
3. **Duration measures** interest rate sensitivity
4. **Credit risk** varies by issuer and rating
5. **Diversification** reduces risk in bond portfolios
6. **Tax treatment** varies by bond type
7. **Yield curve** provides economic insights
8. **Multiple strategies** available for different goals

## Next Steps

1. Understand your investment objectives and timeline
2. Assess your risk tolerance
3. Learn to read bond quotes and analyze yields
4. Start with government bonds for lower risk
5. Build a diversified bond portfolio
6. Monitor credit ratings and market conditions
7. Consider tax implications in your strategy
8. Rebalance as needed to maintain target allocation

Remember: Bonds are generally less volatile than stocks but still carry risks. Understand these risks and how bonds fit into your overall investment strategy.

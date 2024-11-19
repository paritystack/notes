# General

## Sharpe Ratio

The Sharpe Ratio is a widely used metric in finance to evaluate the performance of an investment by measuring the excess return per unit of risk. It is calculated by dividing the difference between the return of the investment and the risk-free rate by the standard deviation of the investment's returns.

$$
SR = \frac{R_p - R_f}{\sigma_p}
$$

Where:
- \( R_p \) is the return of the portfolio
- \( R_f \) is the risk-free rate
- \( \sigma_p \) is the standard deviation of the portfolio's returns

### Sample Scenario

To better understand the Sharpe Ratio, let's consider a practical example.

Assume the following data for a portfolio:
- Portfolio return (\( R_p \)): 12% or 0.12
- Risk-free rate (\( R_f \)): 2% or 0.02
- Portfolio standard deviation (\( \sigma_p \)): 8% or 0.08

Using the Sharpe Ratio formula:

$$
SR = \frac{R_p - R_f}{\sigma_p}
$$

Substituting the values:

$$
SR = \frac{0.12 - 0.02}{0.08} = \frac{0.10}{0.08} = 1.25
$$

In this scenario, the Sharpe Ratio is 1.25, indicating that the portfolio generates 1.25 units of excess return for each unit of risk taken.

## Kelly Criterion

The Kelly Criterion is a formula used to determine the optimal size of a series of bets. It calculates the ratio of edge over odds, helping to maximize the growth of capital over time. The formula is expressed as \(k\), where \(p\) and \(q\) are the probabilities of winning and losing, respectively.

$$
k = \frac{p - q}{o}
$$

Where:
- \(p\) is the probability of winning
- \(q\) is the probability of losing
- \(o\) is the odds of the bet

### Sample Scenario

Consider a scenario to illustrate the Kelly Criterion.

Assume the following data for a bet:
- Probability of winning (\( p \)): 60% or 0.60
- Probability of losing (\( q \)): 40% or 0.40
- Odds of the bet (\( o \)): 2:1

Using the Kelly Criterion formula:

$$
k = \frac{p - q}{o}
$$

Substituting the values:

$$
k = \frac{0.60 - 0.40}{2} = \frac{0.20}{2} = 0.10
$$

In this scenario, the Kelly Criterion suggests betting 10% of your bankroll. For example, with a $1000 bankroll, you should bet $100.

## Pot Geometry

Pot Geometry is a strategic betting approach where a consistent fraction of the pot is wagered on each round. Also known as geometric bet sizing, this strategy aims to maximize the amount of money an opponent contributes to the pot.

### Detailed Explanation of Pot Geometry

Pot Geometry is particularly useful in games like poker, where managing the pot size and betting strategically can significantly impact outcomes. By betting a fixed fraction of the pot on each round, the pot size grows exponentially, maximizing potential winnings while managing risk.

### Key Concepts

1. **Fractional Betting**: A fixed fraction of the current pot size is bet on each round. For instance, if the fraction is 50%, then 50% of the current pot size is added to the pot each round.

2. **Exponential Growth**: Consistent fractional betting leads to exponential growth of the pot size, potentially increasing winnings over multiple rounds.

3. **Risk Management**: Pot Geometry ensures bets are proportional to the current pot size, preventing over-betting and large losses.

### Example Scenario

Consider an example to demonstrate Pot Geometry:

- Initial pot size: $100
- Fraction of pot to bet: 50% (0.50)

#### Round 1:
- Current pot size: $100
- Bet size: 50% of $100 = $50
- New pot size: $100 + $50 = $150

#### Round 2:
- Current pot size: $150
- Bet size: 50% of $150 = $75
- New pot size: $150 + $75 = $225

#### Round 3:
- Current pot size: $225
- Bet size: 50% of $225 = $112.50
- New pot size: $225 + $112.50 = $337.50

As shown, the pot size grows exponentially with each betting round.

### Advantages of Pot Geometry

1. **Consistent Growth**: The pot grows steadily, allowing for potentially higher winnings over multiple rounds.
2. **Controlled Risk**: Betting a fraction of the pot controls risk, keeping it proportional to the current pot size.
3. **Strategic Flexibility**: Players can adjust the betting fraction based on confidence and game dynamics.

### Conclusion

Pot Geometry is a powerful betting strategy that leverages exponential growth and risk management principles. By consistently betting a fraction of the pot, players can maximize potential winnings while maintaining controlled risk. This strategy is particularly effective in poker, where strategic pot management can significantly influence long-term success.

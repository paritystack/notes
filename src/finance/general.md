# General

## Sharpe Ratio

The Sharpe Ratio is a widely used metric in finance to evaluate the performance of an investment by measuring the excess return per unit of risk. It is calculated by dividing the difference between the return of the investment and the risk-free rate by the standard deviation of the investment's returns.

$$
SR = \frac{R_p - R_f}{\sigma_p}
$$

Where:
- \( R_p \) is the return of the portfolio
- \( R_f \) is the risk-free rate (usually the return of a benchmark like the S&P 500)
- \( \sigma_p \) is the standard deviation of the portfolio's returns

## Calculating Standard Deviation of Returns

The standard deviation of returns is a measure of the dispersion or variability of investment returns over a period of time. It helps in understanding the risk associated with the investment. Here is a step-by-step process to calculate the standard deviation of returns:

1. **Collect the Returns Data**: Gather the periodic returns of the investment. These returns can be daily, monthly, or yearly.

2. **Calculate the Mean Return**: Compute the average return over the period.

$$
\bar{R} = \frac{\sum_{i=1}^{n} R_i}{n}
$$

Where:
- \( \bar{R} \) is the mean return
- \( R_i \) is the return for period \( i \)
- \( n \) is the number of periods

3. **Compute the Variance**: Calculate the variance by finding the average of the squared differences between each return and the mean return.

$$
\sigma^2 = \frac{\sum_{i=1}^{n} (R_i - \bar{R})^2}{n}
$$

Where:
- \( \sigma^2 \) is the variance

4. **Calculate the Standard Deviation**: Take the square root of the variance to get the standard deviation.

$$
\sigma = \sqrt{\sigma^2}
$$

Where:
- \( \sigma \) is the standard deviation

### Sample Calculation

Assume the following monthly returns for an investment over 5 months: 2%, 3%, -1%, 4%, and 5%.

1. **Mean Return**:

$$
\bar{R} = \frac{2 + 3 - 1 + 4 + 5}{5} = \frac{13}{5} = 2.6\%
$$

2. **Variance**:

$$
\sigma^2 = \frac{(2 - 2.6)^2 + (3 - 2.6)^2 + (-1 - 2.6)^2 + (4 - 2.6)^2 + (5 - 2.6)^2}{5}
$$

$$
\sigma^2 = \frac{(-0.6)^2 + (0.4)^2 + (-3.6)^2 + (1.4)^2 + (2.4)^2}{5}
$$

$$
\sigma^2 = \frac{0.36 + 0.16 + 12.96 + 1.96 + 5.76}{5} = \frac{21.2}{5} = 4.24
$$

3. **Standard Deviation**:

$$
\sigma = \sqrt{4.24} \approx 2.06\%
$$

In this example, the standard deviation of the returns is approximately 2.06%, indicating the variability of the investment returns over the period.


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

### Understanding Odds of a Bet

The odds of a bet represent the ratio of the probability of winning to the probability of losing. They are a crucial component in betting strategies, including the Kelly Criterion. Odds can be expressed in different formats, such as fractional, decimal, and moneyline.

#### Fractional Odds

Fractional odds are commonly used in the UK and are represented as a fraction (e.g., 2/1). The numerator (first number) represents the potential profit, while the denominator (second number) represents the stake. For example, 2/1 odds mean you win $2 for every $1 bet.

#### Decimal Odds

Decimal odds are popular in Europe and Australia. They are represented as a decimal number (e.g., 3.00). The decimal number includes the original stake, so the total payout is calculated by multiplying the stake by the decimal odds. For example, 3.00 odds mean a $1 bet returns $3 (including the $1 stake).

#### Moneyline Odds

Moneyline odds are commonly used in the United States and can be positive or negative. Positive moneyline odds (e.g., +200) indicate how much profit you make on a $100 bet. Negative moneyline odds (e.g., -150) indicate how much you need to bet to win $100.

### Calculating Odds

To calculate the odds of a bet, you need to know the probabilities of winning and losing. The formula for calculating fractional odds is:

$$
\text{Odds} = \frac{p}{q}
$$

Where:
- \( p \) is the probability of winning
- \( q \) is the probability of losing

For example, if the probability of winning is 60% (0.60) and the probability of losing is 40% (0.40), the fractional odds are:

$$
\text{Odds} = \frac{0.60}{0.40} = \frac{3}{2} = 1.5
$$

To convert fractional odds to decimal odds, add 1 to the fractional odds:

$$
\text{Decimal Odds} = \text{Fractional Odds} + 1
$$

Using the previous example:

$$
\text{Decimal Odds} = 1.5 + 1 = 2.5
$$

To convert fractional odds to moneyline odds:
- If the fractional odds are greater than 1 (e.g., 2/1), the moneyline odds are positive: \( \text{Moneyline Odds} = \text{Fractional Odds} \times 100 \)
- If the fractional odds are less than 1 (e.g., 1/2), the moneyline odds are negative: \( \text{Moneyline Odds} = -\left(\frac{100}{\text{Fractional Odds}}\right) \)

Using the previous example (1.5 fractional odds):

$$
\text{Moneyline Odds} = 1.5 \times 100 = +150
$$

Understanding and calculating the odds of a bet is essential for making informed betting decisions and optimizing strategies like the Kelly Criterion.


Using the Kelly Criterion formula:

$$
k = \frac{p - q}{o}
$$

Substituting the values:

$$
k = \frac{0.60 - 0.40}{2} = \frac{0.20}{2} = 0.10
$$

In this scenario, the Kelly Criterion suggests betting 10% of your bankroll. For example, with a $1000 bankroll, you should bet $100.


## Intuition of the Kelly Criterion

The Kelly Criterion is a mathematical formula used to determine the optimal size of a series of bets to maximize the logarithm of wealth over time. It is particularly useful in scenarios where the goal is to grow wealth exponentially while managing risk. The intuition behind the Kelly Criterion can be broken down into several key concepts:

### Key Concepts

1. **Maximizing Growth**: The Kelly Criterion aims to maximize the long-term growth rate of your bankroll. By betting a fraction of your bankroll that is proportional to the edge you have over the odds, you can achieve exponential growth over time.

2. **Balancing Risk and Reward**: The formula balances the potential reward of a bet with the risk of losing. By betting too much, you risk significant losses that can deplete your bankroll. By betting too little, you miss out on potential gains. The Kelly Criterion finds the optimal balance.

3. **Proportional Betting**: The Kelly Criterion suggests betting a fraction of your bankroll that is proportional to your edge. This means that as your edge increases, the fraction of your bankroll you should bet also increases. Conversely, if your edge decreases, you should bet a smaller fraction.

4. **Logarithmic Utility**: The Kelly Criterion is based on the concept of logarithmic utility, which means that the utility (or satisfaction) derived from wealth increases logarithmically. This approach ensures that the strategy is focused on long-term growth rather than short-term gains.

### Example Scenario

Consider a scenario where you have a 60% chance of winning a bet (probability \( p = 0.60 \)) and a 40% chance of losing (probability \( q = 0.40 \)). The odds offered are 2:1 (decimal odds of 2.0).

Using the Kelly Criterion formula:

$$
k = \frac{p - q}{o}
$$

Substituting the values:

$$
k = \frac{0.60 - 0.40}{2} = \frac{0.20}{2} = 0.10
$$

In this scenario, the Kelly Criterion suggests betting 10% of your bankroll. For example, with a $1000 bankroll, you should bet $100.

### Advantages of the Kelly Criterion

1. **Optimal Growth**: The Kelly Criterion maximizes the long-term growth rate of your bankroll, ensuring that you achieve exponential growth over time.
2. **Risk Management**: By betting a fraction of your bankroll, the Kelly Criterion helps manage risk and prevent significant losses.
3. **Adaptability**: The formula adjusts the bet size based on the edge, allowing for flexible and adaptive betting strategies.

### Conclusion

The Kelly Criterion is a powerful tool for optimizing bet sizes and maximizing long-term growth. By balancing risk and reward and focusing on proportional betting, the Kelly Criterion provides a strategic approach to betting that can lead to exponential wealth growth over time. Understanding the intuition behind the Kelly Criterion can help you make more informed and strategic betting decisions.


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

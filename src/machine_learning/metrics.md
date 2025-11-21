# Machine Learning Metrics

Evaluation metrics are fundamental to machine learning, providing quantitative measures to assess model performance. The choice of metric depends on the problem type, business objectives, data distribution, and the cost of different types of errors. This guide covers the most important metrics across various ML tasks with detailed explanations, mathematical formulations, intuitions, and implementation examples.

## Table of Contents

- [Regression Metrics](#regression-metrics)
- [Classification Metrics](#classification-metrics)
- [Multi-class Classification Strategies](#multi-class-classification-strategies)
- [Ranking and Recommendation Metrics](#ranking-and-recommendation-metrics)
- [NLP Metrics](#nlp-metrics)
- [Clustering Metrics](#clustering-metrics)
- [Object Detection Metrics](#object-detection-metrics)
- [Choosing the Right Metric](#choosing-the-right-metric)
- [Resources](#resources)

---

## Regression Metrics

Regression metrics evaluate how well a model predicts continuous values. The choice of metric affects how the model treats outliers and large errors.

### Mean Absolute Error (MAE)

**Mathematical Formula:**
```
MAE = (1/n) × Σ|y_i - ŷ_i|
```

**Intuition:**
MAE measures the average magnitude of errors without considering their direction. It treats all errors equally - a 5-unit error has exactly 5 times the impact of a 1-unit error. MAE is robust to outliers because it doesn't square the errors.

**When to Use:**
- When outliers should not heavily influence the metric
- When all errors should be weighted equally
- When you want an interpretable metric in the same units as your target variable
- Examples: predicting house prices where a few mansions shouldn't dominate, temperature forecasting

**Pros:**
- Easy to interpret and explain
- Robust to outliers
- Same units as the target variable

**Cons:**
- Not differentiable at zero (can cause issues with gradient-based optimization)
- Doesn't penalize large errors more heavily

```python
import numpy as np
from sklearn.metrics import mean_absolute_error

# Example: Housing price prediction
y_true = np.array([250000, 300000, 275000, 320000, 290000])  # Actual prices
y_pred = np.array([245000, 310000, 270000, 330000, 285000])  # Predicted prices

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: ${mae:,.0f}")  # Output: MAE: $7,000

# Manual implementation
mae_manual = np.mean(np.abs(y_true - y_pred))
print(f"MAE (manual): ${mae_manual:,.0f}")
```

### Mean Squared Error (MSE)

**Mathematical Formula:**
```
MSE = (1/n) × Σ(y_i - ŷ_i)²
```

**Intuition:**
MSE squares each error before averaging, which means larger errors are penalized disproportionately more than smaller errors. A 10-unit error contributes 100 to MSE, while two 5-unit errors contribute only 50. This makes MSE very sensitive to outliers but useful when large errors are particularly undesirable.

**When to Use:**
- When large errors are significantly worse than small errors
- When outliers represent genuine data points that need attention
- When training neural networks (smooth gradient)
- Examples: financial forecasting where large errors have severe consequences

**Pros:**
- Differentiable everywhere (good for optimization)
- Penalizes large errors heavily
- Well-studied mathematical properties

**Cons:**
- Not in the same units as the target (squared units)
- Very sensitive to outliers
- Harder to interpret than MAE

```python
from sklearn.metrics import mean_squared_error

# Example: Temperature prediction
y_true = np.array([22.5, 23.1, 21.8, 24.2, 22.9])  # Actual temperature (°C)
y_pred = np.array([22.3, 25.0, 21.9, 24.0, 22.8])  # Predicted temperature

mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse:.4f} °C²")  # Output: MSE: 0.7240 °C²

# Manual implementation
mse_manual = np.mean((y_true - y_pred) ** 2)

# Compare with MAE
mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.4f} °C")  # MAE is more interpretable
```

### Root Mean Squared Error (RMSE)

**Mathematical Formula:**
```
RMSE = √[MSE] = √[(1/n) × Σ(y_i - ŷ_i)²]
```

**Intuition:**
RMSE is the square root of MSE, bringing the metric back to the original units of the target variable. It inherits MSE's sensitivity to outliers while being more interpretable. RMSE can be thought of as the standard deviation of prediction errors.

**When to Use:**
- When you want MSE's properties but in interpretable units
- When comparing models on the same dataset
- When large errors are particularly costly
- Standard metric for many regression competitions

**Pros:**
- Same units as the target variable (interpretable)
- Penalizes large errors
- Differentiable (good for optimization)

**Cons:**
- Still sensitive to outliers
- Scale-dependent (can't compare across different datasets)

```python
from sklearn.metrics import mean_squared_error
import numpy as np

# Example: Stock price prediction
y_true = np.array([150.2, 152.5, 148.9, 153.1, 151.0])
y_pred = np.array([149.5, 153.0, 149.2, 152.8, 151.5])

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE: ${rmse:.2f}")  # Output: RMSE: $0.52

# Alternative using sklearn
rmse_sklearn = mean_squared_error(y_true, y_pred, squared=False)
print(f"RMSE (sklearn): ${rmse_sklearn:.2f}")

# Relationship between RMSE and MAE
mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: ${mae:.2f}")
print(f"RMSE/MAE ratio: {rmse/mae:.2f}")  # RMSE >= MAE always
```

### R² Score (Coefficient of Determination)

**Mathematical Formula:**
```
R² = 1 - (SS_res / SS_tot)
SS_res = Σ(y_i - ŷ_i)²  (Residual Sum of Squares)
SS_tot = Σ(y_i - ȳ)²     (Total Sum of Squares)
```

**Intuition:**
R² represents the proportion of variance in the target variable that is explained by the model. R² = 1 means perfect predictions, R² = 0 means the model is no better than predicting the mean, and R² < 0 means the model is worse than predicting the mean. Think of it as "how much better is my model than just guessing the average?"

**When to Use:**
- When you want to understand how much variance your model explains
- When comparing models on the same dataset
- When you need a scale-independent metric (0 to 1 range for good models)
- Reporting to stakeholders who want percentage of variance explained

**Pros:**
- Intuitive interpretation (percentage of variance explained)
- Scale-independent
- Standard in statistical analysis

**Cons:**
- Can be negative (confusing for non-statisticians)
- Always increases when adding features (even irrelevant ones)
- Sensitive to outliers

```python
from sklearn.metrics import r2_score
import numpy as np

# Example: Predicting exam scores
y_true = np.array([85, 92, 78, 95, 88, 76, 91, 83])
y_pred = np.array([84, 90, 80, 94, 87, 78, 89, 85])

r2 = r2_score(y_true, y_pred)
print(f"R² Score: {r2:.4f}")  # Output: R² Score: 0.8869
print(f"Variance explained: {r2*100:.2f}%")

# Manual implementation
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2_manual = 1 - (ss_res / ss_tot)
print(f"R² (manual): {r2_manual:.4f}")

# Comparing with baseline (predicting mean)
baseline_pred = np.full_like(y_true, np.mean(y_true))
r2_baseline = r2_score(y_true, baseline_pred)
print(f"Baseline R²: {r2_baseline:.4f}")  # Always 0.0

# Bad model example (R² can be negative)
bad_pred = np.array([50, 50, 50, 50, 50, 50, 50, 50])  # Predicting constant
r2_bad = r2_score(y_true, bad_pred)
print(f"Bad model R²: {r2_bad:.4f}")  # Negative value
```

### Adjusted R²

**Mathematical Formula:**
```
Adjusted R² = 1 - [(1 - R²) × (n - 1) / (n - p - 1)]

where:
n = number of samples
p = number of predictors/features
```

**Intuition:**
Adjusted R² penalizes the addition of features that don't improve the model significantly. While regular R² always increases when adding features (even random ones), Adjusted R² only increases if the new feature improves the model more than would be expected by chance. This prevents overfitting through feature bloat.

**When to Use:**
- When comparing models with different numbers of features
- When performing feature selection
- When you want to prevent overfitting in linear models
- In statistical reporting and analysis

**Pros:**
- Accounts for model complexity
- Better for model comparison with different feature counts
- Prevents spurious feature addition

**Cons:**
- Less intuitive than R²
- Not available in all ML libraries (mainly statistical packages)
- Can still be negative

```python
import numpy as np
from sklearn.metrics import r2_score

def adjusted_r2_score(y_true, y_pred, n_features):
    """
    Calculate Adjusted R² Score.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        n_features: Number of features in the model

    Returns:
        Adjusted R² score
    """
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return adjusted_r2

# Example: Comparing models with different feature counts
y_true = np.array([100, 120, 95, 110, 105, 115, 98, 125, 102, 118])

# Model 1: 3 features, R² = 0.85
y_pred_model1 = np.array([98, 119, 97, 108, 106, 116, 99, 123, 101, 117])
r2_m1 = r2_score(y_true, y_pred_model1)
adj_r2_m1 = adjusted_r2_score(y_true, y_pred_model1, n_features=3)

# Model 2: 7 features, R² = 0.87 (only slightly better)
y_pred_model2 = np.array([99, 120, 96, 109, 105, 115, 98, 124, 102, 118])
r2_m2 = r2_score(y_true, y_pred_model2)
adj_r2_m2 = adjusted_r2_score(y_true, y_pred_model2, n_features=7)

print("Model 1 (3 features):")
print(f"  R²: {r2_m1:.4f}")
print(f"  Adjusted R²: {adj_r2_m1:.4f}")

print("\nModel 2 (7 features):")
print(f"  R²: {r2_m2:.4f}")
print(f"  Adjusted R²: {adj_r2_m2:.4f}")

print("\n✓ Model 1 is better despite lower R² (simpler and nearly as good)")
```

### Mean Absolute Percentage Error (MAPE)

**Mathematical Formula:**
```
MAPE = (100/n) × Σ|((y_i - ŷ_i) / y_i)|
```

**Intuition:**
MAPE expresses the average prediction error as a percentage of the actual value. A MAPE of 10% means predictions are off by 10% on average. This makes errors relative to the scale of the data - a $100 error on a $1000 item (10%) is treated the same as a $1000 error on a $10,000 item (10%).

**When to Use:**
- When you want scale-independent error measurement
- When comparing forecasts across different scales
- When stakeholders want percentage-based metrics
- Examples: sales forecasting, demand prediction

**Pros:**
- Easy to interpret (percentage error)
- Scale-independent
- Intuitive for business stakeholders

**Cons:**
- Undefined when actual values are zero
- Asymmetric (penalizes under-predictions more than over-predictions)
- Can be misleading with small denominators

```python
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate MAPE, handling zero values.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Filter out zero values to avoid division by zero
    non_zero_mask = y_true != 0

    if not np.any(non_zero_mask):
        return np.nan

    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) /
                          y_true[non_zero_mask])) * 100
    return mape

# Example: Sales forecasting across different product categories
y_true = np.array([1000, 5000, 500, 10000, 2500])  # Actual sales
y_pred = np.array([1100, 4800, 550, 9500, 2400])   # Predicted sales

mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"MAPE: {mape:.2f}%")  # Output: MAPE: ~6.40%

# Compare with MAE (not scale-independent)
mae = np.mean(np.abs(y_true - y_pred))
print(f"MAE: {mae:.2f}")  # Less interpretable across different scales

# Demonstrating the asymmetry problem
print("\nMAPE Asymmetry Demonstration:")
# Over-prediction
actual_1, pred_1 = 100, 150  # 50% over
mape_1 = abs((actual_1 - pred_1) / actual_1) * 100
print(f"Over-prediction (100→150): MAPE = {mape_1:.1f}%")

# Under-prediction
actual_2, pred_2 = 150, 100  # 33% under (same absolute error)
mape_2 = abs((actual_2 - pred_2) / actual_2) * 100
print(f"Under-prediction (150→100): MAPE = {mape_2:.1f}%")
print("⚠ Same absolute error, different MAPE!")
```

### Huber Loss

**Mathematical Formula:**
```
Huber(y, ŷ) = { ½(y - ŷ)²           if |y - ŷ| ≤ δ
              { δ|y - ŷ| - ½δ²      otherwise

where δ is a hyperparameter (threshold)
```

**Intuition:**
Huber loss combines the best of MAE and MSE. For small errors (below threshold δ), it behaves like MSE (quadratic), providing smooth gradients. For large errors (above δ), it behaves like MAE (linear), making it robust to outliers. It's like saying "normal errors get squared, but outliers just get absolute value."

**When to Use:**
- When you have outliers but still want to penalize large errors somewhat
- When training regression models with noisy data
- When you want robustness without completely ignoring large errors
- Common in reinforcement learning and robust regression

**Pros:**
- Robust to outliers (more than MSE)
- Differentiable everywhere (better than MAE for optimization)
- Balanced approach between MAE and MSE

**Cons:**
- Requires tuning the δ hyperparameter
- More complex to interpret than MAE or MSE

```python
import numpy as np
import matplotlib.pyplot as plt

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Calculate Huber loss.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        delta: Threshold for switching between quadratic and linear

    Returns:
        Average Huber loss
    """
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta

    squared_loss = 0.5 * (error ** 2)
    linear_loss = delta * np.abs(error) - 0.5 * (delta ** 2)

    return np.mean(np.where(is_small_error, squared_loss, linear_loss))

# Example: Predicting with outliers
y_true = np.array([10, 12, 11, 13, 12, 10, 50, 11])  # One outlier: 50
y_pred = np.array([10, 11, 12, 12, 13, 10, 12, 11])  # Reasonable predictions

# Compare different losses
mse = np.mean((y_true - y_pred) ** 2)
mae = np.mean(np.abs(y_true - y_pred))
huber = huber_loss(y_true, y_pred, delta=1.0)

print(f"MSE: {mse:.2f}  <- Heavily affected by outlier")
print(f"MAE: {mae:.2f}  <- Robust but treats all errors equally")
print(f"Huber Loss (δ=1.0): {huber:.2f}  <- Balanced approach")

# Using PyTorch's Huber Loss
import torch
import torch.nn as nn

huber_fn = nn.HuberLoss(delta=1.0)
y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)
huber_pytorch = huber_fn(y_pred_tensor, y_true_tensor)
print(f"Huber Loss (PyTorch): {huber_pytorch.item():.2f}")

# Visualizing Huber vs MSE vs MAE
errors = np.linspace(-5, 5, 100)
mse_losses = 0.5 * errors**2
mae_losses = np.abs(errors)
huber_losses = np.where(np.abs(errors) <= 1.0,
                        0.5 * errors**2,
                        1.0 * np.abs(errors) - 0.5)

# Uncomment to visualize:
# plt.figure(figsize=(10, 6))
# plt.plot(errors, mse_losses, label='MSE (0.5×error²)', linewidth=2)
# plt.plot(errors, mae_losses, label='MAE (|error|)', linewidth=2)
# plt.plot(errors, huber_losses, label='Huber (δ=1.0)', linewidth=2)
# plt.xlabel('Prediction Error')
# plt.ylabel('Loss')
# plt.title('Comparison of Regression Loss Functions')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()
```

---

## Classification Metrics

Classification metrics evaluate how well a model assigns instances to discrete categories. Understanding the confusion matrix is fundamental to all classification metrics.

### Confusion Matrix

**Structure:**
```
                   Predicted
                 Pos      Neg
Actual  Pos  |   TP   |   FN  |
        Neg  |   FP   |   TN  |

TP (True Positive): Correctly predicted positive
TN (True Negative): Correctly predicted negative
FP (False Positive): Incorrectly predicted positive (Type I Error)
FN (False Negative): Incorrectly predicted negative (Type II Error)
```

**Intuition:**
The confusion matrix shows all possible outcomes of binary classification. Each cell tells a story:
- TP: We said "yes" and we were right
- TN: We said "no" and we were right
- FP: We said "yes" but we were wrong (false alarm)
- FN: We said "no" but we were wrong (missed detection)

All other classification metrics are derived from these four numbers.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# Example: Medical diagnosis (disease detection)
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])  # 1=disease, 0=healthy
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])  # Model predictions

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
print()

# Extract values
tn, fp, fn, tp = cm.ravel()
print(f"True Positives (TP): {tp}  - Correctly identified sick patients")
print(f"True Negatives (TN): {tn}  - Correctly identified healthy patients")
print(f"False Positives (FP): {fp} - Healthy patients wrongly diagnosed as sick")
print(f"False Negatives (FN): {fn} - Sick patients wrongly diagnosed as healthy")

# Visualize confusion matrix
# disp = ConfusionMatrixDisplay(confusion_matrix=cm,
#                                display_labels=['Healthy', 'Disease'])
# disp.plot(cmap='Blues')
# plt.title('Medical Diagnosis Confusion Matrix')
# plt.show()

# Multi-class confusion matrix example
y_true_multi = np.array([0, 1, 2, 0, 1, 2, 1, 2, 0, 1])
y_pred_multi = np.array([0, 2, 2, 0, 1, 2, 1, 1, 0, 1])
cm_multi = confusion_matrix(y_true_multi, y_pred_multi)

print("\nMulti-class Confusion Matrix (3 classes):")
print(cm_multi)
print("Diagonal elements = correct predictions")
print("Off-diagonal = misclassifications")
```

### Accuracy

**Mathematical Formula:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
         = Correct Predictions / Total Predictions
```

**Intuition:**
Accuracy is the proportion of correct predictions out of all predictions. It's the most intuitive metric: "how often is the classifier correct?" However, it can be misleading with imbalanced datasets. A model that predicts "no cancer" 99% of the time achieves 99% accuracy if only 1% of patients have cancer, but it's useless.

**When to Use:**
- When classes are balanced (roughly equal samples per class)
- When all types of errors are equally important
- As a quick first assessment
- NOT for imbalanced datasets

**Pros:**
- Very intuitive and easy to explain
- Good general measure for balanced datasets

**Cons:**
- Misleading on imbalanced datasets
- Doesn't distinguish between types of errors
- Can hide poor performance on minority class

```python
from sklearn.metrics import accuracy_score
import numpy as np

# Example 1: Balanced dataset (email spam detection)
y_true_balanced = np.array([1, 0, 1, 0, 1, 0, 1, 0])  # 50% spam, 50% not spam
y_pred_balanced = np.array([1, 0, 1, 0, 0, 0, 1, 1])

accuracy_balanced = accuracy_score(y_true_balanced, y_pred_balanced)
print(f"Balanced dataset accuracy: {accuracy_balanced:.2%}")  # 75%
print("✓ Accuracy is meaningful here\n")

# Example 2: Imbalanced dataset (fraud detection)
# 99% legitimate transactions, 1% fraudulent
y_true_imbalanced = np.array([0]*99 + [1]*1)  # 1 fraud out of 100
y_pred_dumb = np.array([0]*100)  # Dumb model: always predict "not fraud"

accuracy_imbalanced = accuracy_score(y_true_imbalanced, y_pred_dumb)
print(f"Imbalanced dataset - Dumb model accuracy: {accuracy_imbalanced:.2%}")
print("⚠ 99% accuracy but catches ZERO fraud cases!")
print("⚠ Accuracy is misleading here - need Precision/Recall\n")

# Manual calculation
tn, fp, fn, tp = 99, 0, 1, 0  # From dumb model above
accuracy_manual = (tp + tn) / (tp + tn + fp + fn)
print(f"Manual calculation: {accuracy_manual:.2%}")

# Better model for imbalanced data
y_pred_better = np.array([0]*97 + [1]*3)  # Catches some fraud, few false alarms
# Even with lower accuracy, this might be better if it catches the fraud
```

### Precision

**Mathematical Formula:**
```
Precision = TP / (TP + FP)
          = True Positives / Predicted Positives
```

**Intuition:**
Precision answers: "When the model predicts positive, how often is it correct?" It measures the quality of positive predictions. High precision means few false alarms. Think of it as "trustworthiness of positive predictions." A spam filter with high precision rarely marks legitimate emails as spam.

**When to Use:**
- When false positives are costly
- When you want to minimize false alarms
- Examples: spam detection (don't want to lose important emails), medical treatments (don't want unnecessary procedures), fraud alerts (avoid annoying legitimate customers)

**Pros:**
- Focuses on quality of positive predictions
- Important when false positives are costly
- Works well with imbalanced data

**Cons:**
- Ignores false negatives
- Can be high even if many positives are missed
- Not meaningful alone (need to consider with recall)

```python
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Example: Spam email filter
y_true = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0])  # 1=spam, 0=legitimate
y_pred = np.array([1, 1, 0, 0, 0, 0, 1, 1, 1, 0])

precision = precision_score(y_true, y_pred)
print(f"Precision: {precision:.2%}")

# Manual calculation
tp = np.sum((y_true == 1) & (y_pred == 1))  # 3
fp = np.sum((y_true == 0) & (y_pred == 1))  # 1
precision_manual = tp / (tp + fp)
print(f"Precision (manual): {precision_manual:.2%}")
print(f"Out of {tp + fp} emails marked as spam, {tp} were actually spam")
print(f"False alarms: {fp} legitimate emails marked as spam\n")

# Demonstrating precision vs recall tradeoff
print("Scenario 1: High precision, low recall")
y_pred_conservative = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # Very conservative
prec_1 = precision_score(y_true, y_pred_conservative)
rec_1 = recall_score(y_true, y_pred_conservative)
print(f"Precision: {prec_1:.2%}, Recall: {rec_1:.2%}")
print("Only flags obvious spam → few false alarms but misses many spam emails\n")

print("Scenario 2: Low precision, high recall")
y_pred_aggressive = np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 0])  # Very aggressive
prec_2 = precision_score(y_true, y_pred_aggressive)
rec_2 = recall_score(y_true, y_pred_aggressive)
print(f"Precision: {prec_2:.2%}, Recall: {rec_2:.2%}")
print("Flags almost everything as spam → catches most spam but many false alarms")
```

### Recall (Sensitivity, True Positive Rate)

**Mathematical Formula:**
```
Recall = TP / (TP + FN)
       = True Positives / Actual Positives
       = Sensitivity = TPR
```

**Intuition:**
Recall answers: "Out of all actual positive cases, how many did we find?" It measures the model's ability to find all positive instances. High recall means few missed detections. Think of it as "completeness of detection." A cancer screening test with high recall catches most cancer cases.

**When to Use:**
- When false negatives are costly
- When you need to catch all positive cases
- Examples: cancer screening (can't miss cancer cases), fraud detection (catch all fraud), search engines (show all relevant results)

**Pros:**
- Focuses on finding all positive cases
- Critical when missing positives is dangerous
- Works well with imbalanced data

**Cons:**
- Ignores false positives
- Can be high even with many false alarms
- Not meaningful alone (need to consider with precision)

```python
from sklearn.metrics import recall_score, precision_score
import numpy as np

# Example: Cancer screening
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 0])  # 1=cancer, 0=healthy
y_pred = np.array([1, 1, 1, 1, 0, 1, 1, 0, 0, 0])

recall = recall_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)

print(f"Recall: {recall:.2%}")
print(f"Precision: {precision:.2%}\n")

# Manual calculation
tp = np.sum((y_true == 1) & (y_pred == 1))  # 4 out of 5 cancer cases caught
fn = np.sum((y_true == 1) & (y_pred == 0))  # 1 cancer case missed
recall_manual = tp / (tp + fn)

print(f"Out of {tp + fn} actual cancer cases:")
print(f"  Detected: {tp} cases ({recall:.0%})")
print(f"  Missed: {fn} case ← This is dangerous!\n")

# Why recall matters in cancer screening
print("Cancer Screening Priority:")
print("❌ Missing a cancer case (FN) = Patient doesn't get treatment → Death")
print("✓ False alarm (FP) = Extra tests → Inconvenient but safe")
print("→ High recall is CRITICAL, even if precision is lower\n")

# Compare different decision thresholds
print("Conservative threshold (high confidence needed):")
y_pred_conservative = np.array([1, 0, 1, 1, 0, 0, 0, 0, 0, 0])
recall_cons = recall_score(y_true, y_pred_conservative)
precision_cons = precision_score(y_true, y_pred_conservative)
print(f"Recall: {recall_cons:.2%}, Precision: {precision_cons:.2%}")
print("→ Misses 40% of cancer cases!\n")

print("Aggressive threshold (low confidence needed):")
y_pred_aggressive = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
recall_agg = recall_score(y_true, y_pred_aggressive)
precision_agg = precision_score(y_true, y_pred_aggressive)
print(f"Recall: {recall_agg:.2%}, Precision: {precision_agg:.2%}")
print("→ Catches all cancer cases! (Lower precision acceptable)")
```

### F1-Score

**Mathematical Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = Harmonic Mean of Precision and Recall
```

**Intuition:**
F1-Score balances precision and recall into a single metric. It's the harmonic mean, which punishes extreme values - a model with 100% precision but 10% recall gets F1 = 18%, not 55%. This forces the model to be good at both finding positives (recall) and being accurate when it does (precision). F1 is maximized when precision equals recall.

**When to Use:**
- When you need balance between precision and recall
- When false positives and false negatives are equally important
- When you need a single metric for model selection
- Standard metric for imbalanced classification

**Pros:**
- Balances precision and recall
- Single metric for optimization
- Better than accuracy for imbalanced datasets
- Commonly used in competitions

**Cons:**
- Doesn't work well when precision and recall have different importance
- Weights precision and recall equally (might not match business needs)
- Less interpretable than precision or recall alone

```python
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

# Example: Information retrieval system
y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1])  # Relevant documents
y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0])

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-Score: {f1:.2%}\n")

# Manual calculation
f1_manual = 2 * (precision * recall) / (precision + recall)
print(f"F1-Score (manual): {f1_manual:.2%}\n")

# Why harmonic mean?
print("Why Harmonic Mean (not arithmetic mean)?")
arithmetic_mean = (precision + recall) / 2
print(f"Arithmetic mean: {arithmetic_mean:.2%}")
print(f"Harmonic mean (F1): {f1:.2%}")
print("Harmonic mean penalizes imbalance between precision and recall\n")

# Extreme example
print("Extreme imbalance example:")
precision_high, recall_low = 0.95, 0.10
arithmetic_extreme = (precision_high + recall_low) / 2
harmonic_extreme = 2 * (precision_high * recall_low) / (precision_high + recall_low)
print(f"Precision: {precision_high:.0%}, Recall: {recall_low:.0%}")
print(f"Arithmetic: {arithmetic_extreme:.1%} ← Misleadingly high!")
print(f"Harmonic (F1): {harmonic_extreme:.1%} ← Correctly low!")
print("F1 forces you to improve both metrics\n")

# F-beta score: weighted version of F1
from sklearn.metrics import fbeta_score

# F2-score: weights recall 2x more than precision
f2 = fbeta_score(y_true, y_pred, beta=2)
print(f"F2-Score (recall emphasis): {f2:.2%}")

# F0.5-score: weights precision 2x more than recall
f05 = fbeta_score(y_true, y_pred, beta=0.5)
print(f"F0.5-Score (precision emphasis): {f05:.2%}")

print(f"\nF1-Score (balanced): {f1:.2%}")
print("Use F-beta when one metric is more important than the other")
```

### Specificity (True Negative Rate)

**Mathematical Formula:**
```
Specificity = TN / (TN + FP)
            = True Negatives / Actual Negatives
            = TNR
```

**Intuition:**
Specificity answers: "Out of all actual negative cases, how many did we correctly identify?" It's the recall equivalent for the negative class. High specificity means the model is good at ruling out negatives and avoiding false alarms. In medical testing, high specificity means healthy people rarely get false positive diagnoses.

**When to Use:**
- When false positives are costly
- When confirming absence is important
- Examples: disease screening (don't want to scare healthy people), quality control (don't want to reject good products)

```python
from sklearn.metrics import confusion_matrix
import numpy as np

def specificity_score(y_true, y_pred):
    """Calculate specificity (True Negative Rate)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# Example: Drug test screening
y_true = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0])  # 1=drug user, 0=clean
y_pred = np.array([0, 0, 1, 0, 0, 0, 0, 1, 1, 0])

specificity = specificity_score(y_true, y_pred)
from sklearn.metrics import recall_score, precision_score

sensitivity = recall_score(y_true, y_pred)  # Same as recall for positive class

print(f"Sensitivity (catches drug users): {sensitivity:.2%}")
print(f"Specificity (doesn't flag clean people): {specificity:.2%}\n")

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(f"Out of {tn + fp} clean people:")
print(f"  Correctly cleared: {tn} ({specificity:.0%})")
print(f"  False positives: {fp} (wrongly flagged as drug users)")
print("\n✓ High specificity protects innocent people from false accusations")
```

### ROC Curve and AUC-ROC

**Mathematical Formula:**
```
ROC Curve: Plot of TPR vs FPR at different thresholds
TPR (True Positive Rate) = Recall = TP / (TP + FN)
FPR (False Positive Rate) = FP / (FP + TN)

AUC-ROC = Area Under the ROC Curve
```

**Intuition:**
Most classifiers output probabilities, then use a threshold (e.g., 0.5) to decide the class. The ROC curve shows classifier performance across ALL possible thresholds. The y-axis (TPR/Recall) shows what fraction of positives we catch, and the x-axis (FPR) shows what fraction of negatives we misclassify.

AUC-ROC is the area under this curve:
- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random guessing (diagonal line)
- AUC < 0.5: Worse than random (predictions are inverted)

Think of AUC as: "If I pick a random positive and a random negative, what's the probability the classifier scores the positive higher?"

**When to Use:**
- When you want threshold-independent evaluation
- When comparing models across different operating points
- When class distribution might change in production
- Standard metric for binary classification

**Pros:**
- Threshold-independent
- Good for imbalanced datasets
- Single number summary
- Shows full picture of classifier performance

**Cons:**
- Not informative about performance at specific threshold
- Can be optimistic on highly imbalanced datasets
- Doesn't capture costs of different errors

```python
from sklearn.metrics import roc_curve, roc_auc_score, auc
import numpy as np
import matplotlib.pyplot as plt

# Example: Credit card fraud detection (imbalanced)
np.random.seed(42)
# True labels: 90 legitimate (0), 10 fraudulent (1)
y_true = np.array([0]*90 + [1]*10)

# Predicted probabilities (not hard predictions)
# Good model: higher probabilities for fraud cases
y_proba = np.concatenate([
    np.random.beta(2, 5, 90),   # Legitimate: mostly low scores
    np.random.beta(5, 2, 10)    # Fraudulent: mostly high scores
])

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_proba)
roc_auc = roc_auc_score(y_true, y_proba)

print(f"AUC-ROC: {roc_auc:.3f}\n")

# Interpret different threshold choices
print("Different threshold choices:")
for threshold in [0.3, 0.5, 0.7]:
    y_pred = (y_proba >= threshold).astype(int)

    # Find closest threshold in ROC curve
    idx = np.argmin(np.abs(thresholds - threshold))
    print(f"\nThreshold = {threshold}")
    print(f"  TPR (Sensitivity): {tpr[idx]:.2%} - catches {tpr[idx]:.0%} of fraud")
    print(f"  FPR: {fpr[idx]:.2%} - {fpr[idx]:.0%} of legitimate flagged as fraud")

# Visualize ROC curve
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
# plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier (AUC = 0.5)')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate (Recall)')
# plt.title('ROC Curve - Fraud Detection')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

print("\n" + "="*50)
print("Comparing two models:")

# Model 1: Better model
y_proba_good = y_proba
auc_good = roc_auc_score(y_true, y_proba_good)

# Model 2: Worse model (more random)
y_proba_bad = np.concatenate([
    np.random.beta(3, 3, 90),   # More random
    np.random.beta(3, 3, 10)
])
auc_bad = roc_auc_score(y_true, y_proba_bad)

print(f"Good model AUC: {auc_good:.3f}")
print(f"Bad model AUC: {auc_bad:.3f}")
print(f"\nHigher AUC = Better discrimination between classes")
```

### Precision-Recall Curve and AUC-PR

**Mathematical Formula:**
```
PR Curve: Plot of Precision vs Recall at different thresholds
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)

AUC-PR = Area Under the Precision-Recall Curve
```

**Intuition:**
While ROC curves show the tradeoff between true positives and false positives, PR curves show the tradeoff between precision (quality of positive predictions) and recall (completeness of positive detection). PR curves are especially useful for imbalanced datasets because they focus on the positive class.

Unlike ROC-AUC where 0.5 is random performance, PR-AUC baseline equals the fraction of positives in the dataset. For a dataset with 1% positives, random classifier has AUC-PR ≈ 0.01, not 0.5.

**When to Use:**
- When you have highly imbalanced datasets
- When the positive class is more important
- When false positives matter more than true negatives
- Examples: rare disease detection, fraud detection, information retrieval

**Pros:**
- More informative than ROC for imbalanced data
- Focuses on positive class performance
- Shows precision-recall tradeoff clearly

**Cons:**
- Less intuitive than ROC curves
- Baseline performance varies with class distribution
- Harder to interpret for non-experts

```python
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
from sklearn.metrics import roc_auc_score
import numpy as np

# Example: Rare disease detection (highly imbalanced)
np.random.seed(42)
# 99% healthy (0), 1% diseased (1)
y_true = np.array([0]*990 + [1]*10)

# Model predictions (probabilities)
y_proba = np.concatenate([
    np.random.beta(2, 8, 990),  # Healthy: low scores
    np.random.beta(6, 2, 10)     # Diseased: high scores
])

# Calculate PR curve
precision, recall, thresholds_pr = precision_recall_curve(y_true, y_proba)
auc_pr = average_precision_score(y_true, y_proba)

# Calculate ROC curve for comparison
from sklearn.metrics import roc_curve
fpr, tpr, thresholds_roc = roc_curve(y_true, y_proba)
auc_roc = roc_auc_score(y_true, y_proba)

print("Highly Imbalanced Dataset (1% positive class):")
print(f"AUC-ROC: {auc_roc:.3f} ← Looks good!")
print(f"AUC-PR:  {auc_pr:.3f} ← More honest assessment")
print(f"Baseline PR-AUC: {np.mean(y_true):.3f} (fraction of positives)")
print("\n✓ PR curve better shows performance on rare positive class\n")

# Visualize the tradeoff
print("Precision-Recall Tradeoff:")
print("Threshold | Precision | Recall | Interpretation")
print("-" * 60)

for i, threshold in enumerate([0.2, 0.5, 0.8]):
    y_pred = (y_proba >= threshold).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"{threshold:.1f}      | {prec:.2%}    | {rec:.2%} | ", end="")

    if threshold == 0.2:
        print("High recall, low precision (many false alarms)")
    elif threshold == 0.5:
        print("Balanced")
    else:
        print("Low recall, high precision (miss many cases)")

# When to use ROC vs PR
print("\n" + "="*60)
print("When to use ROC-AUC vs PR-AUC:")
print("-" * 60)
print("Use ROC-AUC when:")
print("  • Classes are balanced")
print("  • Both classes equally important")
print("  • You care about true negative rate")
print("\nUse PR-AUC when:")
print("  • Classes are highly imbalanced")
print("  • Positive class is more important")
print("  • You care more about precision and recall")
print("  • Example: 1% fraud detection, 0.1% disease screening")
```

### Log Loss (Binary Cross-Entropy)

**Mathematical Formula:**
```
Log Loss = -(1/n) × Σ[y_i × log(ŷ_i) + (1 - y_i) × log(1 - ŷ_i)]

where:
y_i ∈ {0, 1} is the true label
ŷ_i ∈ [0, 1] is the predicted probability
```

**Intuition:**
Log Loss measures how well predicted probabilities match true labels. Unlike accuracy which only cares about the final decision (>0.5), Log Loss cares about confidence:
- Predicting 0.9 for a positive case is better than predicting 0.6
- Predicting 0.51 for a negative case is very bad (confident but wrong)
- Being confidently wrong is heavily penalized

Lower log loss is better. Perfect predictions (ŷ=1 for y=1, ŷ=0 for y=0) give log loss = 0.

**When to Use:**
- When probability calibration matters
- When training probabilistic classifiers
- When different confidence levels matter
- Loss function for logistic regression and neural networks

**Pros:**
- Considers prediction confidence
- Penalizes confidently wrong predictions heavily
- Differentiable (good for optimization)
- Proper scoring rule (encourages honest probability estimates)

**Cons:**
- Heavily penalizes confident mistakes (can be sensitive to outliers)
- Less interpretable than accuracy
- Unbounded (can be infinitely large)

```python
from sklearn.metrics import log_loss
import numpy as np

# Example: Medical diagnosis probabilities
y_true = np.array([1, 1, 0, 0, 1])  # True diagnoses

# Model 1: Well-calibrated probabilities
y_proba_good = np.array([0.9, 0.85, 0.1, 0.15, 0.8])
logloss_good = log_loss(y_true, y_proba_good)

# Model 2: Right predictions but less confident
y_proba_less_confident = np.array([0.6, 0.65, 0.4, 0.45, 0.55])
logloss_less_conf = log_loss(y_true, y_proba_less_confident)

# Model 3: Confidently wrong on one prediction
y_proba_bad = np.array([0.9, 0.85, 0.95, 0.15, 0.8])  # 0.95 for a negative case!
logloss_bad = log_loss(y_true, y_proba_bad)

print("Comparing Log Loss for different models:")
print(f"Good model (confident and correct): {logloss_good:.3f}")
print(f"Less confident model: {logloss_less_conf:.3f}")
print(f"Confidently wrong model: {logloss_bad:.3f}")
print("\n✓ Being confidently wrong is heavily penalized!\n")

# Manual calculation for understanding
def manual_log_loss(y_true, y_proba):
    epsilon = 1e-15  # Avoid log(0)
    y_proba = np.clip(y_proba, epsilon, 1 - epsilon)

    loss = 0
    for y, p in zip(y_true, y_proba):
        if y == 1:
            loss -= np.log(p)
        else:
            loss -= np.log(1 - p)

    return loss / len(y_true)

logloss_manual = manual_log_loss(y_true, y_proba_good)
print(f"Manual calculation: {logloss_manual:.3f}\n")

# Demonstrating the penalty for confident mistakes
print("Penalty for different wrong predictions:")
print("True label = 0 (negative)")
print("Predicted | Log Loss Contribution")
print("-" * 35)
for prob in [0.51, 0.7, 0.9, 0.99]:
    contribution = -np.log(1 - prob)
    print(f"  {prob:.2f}    | {contribution:.3f}")

print("\nConfident mistakes are exponentially more costly!")

# Binary predictions vs probabilities
y_pred_binary = (y_proba_good > 0.5).astype(int)
accuracy = np.mean(y_pred_binary == y_true)

print(f"\nAccuracy: {accuracy:.0%}")
print(f"Log Loss: {logloss_good:.3f}")
print("\nAccuracy ignores confidence, Log Loss uses it for better evaluation")
```

### Cohen's Kappa

**Mathematical Formula:**
```
κ = (p_o - p_e) / (1 - p_e)

where:
p_o = observed agreement (accuracy)
p_e = expected agreement by chance
```

**Intuition:**
Cohen's Kappa adjusts accuracy for agreement that would occur by chance. If two random classifiers agree 70% of the time just by luck, and your model agrees 85% of the time, Kappa measures the improvement over chance. κ = 1 means perfect agreement, κ = 0 means agreement is no better than chance, κ < 0 means worse than chance.

**Interpretation:**
- κ < 0: Worse than chance
- κ = 0: No better than chance
- 0.01-0.20: Slight agreement
- 0.21-0.40: Fair agreement
- 0.41-0.60: Moderate agreement
- 0.61-0.80: Substantial agreement
- 0.81-1.00: Almost perfect agreement

**When to Use:**
- When you want to account for chance agreement
- When classes are imbalanced
- When comparing inter-rater reliability
- In medical diagnosis, psychology, and social sciences

**Pros:**
- Accounts for chance agreement
- Works well with imbalanced classes
- More robust than accuracy

**Cons:**
- Less interpretable than accuracy
- Depends on class distribution
- Can be paradoxical in some edge cases

```python
from sklearn.metrics import cohen_kappa_score, accuracy_score
import numpy as np

# Example 1: Balanced classes
y_true_balanced = np.array([0, 1, 0, 1, 0, 1, 0, 1])
y_pred_balanced = np.array([0, 1, 0, 0, 0, 1, 1, 1])

accuracy_bal = accuracy_score(y_true_balanced, y_pred_balanced)
kappa_bal = cohen_kappa_score(y_true_balanced, y_pred_balanced)

print("Balanced dataset:")
print(f"Accuracy: {accuracy_bal:.2%}")
print(f"Kappa: {kappa_bal:.3f}\n")

# Example 2: Imbalanced classes (this is where Kappa shines)
# 90% class 0, 10% class 1
y_true_imb = np.array([0]*90 + [1]*10)

# Dumb classifier: always predict majority class
y_pred_dumb = np.array([0]*100)

# Slightly better classifier
np.random.seed(42)
y_pred_better = y_true_imb.copy()
# Add some noise (20% random errors)
noise_idx = np.random.choice(100, 20, replace=False)
y_pred_better[noise_idx] = 1 - y_pred_better[noise_idx]

print("Imbalanced dataset (90% class 0, 10% class 1):")
print("\nDumb classifier (always predict 0):")
acc_dumb = accuracy_score(y_true_imb, y_pred_dumb)
kappa_dumb = cohen_kappa_score(y_true_imb, y_pred_dumb)
print(f"Accuracy: {acc_dumb:.2%} ← Looks good!")
print(f"Kappa: {kappa_dumb:.3f} ← Shows it's useless!")

print("\nBetter classifier:")
acc_better = accuracy_score(y_true_imb, y_pred_better)
kappa_better = cohen_kappa_score(y_true_imb, y_pred_better)
print(f"Accuracy: {acc_better:.2%}")
print(f"Kappa: {kappa_better:.3f} ← Much better than chance!\n")

# Manual calculation for understanding
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true_imb, y_pred_better)
print("Manual Kappa calculation:")
print(f"Confusion Matrix:\n{cm}\n")

# Observed agreement
p_o = np.sum(np.diag(cm)) / np.sum(cm)

# Expected agreement
marginal_true = np.sum(cm, axis=1) / np.sum(cm)
marginal_pred = np.sum(cm, axis=0) / np.sum(cm)
p_e = np.sum(marginal_true * marginal_pred)

kappa_manual = (p_o - p_e) / (1 - p_e)

print(f"Observed agreement (p_o): {p_o:.3f}")
print(f"Expected agreement (p_e): {p_e:.3f}")
print(f"Kappa: {kappa_manual:.3f}")
print("\nKappa removes the 'luck' component from accuracy")
```

### Matthews Correlation Coefficient (MCC)

**Mathematical Formula:**
```
MCC = (TP×TN - FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]
```

**Intuition:**
MCC is a correlation coefficient between predicted and actual classifications. It ranges from -1 (total disagreement) to +1 (perfect prediction), with 0 being random guessing. Unlike F1-score, MCC uses all four confusion matrix values equally, making it more balanced and informative for imbalanced datasets.

MCC is considered one of the best single metrics because it's symmetric (treats both classes fairly) and works well regardless of class imbalance.

**When to Use:**
- When you want a single, balanced metric
- When both classes are important
- When dealing with imbalanced datasets
- When you need a correlation-based measure

**Pros:**
- Balanced (uses all confusion matrix values)
- Works well with imbalanced datasets
- More informative than accuracy or F1
- Symmetric between positive and negative classes

**Cons:**
- Less intuitive than precision/recall
- Range of -1 to 1 can be confusing
- Less widely used (though gaining popularity)

```python
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
import numpy as np

# Example: Imbalanced binary classification
y_true = np.array([0]*95 + [1]*5)  # 95% negative, 5% positive

# Scenario 1: Dumb classifier (always predict majority)
y_pred_dumb = np.array([0]*100)

# Scenario 2: Decent classifier
y_pred_decent = y_true.copy()
y_pred_decent[0] = 1   # 1 FP
y_pred_decent[95] = 0  # 1 FN

# Scenario 3: Good classifier
y_pred_good = y_true.copy()
y_pred_good[0] = 1     # 1 FP

print("Comparing metrics on imbalanced data (95% negative, 5% positive):")
print("\n" + "="*70)

for name, y_pred in [("Dumb (always 0)", y_pred_dumb),
                      ("Decent", y_pred_decent),
                      ("Good", y_pred_good)]:

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  F1-Score: {f1:.3f}")
    print(f"  MCC: {mcc:.3f}")

print("\n" + "="*70)
print("✓ MCC gives most honest assessment across all scenarios")
print("✓ MCC = 0 for useless classifier, higher for better ones")

# Manual calculation
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred_good)
tn, fp, fn, tp = cm.ravel()

print(f"\nManual MCC calculation for 'Good' classifier:")
print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")

numerator = (tp * tn) - (fp * fn)
denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
mcc_manual = numerator / denominator if denominator != 0 else 0

print(f"MCC = {mcc_manual:.3f}")

# MCC interpretation
print("\nMCC Interpretation:")
print("  +1.0: Perfect prediction")
print("  +0.7 to +1.0: Strong positive correlation")
print("  +0.3 to +0.7: Moderate positive correlation")
print("   0.0: Random guessing")
print("  -1.0: Total disagreement (inverted predictions)")
```

---

## Multi-class Classification Strategies

When dealing with multi-class classification (more than 2 classes), metrics like precision, recall, and F1 need to be adapted. There are several strategies for aggregating metrics across classes.

### Micro, Macro, and Weighted Averaging

**Formulas:**
```
Micro-averaging:
  Calculate metrics globally by counting total TP, FP, FN
  Precision_micro = ΣTP / (ΣTP + ΣFP)

Macro-averaging:
  Calculate metrics for each class, then average
  Precision_macro = (Precision_class1 + ... + Precision_classN) / N

Weighted-averaging:
  Calculate metrics for each class, then weighted average by support
  Precision_weighted = Σ(Precision_classi × n_i) / Σn_i
```

**Intuition:**
- **Micro**: Treats every instance equally, regardless of class. Dominated by performance on large classes.
- **Macro**: Treats every class equally, regardless of size. Gives equal weight to rare classes.
- **Weighted**: Weighted average by class frequency. Between micro and macro.

**When to Use:**
- **Micro**: When you care about overall performance and large classes are more important
- **Macro**: When all classes are equally important, regardless of frequency
- **Weighted**: When you want to account for class imbalance but still emphasize larger classes

```python
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Example: Customer support ticket classification
# Classes: 0=Technical, 1=Billing, 2=General
# Imbalanced: 60% Technical, 30% Billing, 10% General

np.random.seed(42)
n_samples = 100

# True labels (imbalanced)
y_true = np.array([0]*60 + [1]*30 + [2]*10)

# Predictions (model performs worse on rare class 2)
y_pred = y_true.copy()
# Add errors: 10% errors on class 0, 15% on class 1, 40% on class 2
for class_label, error_rate in [(0, 0.10), (1, 0.15), (2, 0.40)]:
    idx = np.where(y_true == class_label)[0]
    n_errors = int(len(idx) * error_rate)
    error_idx = np.random.choice(idx, n_errors, replace=False)
    # Randomly assign to other classes
    y_pred[error_idx] = np.random.choice([c for c in range(3) if c != class_label],
                                         n_errors)

# Calculate metrics with different averaging strategies
print("Multi-class Metrics Comparison:")
print("="*70)

for avg_type in ['micro', 'macro', 'weighted']:
    precision = precision_score(y_true, y_pred, average=avg_type)
    recall = recall_score(y_true, y_pred, average=avg_type)
    f1 = f1_score(y_true, y_pred, average=avg_type)

    print(f"\n{avg_type.upper()}-averaging:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-Score: {f1:.3f}")

# Per-class breakdown
print("\n" + "="*70)
print("\nPer-class Performance:")
print(classification_report(y_true, y_pred,
                          target_names=['Technical', 'Billing', 'General']))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Explanation
print("\n" + "="*70)
print("Understanding the differences:")
print("-"*70)
print("MICRO: Dominated by performance on 'Technical' (largest class)")
print("       Good when you care most about overall accuracy")
print()
print("MACRO: Equal weight to all classes, including rare 'General'")
print("       Shows that model struggles with rare class")
print("       Good when all classes equally important")
print()
print("WEIGHTED: Between micro and macro, accounts for class sizes")
print("          Good general-purpose metric for imbalanced data")
```

### One-vs-Rest (OvR) and One-vs-One (OvO)

**Intuition:**

**One-vs-Rest (OvR):**
For N classes, train N binary classifiers. Each classifier learns to distinguish one class from all others. At prediction time, choose the class with highest confidence.

**One-vs-One (OvO):**
For N classes, train N×(N-1)/2 binary classifiers. Each classifier learns to distinguish between two classes. At prediction time, use voting.

```python
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Generate multi-class dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, n_classes=4, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

# One-vs-Rest
ovr = OneVsRestClassifier(LogisticRegression(max_iter=1000))
ovr.fit(X_train, y_train)
y_pred_ovr = ovr.predict(X_test)

# One-vs-One
ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
ovo.fit(X_train, y_train)
y_pred_ovo = ovo.predict(X_test)

# Standard multi-class (most algorithms do this internally)
multi = LogisticRegression(max_iter=1000, multi_class='multinomial')
multi.fit(X_train, y_train)
y_pred_multi = multi.predict(X_test)

print("Comparing Multi-class Strategies:")
print("="*70)
print(f"\nOne-vs-Rest (OvR):")
print(f"  Number of classifiers: {len(ovr.estimators_)}")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_ovr):.3f}")

print(f"\nOne-vs-One (OvO):")
print(f"  Number of classifiers: {len(ovo.estimators_)}")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_ovo):.3f}")

print(f"\nMultinomial (native multi-class):")
print(f"  Number of classifiers: 1")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_multi):.3f}")

print("\n" + "="*70)
print("When to use each:")
print("-"*70)
print("One-vs-Rest (OvR):")
print("  • Most efficient (N classifiers for N classes)")
print("  • Good when classes are well-separated")
print("  • Default for many algorithms (SVM, Logistic Regression)")
print()
print("One-vs-One (OvO):")
print("  • More classifiers but each is simpler (N×(N-1)/2)")
print("  • Better when classes overlap")
print("  • More robust to class imbalance")
print("  • Used in SVM implementations")
print()
print("Native Multi-class:")
print("  • Best when algorithm supports it (Neural Networks, Trees)")
print("  • Most efficient and often most accurate")
```

---

## Ranking and Recommendation Metrics

Ranking metrics evaluate how well a model orders items, crucial for search engines, recommendation systems, and information retrieval.

### Precision@K and Recall@K

**Mathematical Formula:**
```
Precision@K = (Relevant items in top K) / K
Recall@K = (Relevant items in top K) / (Total relevant items)
```

**Intuition:**
These metrics evaluate the top K recommendations:
- **Precision@K**: "Of the K items I recommended, how many were relevant?"
- **Recall@K**: "Of all relevant items, how many are in my top K?"

For example, if you recommend 10 movies and 7 are good (Precision@10 = 70%), and there are 20 good movies total (Recall@10 = 35%), these metrics capture both quality and coverage.

**When to Use:**
- Search engines (top 10 results)
- Recommendation systems (top N products)
- Information retrieval
- When position matters but exact order doesn't

```python
import numpy as np

def precision_at_k(y_true, y_pred, k):
    """
    Calculate Precision@K.

    Args:
        y_true: Binary relevance (1=relevant, 0=not relevant)
        y_pred: Predicted scores (higher = more relevant)
        k: Number of top items to consider

    Returns:
        Precision@K score
    """
    # Get indices of top k predictions
    top_k_idx = np.argsort(y_pred)[::-1][:k]

    # Count how many are actually relevant
    relevant_in_top_k = np.sum(y_true[top_k_idx])

    return relevant_in_top_k / k

def recall_at_k(y_true, y_pred, k):
    """Calculate Recall@K."""
    top_k_idx = np.argsort(y_pred)[::-1][:k]
    relevant_in_top_k = np.sum(y_true[top_k_idx])
    total_relevant = np.sum(y_true)

    return relevant_in_top_k / total_relevant if total_relevant > 0 else 0

# Example: Movie recommendation system
# 20 movies, user likes 6 of them
y_true = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1,
                   0, 0, 1, 0, 0, 0, 0, 1, 0, 0])  # 6 relevant movies

# Model's predicted scores (higher = better)
y_pred = np.array([0.9, 0.1, 0.8, 0.3, 0.2, 0.95, 0.4, 0.15, 0.25, 0.7,
                   0.5, 0.35, 0.85, 0.2, 0.1, 0.3, 0.2, 0.75, 0.15, 0.05])

print("Movie Recommendation Evaluation:")
print("="*70)
print(f"Total movies: {len(y_true)}")
print(f"Relevant movies: {np.sum(y_true)}")
print()

# Evaluate at different K values
for k in [5, 10, 15]:
    prec_k = precision_at_k(y_true, y_pred, k)
    rec_k = recall_at_k(y_true, y_pred, k)

    print(f"K = {k}:")
    print(f"  Precision@{k}: {prec_k:.2%} - {int(prec_k*k)}/{k} recommended movies are relevant")
    print(f"  Recall@{k}: {rec_k:.2%} - Found {int(rec_k*np.sum(y_true))}/{int(np.sum(y_true))} relevant movies")
    print()

# Show actual top-5 recommendations
top_5_idx = np.argsort(y_pred)[::-1][:5]
print("Top 5 Recommendations:")
for i, idx in enumerate(top_5_idx, 1):
    relevant = "✓ RELEVANT" if y_true[idx] == 1 else "✗ not relevant"
    print(f"  {i}. Movie {idx} (score: {y_pred[idx]:.2f}) - {relevant}")
```

### Mean Average Precision (MAP)

**Mathematical Formula:**
```
AP@N = (1/min(m,N)) × Σ(Precision@k × rel(k))
where:
  - m = number of relevant items
  - N = number of items considered
  - rel(k) = relevance of item at position k (1 if relevant, 0 otherwise)
  - Sum is over all positions k from 1 to N

MAP = mean of AP across all queries
```

**Intuition:**
MAP considers both the relevance AND the ranking of results. It rewards models that put relevant items higher in the ranked list. Unlike Precision@K which treats all positions equally, MAP gives more weight to relevant items that appear earlier.

Think of it as: "On average, how good is the precision at each relevant item's position?"

**When to Use:**
- Information retrieval systems
- Search engines
- When ranking order is important
- When you have multiple queries to evaluate

**Pros:**
- Considers ranking order (earlier = better)
- Single metric summarizing ranking quality
- Standard in information retrieval

**Cons:**
- Can be dominated by easy queries
- Doesn't consider user behavior (top results matter most)
- Assumes binary relevance (relevant/not)

```python
import numpy as np

def average_precision(y_true, y_pred, k=None):
    """
    Calculate Average Precision.

    Args:
        y_true: Binary relevance (1=relevant, 0=not)
        y_pred: Predicted scores
        k: Consider only top k items (None = all)

    Returns:
        Average Precision score
    """
    # Sort by predicted scores (descending)
    sorted_indices = np.argsort(y_pred)[::-1]

    if k is not None:
        sorted_indices = sorted_indices[:k]

    y_true_sorted = y_true[sorted_indices]

    # Calculate precision at each position where there's a relevant item
    precisions = []
    num_relevant = 0

    for i, relevant in enumerate(y_true_sorted, 1):
        if relevant:
            num_relevant += 1
            precision_at_i = num_relevant / i
            precisions.append(precision_at_i)

    if not precisions:
        return 0.0

    return np.mean(precisions)

def mean_average_precision(y_true_list, y_pred_list, k=None):
    """Calculate MAP across multiple queries."""
    aps = [average_precision(yt, yp, k) for yt, yp in zip(y_true_list, y_pred_list)]
    return np.mean(aps)

# Example: Search engine with 3 different queries
print("Search Engine Evaluation:")
print("="*70)

# Query 1: "python tutorial"
y_true_q1 = np.array([1, 0, 1, 1, 0, 0, 1, 0])  # 4 relevant docs
y_pred_q1 = np.array([0.9, 0.2, 0.8, 0.85, 0.3, 0.1, 0.75, 0.4])

# Query 2: "machine learning"
y_true_q2 = np.array([1, 1, 0, 1, 0, 0, 0, 1])  # 4 relevant docs
y_pred_q2 = np.array([0.95, 0.9, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3])

# Query 3: "data science"
y_true_q3 = np.array([0, 1, 0, 0, 1, 1, 0, 0])  # 3 relevant docs
y_pred_q3 = np.array([0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35])

queries = [
    ("python tutorial", y_true_q1, y_pred_q1),
    ("machine learning", y_true_q2, y_pred_q2),
    ("data science", y_true_q3, y_pred_q3)
]

aps = []
for query_name, y_true, y_pred in queries:
    ap = average_precision(y_true, y_pred)
    aps.append(ap)

    print(f"\nQuery: '{query_name}'")
    print(f"  Relevant documents: {np.sum(y_true)}")
    print(f"  Average Precision: {ap:.3f}")

    # Show ranking
    sorted_idx = np.argsort(y_pred)[::-1]
    print("  Top 5 results: ", end="")
    for i, idx in enumerate(sorted_idx[:5], 1):
        print(f"{'✓' if y_true[idx] else '✗'}", end=" ")
    print()

map_score = np.mean(aps)
print(f"\n{'='*70}")
print(f"Mean Average Precision (MAP): {map_score:.3f}")

# Demonstrate impact of ranking order
print(f"\n{'='*70}")
print("Importance of Ranking Order:")
print("-"*70)

# Same relevant items, different order
y_true_same = np.array([1, 1, 0, 0, 0])
y_pred_good = np.array([0.9, 0.8, 0.3, 0.2, 0.1])  # Relevant items first
y_pred_bad = np.array([0.3, 0.2, 0.9, 0.8, 0.1])   # Relevant items last

ap_good = average_precision(y_true_same, y_pred_good)
ap_bad = average_precision(y_true_same, y_pred_bad)

print(f"Relevant items ranked high: AP = {ap_good:.3f}")
print(f"Relevant items ranked low:  AP = {ap_bad:.3f}")
print("✓ MAP rewards putting relevant items at the top!")
```

### Normalized Discounted Cumulative Gain (NDCG)

**Mathematical Formula:**
```
DCG@K = Σ(rel_i / log₂(i + 1)) for i=1 to K
where rel_i is the relevance score at position i

IDCG@K = DCG of ideal ranking (best possible ordering)

NDCG@K = DCG@K / IDCG@K
```

**Intuition:**
NDCG evaluates ranking quality with graded relevance (not just binary). Key insights:
1. More relevant items = higher scores (rel_i)
2. Higher positions = more weight (division by log₂(i+1) means position 1 >> position 10)
3. Normalized by ideal ranking (NDCG ∈ [0, 1])

The logarithmic discount means moving an item from position 10 to 9 matters less than moving from position 2 to 1.

**When to Use:**
- When relevance has multiple levels (highly relevant, somewhat relevant, not relevant)
- Search engines with graded relevance judgments
- Recommendation systems with ratings
- When top positions are much more important than lower positions

**Pros:**
- Handles graded relevance (not just binary)
- Position-aware (top results matter most)
- Normalized (comparable across queries)
- Industry standard for ranking evaluation

**Cons:**
- Requires graded relevance scores (more effort to label)
- Less interpretable than Precision@K
- Logarithmic discount may not match user behavior

```python
import numpy as np

def dcg_at_k(relevance_scores, k):
    """
    Calculate Discounted Cumulative Gain at K.

    Args:
        relevance_scores: Array of relevance scores in ranked order
        k: Number of top items to consider

    Returns:
        DCG@K score
    """
    relevance_scores = np.array(relevance_scores)[:k]

    if relevance_scores.size == 0:
        return 0.0

    # Positions start at 1
    discounts = np.log2(np.arange(2, relevance_scores.size + 2))

    return np.sum(relevance_scores / discounts)

def ndcg_at_k(y_true, y_pred, k):
    """
    Calculate Normalized Discounted Cumulative Gain at K.

    Args:
        y_true: True relevance scores
        y_pred: Predicted scores (for ranking)
        k: Number of top items to consider

    Returns:
        NDCG@K score
    """
    # Sort by predictions
    sorted_indices = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[sorted_indices]

    # DCG with actual ranking
    dcg = dcg_at_k(y_true_sorted, k)

    # IDCG with ideal ranking (sorted by true relevance)
    y_true_ideal = np.sort(y_true)[::-1]
    idcg = dcg_at_k(y_true_ideal, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg

# Example: Movie recommendation with graded relevance
# Relevance scale: 0=not relevant, 1=somewhat relevant, 2=relevant, 3=highly relevant

print("Movie Recommendation with Graded Relevance:")
print("="*70)

# True relevance scores for 10 movies
y_true = np.array([3, 0, 2, 1, 0, 3, 1, 0, 2, 0])

# Model's predicted scores
y_pred = np.array([0.95, 0.1, 0.85, 0.6, 0.2, 0.9, 0.5, 0.15, 0.75, 0.3])

# Calculate NDCG at different K values
for k in [3, 5, 10]:
    ndcg = ndcg_at_k(y_true, y_pred, k)
    print(f"\nNDCG@{k}: {ndcg:.3f}")

    # Show ranking
    sorted_idx = np.argsort(y_pred)[::-1][:k]
    print(f"Top {k} recommendations:")
    for i, idx in enumerate(sorted_idx, 1):
        relevance_names = ["Not relevant", "Somewhat", "Relevant", "Highly relevant"]
        rel_name = relevance_names[y_true[idx]]
        print(f"  {i}. Movie {idx}: score={y_pred[idx]:.2f}, " +
              f"relevance={y_true[idx]} ({rel_name})")

# Compare with ideal ranking
print("\n" + "="*70)
print("Ideal Ranking (sorted by true relevance):")
ideal_idx = np.argsort(y_true)[::-1][:5]
for i, idx in enumerate(ideal_idx, 1):
    print(f"  {i}. Movie {idx}: relevance={y_true[idx]}")

# Demonstrate position importance
print("\n" + "="*70)
print("Impact of Position:")
print("-"*70)

# Scenario 1: Highly relevant item at position 1
ranking_1 = np.array([3, 2, 1, 0, 0])  # Best item first
dcg_1 = dcg_at_k(ranking_1, 5)

# Scenario 2: Highly relevant item at position 3
ranking_2 = np.array([1, 0, 3, 2, 0])  # Best item third
dcg_2 = dcg_at_k(ranking_2, 5)

print(f"Highly relevant (3) at position 1: DCG = {dcg_1:.3f}")
print(f"Highly relevant (3) at position 3: DCG = {dcg_2:.3f}")
print(f"Difference: {dcg_1 - dcg_2:.3f}")
print("✓ Position 1 is worth much more than position 3!")

# Using sklearn's NDCG (available in newer versions)
try:
    from sklearn.metrics import ndcg_score

    # sklearn expects (n_samples, n_items) shape
    y_true_sk = y_true.reshape(1, -1)
    y_pred_sk = y_pred.reshape(1, -1)

    ndcg_sklearn = ndcg_score(y_true_sk, y_pred_sk, k=5)
    print(f"\nNDCG@5 (sklearn): {ndcg_sklearn:.3f}")
except ImportError:
    print("\n(sklearn.metrics.ndcg_score requires sklearn >= 0.24)")
```

### Mean Reciprocal Rank (MRR)

**Mathematical Formula:**
```
RR = 1 / rank of first relevant item

MRR = (1/|Q|) × Σ(1 / rank_i)
where |Q| is number of queries
```

**Intuition:**
MRR measures how quickly users find a relevant result. It only cares about the first relevant item:
- First result is relevant: RR = 1/1 = 1.0
- Second result is relevant: RR = 1/2 = 0.5
- Third result is relevant: RR = 1/3 = 0.33
- No relevant results: RR = 0

MRR is the average RR across multiple queries. It's commonly used in question answering and search where users typically click the first good result.

**When to Use:**
- Question answering systems (users want THE answer)
- Search where users usually click first relevant result
- When finding one relevant item is sufficient
- Conversational AI and chatbots

**Pros:**
- Simple and intuitive
- Focuses on user experience (first relevant result)
- Easy to compute
- Good for QA systems

**Cons:**
- Ignores all relevant items after the first
- Doesn't measure overall ranking quality
- Not useful when users need multiple results

```python
import numpy as np

def reciprocal_rank(y_true, y_pred):
    """
    Calculate Reciprocal Rank for a single query.

    Args:
        y_true: Binary relevance (1=relevant, 0=not)
        y_pred: Predicted scores

    Returns:
        Reciprocal rank (1/rank of first relevant item)
    """
    # Sort by predictions
    sorted_indices = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[sorted_indices]

    # Find position of first relevant item (positions start at 1)
    for rank, relevant in enumerate(y_true_sorted, 1):
        if relevant:
            return 1.0 / rank

    # No relevant items found
    return 0.0

def mean_reciprocal_rank(y_true_list, y_pred_list):
    """Calculate MRR across multiple queries."""
    rrs = [reciprocal_rank(yt, yp) for yt, yp in zip(y_true_list, y_pred_list)]
    return np.mean(rrs)

# Example: Question Answering System
print("Question Answering System Evaluation:")
print("="*70)

queries = [
    # Query 1: Best answer is ranked first
    {
        "question": "What is the capital of France?",
        "y_true": np.array([1, 0, 0, 0, 0]),  # First result is correct
        "y_pred": np.array([0.95, 0.6, 0.5, 0.4, 0.3])
    },
    # Query 2: Best answer is ranked third
    {
        "question": "Who invented Python?",
        "y_true": np.array([0, 0, 1, 0, 0]),  # Third result is correct
        "y_pred": np.array([0.7, 0.65, 0.6, 0.5, 0.4])
    },
    # Query 3: Best answer is ranked second
    {
        "question": "What is machine learning?",
        "y_true": np.array([0, 1, 1, 0, 0]),  # Second and third relevant
        "y_pred": np.array([0.8, 0.75, 0.6, 0.5, 0.4])
    },
    # Query 4: No relevant results in top 5
    {
        "question": "How does quantum computing work?",
        "y_true": np.array([0, 0, 0, 0, 0]),  # No good answers
        "y_pred": np.array([0.6, 0.5, 0.45, 0.4, 0.35])
    }
]

rrs = []
for i, query in enumerate(queries, 1):
    rr = reciprocal_rank(query["y_true"], query["y_pred"])
    rrs.append(rr)

    # Find rank of first relevant
    sorted_idx = np.argsort(query["y_pred"])[::-1]
    y_true_sorted = query["y_true"][sorted_idx]

    first_rel_rank = "N/A"
    for rank, rel in enumerate(y_true_sorted, 1):
        if rel:
            first_rel_rank = rank
            break

    print(f"\nQuery {i}: {query['question']}")
    print(f"  First relevant result at position: {first_rel_rank}")
    print(f"  Reciprocal Rank: {rr:.3f}")

    # Show top 3 results
    print(f"  Top 3: ", end="")
    for rank, idx in enumerate(sorted_idx[:3], 1):
        symbol = "✓" if query["y_true"][idx] else "✗"
        print(f"{symbol}", end=" ")
    print()

mrr = np.mean(rrs)
print(f"\n{'='*70}")
print(f"Mean Reciprocal Rank (MRR): {mrr:.3f}")

# Interpretation
print(f"\nInterpretation:")
print(f"On average, users find a relevant answer at position {1/mrr:.1f}")

# Compare with other metrics
print(f"\n{'='*70}")
print("MRR vs Other Metrics:")
print("-"*70)

y_true_all = [q["y_true"] for q in queries]
y_pred_all = [q["y_pred"] for q in queries]

# Calculate MAP for comparison
from sklearn.metrics import label_ranking_average_precision_score

# Reshape for sklearn
y_true_array = np.array(y_true_all)
y_pred_array = np.array(y_pred_all)

map_score = label_ranking_average_precision_score(y_true_array, y_pred_array)

print(f"MRR:  {mrr:.3f} - Focus on first relevant result")
print(f"MAP:  {map_score:.3f} - Consider all relevant results")
print("\n✓ Use MRR for QA, MAP for multi-result retrieval")
```

---

## NLP Metrics

NLP metrics evaluate generated text quality, measuring how well machine-generated text matches human references.

### Perplexity

**Mathematical Formula:**
```
Perplexity = exp(-1/N × Σ log P(w_i | context))
           = 2^(-entropy)

where:
N = number of tokens
P(w_i | context) = model's probability for token w_i
```

**Intuition:**
Perplexity measures how "surprised" a language model is by a text sequence. Lower perplexity = better model (less surprised). It can be interpreted as "the model is as confused as if it had to choose uniformly from N words at each position."

Examples:
- Perplexity = 1: Perfect prediction (always certain)
- Perplexity = 100: As confused as choosing randomly from 100 words
- Perplexity = 1000: Very confused

**When to Use:**
- Evaluating language models
- Comparing different LM architectures
- Model selection and hyperparameter tuning
- NOT for evaluating generation quality (only measures probability)

**Pros:**
- Intrinsic metric (doesn't need references)
- Good for comparing language models
- Widely used and understood
- Useful during training

**Cons:**
- Doesn't measure generation quality directly
- Can't compare across different vocabularies
- Lower perplexity doesn't always mean better generation
- Sensitive to vocabulary size and tokenization

```python
import numpy as np
import torch
import torch.nn as nn

def calculate_perplexity(log_probs):
    """
    Calculate perplexity from log probabilities.

    Args:
        log_probs: Log probabilities of tokens

    Returns:
        Perplexity score
    """
    # Average negative log likelihood
    avg_nll = -np.mean(log_probs)

    # Perplexity is exp of NLL
    perplexity = np.exp(avg_nll)

    return perplexity

# Example: Language model evaluation
print("Language Model Perplexity:")
print("="*70)

# Simulate token predictions from a language model
# Higher probability (lower negative log prob) = better

# Good model: confident about most tokens
log_probs_good = np.array([
    -0.1, -0.2, -0.15, -0.1, -0.3, -0.2, -0.1, -0.25
])  # High probabilities (exp(-0.1) ≈ 0.9)

# Okay model: less confident
log_probs_okay = np.array([
    -1.0, -1.2, -0.8, -1.5, -1.1, -0.9, -1.3, -1.0
])  # Medium probabilities (exp(-1.0) ≈ 0.37)

# Bad model: very uncertain
log_probs_bad = np.array([
    -3.0, -3.5, -2.8, -3.2, -3.1, -3.4, -2.9, -3.3
])  # Low probabilities (exp(-3.0) ≈ 0.05)

ppl_good = calculate_perplexity(log_probs_good)
ppl_okay = calculate_perplexity(log_probs_okay)
ppl_bad = calculate_perplexity(log_probs_bad)

print(f"Good model perplexity: {ppl_good:.2f}")
print(f"Okay model perplexity: {ppl_okay:.2f}")
print(f"Bad model perplexity: {ppl_bad:.2f}")
print("\n✓ Lower perplexity = Better language model\n")

# Real example with PyTorch
print("="*70)
print("PyTorch Language Model Example:")
print("-"*70)

# Simulated language model output
# Vocabulary size = 1000
vocab_size = 1000
sequence_length = 20

# Target tokens (ground truth)
targets = torch.randint(0, vocab_size, (sequence_length,))

# Model logits (before softmax)
logits_good = torch.randn(sequence_length, vocab_size)
# Boost correct predictions for good model
for i, target in enumerate(targets):
    logits_good[i, target] += 3.0

logits_bad = torch.randn(sequence_length, vocab_size)

# Calculate perplexity using cross-entropy loss
criterion = nn.CrossEntropyLoss()

loss_good = criterion(logits_good, targets)
loss_bad = criterion(logits_bad, targets)

ppl_good_torch = torch.exp(loss_good).item()
ppl_bad_torch = torch.exp(loss_bad).item()

print(f"\nGood model:")
print(f"  Cross-entropy loss: {loss_good:.3f}")
print(f"  Perplexity: {ppl_good_torch:.2f}")

print(f"\nBad model:")
print(f"  Cross-entropy loss: {loss_bad:.3f}")
print(f"  Perplexity: {ppl_bad_torch:.2f}")

# Interpretation
print(f"\n{'='*70}")
print("Interpretation:")
print(f"Good model perplexity {ppl_good_torch:.0f} means:")
print(f"  'On average, the model is as uncertain as if choosing")
print(f"   uniformly from {ppl_good_torch:.0f} words'")

# Relationship to accuracy
print(f"\n{'='*70}")
print("Perplexity vs Accuracy:")
print("-"*70)

# Calculate top-1 accuracy
_, predicted_good = logits_good.max(dim=1)
_, predicted_bad = logits_bad.max(dim=1)

acc_good = (predicted_good == targets).float().mean().item()
acc_bad = (predicted_bad == targets).float().mean().item()

print(f"Good model: PPL={ppl_good_torch:.1f}, Accuracy={acc_good:.1%}")
print(f"Bad model:  PPL={ppl_bad_torch:.1f}, Accuracy={acc_bad:.1%}")
print("\n✓ Lower perplexity generally correlates with higher accuracy")
```

### BLEU (Bilingual Evaluation Understudy)

**Mathematical Formula:**
```
BLEU = BP × exp(Σ w_n log p_n)

where:
p_n = n-gram precision (modified to avoid repetition)
BP = Brevity Penalty = min(1, exp(1 - ref_len/pred_len))
w_n = weights for different n-grams (usually uniform: 1/4 each for 1,2,3,4-grams)
```

**Intuition:**
BLEU measures how many n-grams (word sequences) in the generated text appear in reference translations. It combines:
1. **Precision**: What fraction of generated n-grams match the reference?
2. **Brevity Penalty**: Penalizes short translations (prevents gaming by generating very short texts)
3. **Multiple n-grams**: Checks 1-grams (words), 2-grams (pairs), 3-grams, 4-grams

BLEU scores range from 0 (no match) to 1 (perfect match), often reported as 0-100.

**When to Use:**
- Machine translation evaluation
- Text summarization
- Image captioning
- Any task where output should match reference text

**Pros:**
- Fast and easy to compute
- Language-independent
- Correlates reasonably with human judgment
- Standard metric for MT

**Cons:**
- Doesn't consider meaning/semantics
- Rewards exact matches only (synonyms don't count)
- Brevity penalty can be harsh
- Doesn't work well for single sentences (needs corpus-level)

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import numpy as np

# Example: Machine Translation Evaluation
print("BLEU Score for Machine Translation:")
print("="*70)

# Reference translation (can have multiple references)
reference = ["the", "cat", "is", "on", "the", "mat"]

# Different translation candidates
candidate_perfect = ["the", "cat", "is", "on", "the", "mat"]
candidate_good = ["the", "cat", "sits", "on", "the", "mat"]
candidate_okay = ["a", "cat", "is", "on", "the", "mat"]
candidate_bad = ["there", "is", "a", "cat", "here"]

candidates = [
    ("Perfect match", candidate_perfect),
    ("Good (1 word different)", candidate_good),
    ("Okay (2 words different)", candidate_okay),
    ("Bad (different structure)", candidate_bad)
]

# Calculate BLEU for each candidate
smoothing = SmoothingFunction().method1  # Avoid zero scores

for name, candidate in candidates:
    # Single reference
    bleu = sentence_bleu([reference], candidate, smoothing_function=smoothing)

    print(f"\n{name}:")
    print(f"  Candidate: {' '.join(candidate)}")
    print(f"  BLEU: {bleu:.3f} ({bleu*100:.1f}/100)")

# Detailed n-gram analysis
print(f"\n{'='*70}")
print("N-gram Analysis for 'Good' translation:")
print("-"*70)

candidate = candidate_good

# Calculate BLEU with different n-gram weights
# Default BLEU-4: equal weights to 1,2,3,4-grams
bleu_4 = sentence_bleu([reference], candidate, smoothing_function=smoothing)

# BLEU-1: only unigrams
bleu_1 = sentence_bleu([reference], candidate, weights=(1, 0, 0, 0))

# BLEU-2: only bigrams
bleu_2 = sentence_bleu([reference], candidate, weights=(0, 1, 0, 0))

print(f"BLEU-1 (unigrams only): {bleu_1:.3f}")
print(f"BLEU-2 (bigrams only):  {bleu_2:.3f}")
print(f"BLEU-4 (1-4 grams):     {bleu_4:.3f}")

# Multiple references (more realistic)
print(f"\n{'='*70}")
print("Multiple Reference Translations:")
print("-"*70)

references = [
    ["the", "cat", "is", "on", "the", "mat"],
    ["the", "cat", "sits", "on", "the", "mat"],
    ["there", "is", "a", "cat", "on", "the", "mat"]
]

candidate_multi = ["the", "cat", "is", "sitting", "on", "the", "mat"]

bleu_single = sentence_bleu([references[0]], candidate_multi,
                            smoothing_function=smoothing)
bleu_multi = sentence_bleu(references, candidate_multi,
                          smoothing_function=smoothing)

print(f"Candidate: {' '.join(candidate_multi)}")
print(f"\nWith 1 reference: BLEU = {bleu_single:.3f}")
print(f"With 3 references: BLEU = {bleu_multi:.3f}")
print("✓ Multiple references usually give higher scores")

# Demonstrating brevity penalty
print(f"\n{'='*70}")
print("Brevity Penalty Effect:")
print("-"*70)

reference_bp = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]

candidate_full = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
candidate_short = ["the", "quick", "brown", "fox"]  # Truncated

bleu_full = sentence_bleu([reference_bp], candidate_full)
bleu_short = sentence_bleu([reference_bp], candidate_short)

print(f"Full translation: {' '.join(candidate_full)}")
print(f"BLEU: {bleu_full:.3f}\n")

print(f"Short translation: {' '.join(candidate_short)}")
print(f"BLEU: {bleu_short:.3f}")
print("✓ Brevity penalty reduces score for short translations")

# BLEU interpretation guide
print(f"\n{'='*70}")
print("BLEU Score Interpretation (for MT):")
print("-"*70)
print("  < 10: Almost useless")
print(" 10-20: Difficult to get the gist")
print(" 20-30: Clear gist, significant grammatical errors")
print(" 30-40: Understandable, some grammatical errors")
print(" 40-50: High quality, fluent")
print(" 50-60: Very high quality, near-human")
print("   60+: Often better than human (rare)")
```

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

**Mathematical Formula:**
```
ROUGE-N:
Recall = Σ Count_match(n-gram) / Σ Count(n-gram in reference)
Precision = Σ Count_match(n-gram) / Σ Count(n-gram in candidate)
F1 = 2 × (Precision × Recall) / (Precision + Recall)

ROUGE-L:
Based on Longest Common Subsequence (LCS)
```

**Intuition:**
While BLEU focuses on precision (how much of the generation is in the reference), ROUGE focuses on recall (how much of the reference is in the generation). This makes ROUGE better for summarization:
- **BLEU**: "Is the generated text correct?" (good for translation)
- **ROUGE**: "Does the summary cover the important content?" (good for summarization)

ROUGE variants:
- **ROUGE-N**: N-gram overlap (ROUGE-1 = unigrams, ROUGE-2 = bigrams)
- **ROUGE-L**: Longest Common Subsequence (captures sentence-level structure)
- **ROUGE-S**: Skip-bigram overlap (allows gaps)

**When to Use:**
- Text summarization (primary use case)
- Question answering
- When recall is more important than precision
- Comparing generated text to human references

**Pros:**
- Recall-oriented (good for summarization)
- Multiple variants capture different aspects
- Correlates well with human judgment for summarization
- Fast to compute

**Cons:**
- Doesn't capture semantics
- Sensitive to word choice (synonyms don't match)
- Can be gamed by including more content
- Needs reference summaries

```python
from rouge_score import rouge_scorer
import numpy as np

print("ROUGE Scores for Text Summarization:")
print("="*70)

# Original document (simplified news article)
document = """
The new climate report released today shows alarming trends in global
temperatures. Scientists warn that without immediate action, sea levels
could rise by 2 meters by 2100. The report recommends reducing carbon
emissions by 50% in the next decade to avoid catastrophic consequences.
"""

# Reference summary (human-written)
reference = "Climate report warns of rising sea levels and urges 50% emission cuts."

# Different summary candidates
summaries = [
    ("Good summary",
     "New climate report shows alarming temperature trends and urges immediate emission reductions."),

    ("Too short - misses key info",
     "Climate report released today."),

    ("Too detailed - includes minor info",
     "Climate report released today shows temperature trends. Scientists warn about sea levels rising 2 meters by 2100."),

    ("Different wording - same meaning",
     "Latest climate study indicates severe temperature increases and calls for halving carbon output."),
]

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

print("\nReference summary:")
print(f'"{reference}"\n')
print("="*70)

for name, candidate in summaries:
    scores = scorer.score(reference, candidate)

    print(f"\n{name}:")
    print(f'"{candidate}"')
    print()
    print(f"  ROUGE-1 (unigram) - Recall: {scores['rouge1'].recall:.3f}, " +
          f"Precision: {scores['rouge1'].precision:.3f}, " +
          f"F1: {scores['rouge1'].fmeasure:.3f}")
    print(f"  ROUGE-2 (bigram)  - Recall: {scores['rouge2'].recall:.3f}, " +
          f"Precision: {scores['rouge2'].precision:.3f}, " +
          f"F1: {scores['rouge2'].fmeasure:.3f}")
    print(f"  ROUGE-L (LCS)     - Recall: {scores['rougeL'].recall:.3f}, " +
          f"Precision: {scores['rougeL'].precision:.3f}, " +
          f"F1: {scores['rougeL'].fmeasure:.3f}")

# Understanding Recall vs Precision in ROUGE
print(f"\n{'='*70}")
print("Understanding ROUGE Recall vs Precision:")
print("-"*70)

ref_short = "The cat sat on the mat"
cand_verbose = "The big fluffy cat sat comfortably on the soft mat yesterday"
cand_minimal = "Cat on mat"

scores_verbose = scorer.score(ref_short, cand_verbose)
scores_minimal = scorer.score(ref_short, cand_minimal)

print("\nVerbose summary (includes extra words):")
print(f'"{cand_verbose}"')
print(f"  Recall: {scores_verbose['rouge1'].recall:.3f} ← High (covers reference)")
print(f"  Precision: {scores_verbose['rouge1'].precision:.3f} ← Low (extra words)")

print("\nMinimal summary (too short):")
print(f'"{cand_minimal}"')
print(f"  Recall: {scores_minimal['rouge1'].recall:.3f} ← Low (misses content)")
print(f"  Precision: {scores_minimal['rouge1'].precision:.3f} ← High (no extra words)")

# ROUGE-1 vs ROUGE-2 vs ROUGE-L
print(f"\n{'='*70}")
print("ROUGE-1 vs ROUGE-2 vs ROUGE-L:")
print("-"*70)

ref_example = "The quick brown fox jumps over the lazy dog"
cand_shuffled = "The lazy dog jumps over the quick brown fox"  # Same words, wrong order
cand_partial = "The quick brown fox walks slowly"  # Some words match

for name, cand in [("Shuffled order", cand_shuffled),
                   ("Partial match", cand_partial)]:
    scores = scorer.score(ref_example, cand)
    print(f"\n{name}:")
    print(f'"{cand}"')
    print(f"  ROUGE-1: {scores['rouge1'].fmeasure:.3f} ← Word overlap")
    print(f"  ROUGE-2: {scores['rouge2'].fmeasure:.3f} ← Phrase overlap")
    print(f"  ROUGE-L: {scores['rougeL'].fmeasure:.3f} ← Sequence order")

print(f"\n{'='*70}")
print("Typical ROUGE score ranges (summarization):")
print("-"*70)
print("  ROUGE-1: 0.3-0.5 (good), 0.5+ (excellent)")
print("  ROUGE-2: 0.1-0.25 (good), 0.25+ (excellent)")
print("  ROUGE-L: 0.25-0.45 (good), 0.45+ (excellent)")
print("\nROUGE-1 > ROUGE-2 > ROUGE-L is typical")
```

### METEOR (Metric for Evaluation of Translation with Explicit ORdering)

**Intuition:**
METEOR improves upon BLEU by:
1. Using stemming and synonyms (WordNet)
2. Considering both precision and recall (not just precision like BLEU)
3. Penalizing fragmentation (chunks of matches are better than scattered matches)

This makes METEOR more aligned with human judgment than BLEU.

**When to Use:**
- Machine translation evaluation
- When you want more semantic awareness than BLEU
- When word order matters but synonyms should count

```python
# METEOR requires nltk
from nltk.translate.meteor_score import meteor_score
import nltk

# Download required NLTK data (uncomment if needed)
# nltk.download('wordnet')
# nltk.download('omw-1.4')

print("METEOR vs BLEU Comparison:")
print("="*70)

# Reference
reference = ["the", "cat", "is", "on", "the", "mat"]

# Candidates
candidates = [
    ("Exact match",
     ["the", "cat", "is", "on", "the", "mat"]),

    ("Synonym used",
     ["the", "feline", "is", "on", "the", "mat"]),  # 'feline' ~ 'cat'

    ("Similar meaning, different words",
     ["a", "cat", "sits", "on", "the", "rug"]),  # 'sits' ~ 'is', 'rug' ~ 'mat'
]

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smoothing = SmoothingFunction().method1

print("\nReference:", " ".join(reference))
print("="*70)

for name, candidate in candidates:
    # Calculate BLEU
    bleu = sentence_bleu([reference], candidate, smoothing_function=smoothing)

    # Calculate METEOR
    # METEOR expects strings, not token lists
    ref_str = " ".join(reference)
    cand_str = " ".join(candidate)
    meteor = meteor_score([ref_str], cand_str)

    print(f"\n{name}:")
    print(f"  Candidate: {' '.join(candidate)}")
    print(f"  BLEU:   {bleu:.3f}")
    print(f"  METEOR: {meteor:.3f}")

print(f"\n{'='*70}")
print("Key Differences:")
print("-"*70)
print("BLEU:   Exact match only, precision-focused")
print("METEOR: Synonyms match, balances precision/recall, considers order")
print("\n✓ METEOR generally correlates better with human judgment")
```

### BERTScore

**Intuition:**
BERTScore uses contextual embeddings from BERT to measure semantic similarity, going beyond surface-level word matching. Instead of checking if words match exactly, it computes:
1. Embedding for each token in candidate and reference
2. Cosine similarity between embeddings
3. Greedy matching to align candidate and reference tokens
4. Average of similarity scores

This captures semantic similarity even when words are different.

**When to Use:**
- When semantic similarity is more important than exact wording
- Evaluating paraphrasing, summarization, generation
- When references might use different words with same meaning
- Modern alternative to BLEU/ROUGE

**Pros:**
- Captures semantic similarity
- Works with paraphrases and synonyms
- Correlates very well with human judgment
- Language-agnostic (works for many languages)

**Cons:**
- Computationally expensive (requires BERT)
- Harder to interpret than n-gram metrics
- Requires GPU for reasonable speed
- Scores can be high even for somewhat different meanings

```python
# BERTScore requires the bert-score package
# Install with: pip install bert-score

try:
    from bert_score import score as bert_score

    print("BERTScore Evaluation:")
    print("="*70)

    # Reference
    references = ["The cat is sitting on the mat"]

    # Candidates with varying semantic similarity
    candidates = [
        "The cat is sitting on the mat",           # Exact match
        "A cat sits on the mat",                   # Paraphrase
        "The feline is resting on the carpet",     # Synonyms
        "There is a cat on a mat",                 # Different structure, same meaning
        "The dog is running in the park"           # Different meaning
    ]

    # Calculate BERTScore
    P, R, F1 = bert_score(candidates, references, lang="en", verbose=False)

    print("\nReference:", references[0])
    print("="*70)

    for i, candidate in enumerate(candidates):
        print(f"\nCandidate {i+1}: {candidate}")
        print(f"  Precision: {P[i]:.3f}")
        print(f"  Recall:    {R[i]:.3f}")
        print(f"  F1:        {F1[i]:.3f}")

    print(f"\n{'='*70}")
    print("Notice: Even with different words (candidate 3),")
    print("BERTScore recognizes semantic similarity!")

except ImportError:
    print("BERTScore not installed.")
    print("Install with: pip install bert-score")
    print("\nBERTScore uses BERT embeddings to measure semantic similarity.")
    print("It matches tokens based on contextual meaning, not exact words.")
```

---

## Clustering Metrics

Clustering metrics evaluate how well unsupervised algorithms group similar data points.

### Silhouette Score

**Mathematical Formula:**
```
For each sample i:
a(i) = average distance to other points in same cluster
b(i) = average distance to points in nearest other cluster

Silhouette(i) = (b(i) - a(i)) / max(a(i), b(i))

Silhouette Score = average of Silhouette(i) across all samples
```

**Intuition:**
Silhouette score measures how similar a point is to its own cluster compared to other clusters:
- s ≈ 1: Point is well-matched to its cluster and far from others (good)
- s ≈ 0: Point is on the boundary between clusters (ambiguous)
- s < 0: Point might be in the wrong cluster (bad)

**When to Use:**
- Evaluating clustering quality
- Choosing optimal number of clusters
- Comparing different clustering algorithms
- When you don't have ground truth labels

**Pros:**
- Intuitive interpretation (-1 to 1)
- Works without ground truth
- Considers both cohesion and separation
- Useful for determining number of clusters

**Cons:**
- Computationally expensive for large datasets
- Favors convex clusters
- Sensitive to outliers
- Not suitable for density-based clusters (DBSCAN)

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data with clear clusters
X, y_true = make_blobs(n_samples=300, centers=4, n_features=2,
                       cluster_std=0.6, random_state=42)

print("Silhouette Score Analysis:")
print("="*70)

# Try different numbers of clusters
for n_clusters in [2, 3, 4, 5, 6]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(X, cluster_labels)

    print(f"\nNumber of clusters: {n_clusters}")
    print(f"  Silhouette Score: {silhouette_avg:.3f}", end="")

    if n_clusters == 4:  # True number of clusters
        print(" ← Optimal (matches true clusters)")
    elif abs(n_clusters - 4) <= 1:
        print(" ← Good")
    else:
        print(" ← Suboptimal")

# Detailed analysis for optimal clustering
print(f"\n{'='*70}")
print("Detailed Silhouette Analysis (k=4):")
print("-"*70)

kmeans_best = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_best = kmeans_best.fit_predict(X)

# Per-sample silhouette scores
silhouette_vals = silhouette_samples(X, labels_best)

# Analyze each cluster
for cluster_id in range(4):
    cluster_silhouette_vals = silhouette_vals[labels_best == cluster_id]

    print(f"\nCluster {cluster_id}:")
    print(f"  Size: {len(cluster_silhouette_vals)}")
    print(f"  Avg Silhouette: {cluster_silhouette_vals.mean():.3f}")
    print(f"  Min Silhouette: {cluster_silhouette_vals.min():.3f}")
    print(f"  Max Silhouette: {cluster_silhouette_vals.max():.3f}")

    # Count poorly assigned points
    poorly_assigned = np.sum(cluster_silhouette_vals < 0)
    if poorly_assigned > 0:
        print(f"  ⚠ {poorly_assigned} points might be in wrong cluster")

# Silhouette score interpretation
print(f"\n{'='*70}")
print("Silhouette Score Interpretation:")
print("-"*70)
print("  0.71-1.0:  Strong structure")
print("  0.51-0.70: Reasonable structure")
print("  0.26-0.50: Weak structure, might be artificial")
print("  < 0.25:    No substantial structure")

# Visualize (uncomment to see plot)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
#
# # Scatter plot
# scatter = ax1.scatter(X[:, 0], X[:, 1], c=labels_best, cmap='viridis', alpha=0.6)
# ax1.scatter(kmeans_best.cluster_centers_[:, 0],
#             kmeans_best.cluster_centers_[:, 1],
#             c='red', marker='X', s=200, label='Centroids')
# ax1.set_title(f'Clusters (Silhouette = {silhouette_score(X, labels_best):.3f})')
# ax1.legend()
#
# # Silhouette plot
# y_lower = 10
# for i in range(4):
#     cluster_silhouette_vals = silhouette_vals[labels_best == i]
#     cluster_silhouette_vals.sort()
#
#     size_cluster_i = cluster_silhouette_vals.shape[0]
#     y_upper = y_lower + size_cluster_i
#
#     ax2.fill_betweenx(np.arange(y_lower, y_upper),
#                       0, cluster_silhouette_vals,
#                       alpha=0.7)
#
#     y_lower = y_upper + 10
#
# ax2.axvline(x=silhouette_score(X, labels_best), color="red", linestyle="--",
#             label='Average')
# ax2.set_title('Silhouette Plot for Each Cluster')
# ax2.set_xlabel('Silhouette Coefficient')
# ax2.set_ylabel('Cluster')
# ax2.legend()
# plt.show()
```

### Davies-Bouldin Index

**Mathematical Formula:**
```
DB = (1/k) × Σ max_j(R_ij)

where:
R_ij = (s_i + s_j) / d_ij
s_i = average distance of points in cluster i to centroid
d_ij = distance between centroids of clusters i and j
```

**Intuition:**
Davies-Bouldin Index measures the ratio of within-cluster scatter to between-cluster separation. Lower is better:
- Low DB: Clusters are compact (small s_i) and well-separated (large d_ij)
- High DB: Clusters are spread out or close together

Unlike Silhouette, DB is faster to compute and only uses centroids.

**When to Use:**
- Quick clustering evaluation
- Choosing number of clusters
- When computational efficiency matters
- Centroid-based clustering algorithms (K-means, hierarchical)

**Pros:**
- Fast to compute
- Intuitive (based on cluster scatter and separation)
- Works without ground truth

**Cons:**
- Assumes convex clusters
- Sensitive to outliers
- Requires centroids (not suitable for DBSCAN)
- Lower bound is 0 but no upper bound

```python
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np

# Generate data
X, _ = make_blobs(n_samples=300, centers=4, n_features=2,
                  cluster_std=0.6, random_state=42)

print("Davies-Bouldin Index Analysis:")
print("="*70)

# Try different numbers of clusters
scores = []
for n_clusters in range(2, 8):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    db_score = davies_bouldin_score(X, labels)
    scores.append((n_clusters, db_score))

    print(f"k={n_clusters}: DB Index = {db_score:.3f}", end="")
    if n_clusters == 4:
        print(" ← Lowest (best)")
    else:
        print()

# Find optimal k
optimal_k = min(scores, key=lambda x: x[1])[0]
print(f"\n✓ Optimal number of clusters: {optimal_k}")

print(f"\n{'='*70}")
print("Davies-Bouldin Index Interpretation:")
print("-"*70)
print("  • Lower is better (0 is optimal)")
print("  • Measures ratio of within-cluster to between-cluster distances")
print("  • No fixed scale (compare relative values)")
print("  • Sensitive to number of clusters and their separation")
```

### Calinski-Harabasz Index (Variance Ratio Criterion)

**Mathematical Formula:**
```
CH = (SS_B / SS_W) × ((N - k) / (k - 1))

where:
SS_B = between-cluster variance
SS_W = within-cluster variance
N = number of samples
k = number of clusters
```

**Intuition:**
Calinski-Harabasz Index is the ratio of between-cluster variance to within-cluster variance. Higher is better:
- High CH: Clusters are dense and well-separated
- Low CH: Clusters are sparse or poorly separated

It's analogous to F-statistic in ANOVA.

**When to Use:**
- Quick clustering evaluation
- Choosing number of clusters
- Alternative to Silhouette (faster)
- Convex cluster shapes

**Pros:**
- Very fast to compute
- Higher values = better clustering (intuitive)
- Works well for convex clusters
- Good for comparing different k values

**Cons:**
- Assumes convex clusters
- Not suitable for density-based clustering
- No absolute interpretation (relative comparison only)

```python
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np

# Generate data
X, _ = make_blobs(n_samples=300, centers=4, n_features=2,
                  cluster_std=0.6, random_state=42)

print("Calinski-Harabasz Index Analysis:")
print("="*70)

# Try different numbers of clusters
scores = []
for n_clusters in range(2, 8):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    ch_score = calinski_harabasz_score(X, labels)
    scores.append((n_clusters, ch_score))

    print(f"k={n_clusters}: CH Index = {ch_score:.1f}", end="")
    if n_clusters == 4:
        print(" ← Highest (best)")
    else:
        print()

# Find optimal k
optimal_k = max(scores, key=lambda x: x[1])[0]
print(f"\n✓ Optimal number of clusters: {optimal_k}")

# Compare all three metrics
print(f"\n{'='*70}")
print("Comparing Clustering Metrics (k=4):")
print("-"*70)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

silhouette = silhouette_score(X, labels)
davies_bouldin = davies_bouldin_score(X, labels)
calinski = calinski_harabasz_score(X, labels)

print(f"Silhouette Score:      {silhouette:.3f}  (higher is better, [-1,1])")
print(f"Davies-Bouldin Index:  {davies_bouldin:.3f}  (lower is better, [0,∞))")
print(f"Calinski-Harabasz:     {calinski:.1f}  (higher is better, [0,∞))")

print("\n✓ All metrics agree that k=4 is optimal!")
```

---

## Object Detection Metrics

Object detection metrics evaluate both localization (where is the object?) and classification (what is it?).

### Intersection over Union (IoU)

**Mathematical Formula:**
```
IoU = Area of Overlap / Area of Union
    = (Predicted Box ∩ Ground Truth Box) / (Predicted Box ∪ Ground Truth Box)
```

**Intuition:**
IoU measures how well two bounding boxes overlap:
- IoU = 1: Perfect overlap
- IoU = 0.5: Moderate overlap (common threshold for "correct" detection)
- IoU = 0: No overlap

It's used to determine if a predicted bounding box correctly detects an object.

**When to Use:**
- Object detection evaluation
- Semantic segmentation
- Instance segmentation
- Any task involving bounding boxes or masks

**Thresholds:**
- IoU ≥ 0.5: Common threshold for "correct" detection
- IoU ≥ 0.75: Strict threshold
- IoU ≥ [0.5:0.95]: COCO evaluation (average over multiple thresholds)

```python
import numpy as np

def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes.

    Args:
        box1, box2: [x1, y1, x2, y2] where (x1,y1) is top-left, (x2,y2) is bottom-right

    Returns:
        IoU score [0, 1]
    """
    # Coordinates of intersection rectangle
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Area of intersection
    if x2_inter < x1_inter or y2_inter < y1_inter:
        intersection = 0  # No overlap
    else:
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # Areas of both boxes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Area of union
    union = area1 + area2 - intersection

    # IoU
    iou = intersection / union if union > 0 else 0

    return iou

# Example: Object detection on an image
print("Intersection over Union (IoU):")
print("="*70)

# Ground truth box (actual object location)
# Format: [x1, y1, x2, y2]
ground_truth = [100, 100, 200, 200]  # 100x100 box

# Different prediction scenarios
predictions = [
    ("Perfect match", [100, 100, 200, 200]),
    ("Good detection", [95, 95, 205, 205]),
    ("Moderate overlap", [150, 150, 250, 250]),
    ("Poor detection", [180, 180, 280, 280]),
    ("No overlap", [300, 300, 400, 400]),
]

print(f"Ground Truth Box: {ground_truth}\n")

for name, pred_box in predictions:
    iou = calculate_iou(ground_truth, pred_box)

    # Determine if detection is "correct" at different thresholds
    status = []
    if iou >= 0.5:
        status.append("✓ PASS IoU≥0.5")
    else:
        status.append("✗ FAIL IoU≥0.5")

    if iou >= 0.75:
        status.append("✓ PASS IoU≥0.75")
    else:
        status.append("✗ FAIL IoU≥0.75")

    print(f"{name}:")
    print(f"  Predicted Box: {pred_box}")
    print(f"  IoU: {iou:.3f}")
    print(f"  Status: {', '.join(status)}")
    print()

# Visualizing different IoU thresholds
print("="*70)
print("IoU Threshold Guidelines:")
print("-"*70)
print("  IoU ≥ 0.9:  Almost perfect localization")
print("  IoU ≥ 0.75: Good localization (strict)")
print("  IoU ≥ 0.5:  Acceptable localization (standard)")
print("  IoU ≥ 0.3:  Poor localization")
print("  IoU < 0.3:  Very poor/no localization")

# Multiple detections example
print(f"\n{'='*70}")
print("Multiple Detections (Non-Maximum Suppression use case):")
print("-"*70)

# When a model produces multiple boxes for same object
ground_truth_multi = [150, 150, 250, 250]

detections = [
    ("Detection 1", [145, 145, 255, 255], 0.9),  # (box, confidence)
    ("Detection 2", [148, 148, 252, 252], 0.85),
    ("Detection 3", [200, 200, 300, 300], 0.7),
]

print(f"Ground Truth: {ground_truth_multi}\n")

for name, box, conf in detections:
    iou = calculate_iou(ground_truth_multi, box)
    print(f"{name} (confidence {conf}):")
    print(f"  Box: {box}")
    print(f"  IoU: {iou:.3f}")

    if iou >= 0.5 and conf >= 0.5:
        print(f"  Status: ✓ True Positive")
    elif iou >= 0.5:
        print(f"  Status: Low confidence")
    else:
        print(f"  Status: ✗ False Positive (poor localization)")
    print()

print("NMS would keep Detection 1 (highest confidence, good IoU)")
print("and suppress Detection 2 (overlaps too much with Detection 1)")
```

### Mean Average Precision (mAP) for Object Detection

**Mathematical Formula:**
```
For each class:
1. Sort detections by confidence
2. Calculate Precision and Recall at each threshold
3. Calculate AP (area under Precision-Recall curve)

mAP = mean of AP across all classes

mAP@0.5 = mAP with IoU threshold of 0.5
mAP@[.5:.95] = average mAP over IoU thresholds [0.5, 0.55, ..., 0.95]
```

**Intuition:**
mAP combines:
1. **Localization**: IoU determines if detection is correct
2. **Classification**: Predicted class must match ground truth
3. **Confidence**: Higher confidence detections ranked first
4. **Recall**: Ability to find all objects

The metric rewards models that:
- Detect objects with high confidence
- Localize accurately (high IoU)
- Find all instances (high recall)
- Work well across all classes

**When to Use:**
- Standard metric for object detection
- Comparing detection models (YOLO, Faster R-CNN, etc.)
- Benchmark datasets (COCO, Pascal VOC)

**Variants:**
- **mAP@0.5** (Pascal VOC): IoU ≥ 0.5
- **mAP@[.5:.95]** (COCO): Average over IoU from 0.5 to 0.95 (stricter)

```python
import numpy as np

def calculate_map(detections, ground_truths, iou_threshold=0.5):
    """
    Calculate mAP for object detection.

    Args:
        detections: List of (class_id, confidence, box) sorted by confidence
        ground_truths: List of (class_id, box)
        iou_threshold: IoU threshold for considering detection correct

    Returns:
        mAP score
    """
    # This is a simplified version for illustration
    # Real implementations handle multiple images, classes, etc.

    # Sort detections by confidence (descending)
    detections = sorted(detections, key=lambda x: x[1], reverse=True)

    # Track which ground truths have been matched
    gt_matched = [False] * len(ground_truths)

    # Calculate precision and recall at each detection
    tp = 0
    fp = 0
    precisions = []
    recalls = []

    for det_class, det_conf, det_box in detections:
        # Find best matching ground truth
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, (gt_class, gt_box) in enumerate(ground_truths):
            if gt_matched[gt_idx]:
                continue

            if det_class != gt_class:
                continue

            iou = calculate_iou(det_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Is this a true positive?
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1

        precision = tp / (tp + fp)
        recall = tp / len(ground_truths)

        precisions.append(precision)
        recalls.append(recall)

    # Calculate AP (area under precision-recall curve)
    # Using 11-point interpolation (simplified)
    ap = 0
    for recall_threshold in np.linspace(0, 1, 11):
        # Maximum precision for recall >= threshold
        precs_above_threshold = [p for p, r in zip(precisions, recalls)
                                 if r >= recall_threshold]
        if precs_above_threshold:
            ap += max(precs_above_threshold) / 11

    return ap

# Example: Simple object detection scenario
print("Mean Average Precision (mAP) for Object Detection:")
print("="*70)

# Ground truth: 5 objects (class_id, box)
ground_truths = [
    (0, [10, 10, 50, 50]),    # Class 0 (cat)
    (0, [100, 100, 140, 140]), # Class 0 (cat)
    (1, [200, 200, 240, 240]), # Class 1 (dog)
    (1, [300, 300, 340, 340]), # Class 1 (dog)
    (0, [400, 400, 440, 440]), # Class 0 (cat)
]

# Detections: (class_id, confidence, box)
detections = [
    (0, 0.95, [12, 12, 52, 52]),     # TP - good match to GT 1
    (0, 0.90, [98, 98, 138, 138]),   # TP - good match to GT 2
    (1, 0.85, [202, 202, 242, 242]), # TP - good match to GT 3
    (0, 0.80, [50, 50, 90, 90]),     # FP - no matching GT
    (1, 0.75, [305, 305, 345, 345]), # TP - good match to GT 4
    (0, 0.70, [398, 398, 438, 438]), # TP - good match to GT 5
    (1, 0.60, [500, 500, 540, 540]), # FP - no matching GT
]

# Calculate mAP
mAP_05 = calculate_map(detections, ground_truths, iou_threshold=0.5)
mAP_075 = calculate_map(detections, ground_truths, iou_threshold=0.75)

print(f"mAP@0.5:  {mAP_05:.3f}")
print(f"mAP@0.75: {mAP_075:.3f}")

print(f"\n{'='*70}")
print("Detection Analysis:")
print("-"*70)
print(f"Ground Truth Objects: {len(ground_truths)}")
print(f"Detections: {len(detections)}")
print(f"True Positives (IoU≥0.5): ~5")
print(f"False Positives: ~2")
print(f"Missed Detections: 0")

# mAP interpretation
print(f"\n{'='*70}")
print("mAP Score Interpretation (COCO dataset benchmarks):")
print("-"*70)
print("  mAP@[.5:.95]:")
print("    > 0.50: State-of-the-art")
print("    0.40-0.50: Excellent")
print("    0.30-0.40: Good")
print("    0.20-0.30: Moderate")
print("    < 0.20: Poor")
print()
print("  mAP@0.5 (more lenient):")
print("    > 0.70: Excellent")
print("    0.50-0.70: Good")
print("    0.30-0.50: Moderate")
print("    < 0.30: Poor")

print(f"\n{'='*70}")
print("Key Differences from Classification mAP:")
print("-"*70)
print("• Object detection mAP requires BOTH correct class AND localization")
print("• Uses IoU threshold to determine if detection is 'correct'")
print("• Handles multiple objects per image")
print("• More complex: must match predictions to ground truths")
```

---

## Choosing the Right Metric

Selecting the appropriate metric is crucial for model development. The wrong metric can lead to models that appear good but fail in production.

### Decision Framework

```python
import pandas as pd

# Create a decision guide
print("Metric Selection Guide:")
print("="*70)

decision_tree = {
    "Task Type": {
        "Regression": {
            "Considerations": [
                "Outliers present? → Huber Loss or MAE",
                "Large errors very bad? → MSE or RMSE",
                "Need interpretable units? → MAE or RMSE",
                "Want percentage error? → MAPE",
                "Comparing models? → R² or Adjusted R²"
            ],
            "Default": "RMSE (interpretable) or R² (variance explained)"
        },

        "Binary Classification": {
            "Balanced classes": {
                "Equal error costs": "Accuracy",
                "FP more costly": "Precision (minimize false alarms)",
                "FN more costly": "Recall (catch all positives)",
                "Need balance": "F1-Score",
            },
            "Imbalanced classes": {
                "General purpose": "AUC-PR or MCC",
                "Threshold-independent": "AUC-ROC",
                "Positive class focus": "F1-Score or AUC-PR",
                "Account for chance": "Cohen's Kappa or MCC"
            },
            "Default": "AUC-ROC (threshold-free) or F1-Score (single metric)"
        },

        "Multi-class Classification": {
            "All classes equal importance": "Macro-averaged F1",
            "Weighted by class frequency": "Weighted F1",
            "Overall performance": "Micro-averaged F1 or Accuracy",
            "Imbalanced": "Macro F1 or MCC",
            "Default": "Weighted F1 (balanced approach)"
        },

        "Ranking/Retrieval": {
            "Top-K important": "Precision@K, Recall@K",
            "Order matters": "NDCG or MAP",
            "First result critical": "MRR",
            "Binary relevance": "MAP",
            "Graded relevance": "NDCG",
            "Default": "NDCG@10 (order-aware, handles grades)"
        },

        "NLP": {
            "Machine Translation": "BLEU (standard) or METEOR (better)",
            "Summarization": "ROUGE-1, ROUGE-2, ROUGE-L",
            "Semantic similarity": "BERTScore",
            "Language Model": "Perplexity",
            "Default": "Task-specific (BLEU/ROUGE) + BERTScore"
        },

        "Clustering": {
            "Quick evaluation": "Calinski-Harabasz (fastest)",
            "Detailed analysis": "Silhouette Score",
            "Multiple algorithms": "Davies-Bouldin + Silhouette",
            "Find optimal K": "Try all three, look for agreement",
            "Default": "Silhouette Score (most interpretable)"
        },

        "Object Detection": {
            "Bounding box quality": "IoU",
            "Overall performance": "mAP@0.5 or mAP@[.5:.95]",
            "Research/COCO": "mAP@[.5:.95]",
            "Production": "mAP@0.5 (more lenient)",
            "Default": "mAP@0.5 (Pascal VOC) or mAP@[.5:.95] (COCO)"
        }
    }
}

# Example scenarios
print("\nCommon Scenarios and Recommended Metrics:")
print("="*70)

scenarios = [
    ("Medical diagnosis (cancer screening)",
     "Recall (can't miss cases) + AUC-PR (imbalanced)"),

    ("Spam email filter",
     "Precision (avoid blocking important emails) + F1-Score"),

    ("House price prediction",
     "RMSE (interpretable in $) + R² (variance explained)"),

    ("Search engine results",
     "NDCG@10 (order matters) + MAP (average quality)"),

    ("Customer churn prediction (20% churn rate)",
     "AUC-PR (imbalanced) + F1-Score + Precision-Recall curve"),

    ("Machine translation system",
     "BLEU (standard) + METEOR (better) + human evaluation"),

    ("Object detection in images",
     "mAP@0.5 (standard) + mAP@[.5:.95] (strict)"),

    ("Customer segmentation",
     "Silhouette Score + Calinski-Harabasz + business validation"),

    ("Multi-class document classification (balanced)",
     "Accuracy + per-class F1-Score (check weak classes)"),

    ("Rare fraud detection (0.1% fraud rate)",
     "AUC-PR >> AUC-ROC, Precision-Recall curve, MCC"),
]

for scenario, metrics in scenarios:
    print(f"\n{scenario}:")
    print(f"  → {metrics}")

# Metric combinations
print(f"\n{'='*70}")
print("Recommended Metric Combinations:")
print("-"*70)
print("\nRegression:")
print("  Primary: RMSE or R²")
print("  Secondary: MAE (outlier check), residual plots")

print("\nBinary Classification (Imbalanced):")
print("  Primary: AUC-PR or F1-Score")
print("  Secondary: Precision, Recall, Confusion Matrix")
print("  Curve: Precision-Recall curve")

print("\nBinary Classification (Balanced):")
print("  Primary: AUC-ROC or Accuracy")
print("  Secondary: F1-Score, Confusion Matrix")
print("  Curve: ROC curve")

print("\nMulti-class Classification:")
print("  Primary: Macro/Weighted F1")
print("  Secondary: Per-class Precision/Recall, Confusion Matrix")

print("\nRanking/Recommendation:")
print("  Primary: NDCG@K or MAP")
print("  Secondary: Precision@K, Recall@K")

print("\nNLP Generation:")
print("  Primary: Task-specific (BLEU/ROUGE)")
print("  Secondary: BERTScore")
print("  Gold standard: Human evaluation")

print("\nClustering:")
print("  Primary: Silhouette Score")
print("  Secondary: Davies-Bouldin, Calinski-Harabasz")
print("  Validation: Domain expert review")

print("\n" + "="*70)
print("⚠ IMPORTANT: Never rely on a single metric!")
print("Always use multiple complementary metrics and visualizations.")
```

### Common Pitfalls

```python
print("\nCommon Metric Selection Pitfalls:")
print("="*70)

pitfalls = [
    {
        "Pitfall": "Using Accuracy on Imbalanced Data",
        "Problem": "99% accuracy sounds great, but might catch 0% of minority class",
        "Example": "Fraud detection with 1% fraud rate",
        "Solution": "Use AUC-PR, F1-Score, MCC, or Precision-Recall curve",
    },
    {
        "Pitfall": "Optimizing BLEU/ROUGE Alone",
        "Problem": "High scores don't guarantee fluent or meaningful text",
        "Example": "Translation with high BLEU but unnatural phrasing",
        "Solution": "Combine with BERTScore and human evaluation",
    },
    {
        "Pitfall": "Ignoring Class Distribution Changes",
        "Problem": "Model works in dev (10% positives) but fails in prod (1% positives)",
        "Example": "Medical screening rolled out to general population",
        "Solution": "Use threshold-independent metrics (AUC), test on realistic data",
    },
    {
        "Pitfall": "Using MSE with Outliers",
        "Problem": "Few outliers dominate the metric, hide overall performance",
        "Example": "House price prediction with few $10M mansions",
        "Solution": "Use MAE or Huber Loss, analyze outliers separately",
    },
    {
        "Pitfall": "Choosing Metric That Doesn't Match Business Goal",
        "Problem": "Optimizing for what's measurable, not what matters",
        "Example": "Optimizing clicks instead of user satisfaction",
        "Solution": "Align metric with business objective, consider custom metrics",
    },
    {
        "Pitfall": "Not Considering Costs of Errors",
        "Problem": "FP and FN have different costs in reality",
        "Example": "False fraud alert (annoying) vs missing fraud (costly)",
        "Solution": "Use Precision/Recall based on cost, consider weighted metrics",
    },
    {
        "Pitfall": "Using R² with Non-Linear Relationships",
        "Problem": "R² can be low even for perfect non-linear fit",
        "Example": "Predicting with polynomial relationship",
        "Solution": "Check residual plots, use appropriate model",
    },
    {
        "Pitfall": "Forgetting Brevity Penalty in BLEU",
        "Problem": "Short outputs can game the metric",
        "Example": "Translation of 50 words → 5 words, all correct",
        "Solution": "Check output length, use METEOR or ROUGE as well",
    },
]

for i, pitfall in enumerate(pitfalls, 1):
    print(f"\n{i}. {pitfall['Pitfall']}")
    print(f"   Problem: {pitfall['Problem']}")
    print(f"   Example: {pitfall['Example']}")
    print(f"   ✓ Solution: {pitfall['Solution']}")

print("\n" + "="*70)
```

---

## Resources

### Documentation and Tutorials

**Scikit-learn Metrics Guide**
- https://scikit-learn.org/stable/modules/model_evaluation.html
- Comprehensive guide for classification, regression, and clustering metrics

**PyTorch Loss Functions**
- https://pytorch.org/docs/stable/nn.html#loss-functions
- Neural network training objectives

**COCO Detection Evaluation**
- https://cocodataset.org/#detection-eval
- Standard object detection metrics

**Papers With Code**
- https://paperswithcode.com/
- Benchmark datasets and state-of-the-art metrics

### Key Papers

**BLEU (Machine Translation)**
- Papineni et al., 2002: "BLEU: a Method for Automatic Evaluation of Machine Translation"

**ROUGE (Summarization)**
- Lin, 2004: "ROUGE: A Package for Automatic Evaluation of Summaries"

**METEOR (Translation)**
- Banerjee & Lavie, 2005: "METEOR: An Automatic Metric for MT Evaluation"

**BERTScore (Semantic Similarity)**
- Zhang et al., 2019: "BERTScore: Evaluating Text Generation with BERT"

**Object Detection (mAP)**
- Everingham et al., 2010: "The Pascal Visual Object Classes (VOC) Challenge"
- Lin et al., 2014: "Microsoft COCO: Common Objects in Context"

### Best Practices

1. **Always use multiple metrics**: No single metric tells the whole story
2. **Visualize**: Confusion matrices, PR curves, ROC curves provide insights
3. **Understand your data**: Class balance, outliers, distribution matter
4. **Match metric to business goal**: Technical metric ≠ business success
5. **Consider costs**: False positives and false negatives have different impacts
6. **Test on realistic data**: Dev set should match production distribution
7. **Use cross-validation**: Single test set can be misleading
8. **Report confidence intervals**: Metrics are estimates with uncertainty
9. **Compare to baselines**: Simple baselines (random, majority class) provide context
10. **Human evaluation**: For generation tasks, automated metrics are proxies

### Libraries

```python
# Scikit-learn: Most ML metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)

# PyTorch: Loss functions for training
import torch.nn as nn
# CrossEntropyLoss, MSELoss, BCELoss, HuberLoss, etc.

# NLP metrics
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
# pip install bert-score
from bert_score import score as bert_score

# Object detection
# pip install pycocotools
from pycocotools.cocoeval import COCOeval
```

---

## Summary

Machine learning metrics are essential tools for evaluating model performance, but choosing the right metric requires understanding:

1. **Task Type**: Regression, classification, ranking, NLP, clustering, detection
2. **Data Characteristics**: Balanced vs imbalanced, outliers, distribution
3. **Business Objectives**: What errors are most costly? What matters most?
4. **Metric Properties**: Interpretability, sensitivity, computational cost

**Key Takeaways:**
- **Regression**: RMSE for interpretability, R² for variance explained, MAE for outlier robustness
- **Classification**: AUC-ROC/F1 for balanced, AUC-PR/MCC for imbalanced, Precision/Recall based on cost
- **Multi-class**: Macro F1 for equal class importance, Weighted F1 for balanced approach
- **Ranking**: NDCG for order-aware, MAP for average quality, P@K/R@K for top results
- **NLP**: BLEU for translation, ROUGE for summarization, BERTScore for semantics, Perplexity for LMs
- **Clustering**: Silhouette for interpretability, Calinski-Harabasz for speed
- **Detection**: IoU for localization, mAP for overall performance

Always combine multiple metrics and visualizations to get a complete picture of model performance. The best metric aligns with your business goals and the costs of different types of errors.

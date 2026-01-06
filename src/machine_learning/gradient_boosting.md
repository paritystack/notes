# Gradient Boosting

> **Domain:** Machine Learning, Tabular Data
> **Key Concepts:** Ensemble Learning, Decision Trees, Residuals, XGBoost, LightGBM

**Gradient Boosting** is a powerful ensemble machine learning technique primarily used for regression and classification tasks. While Deep Learning dominates unstructured data (images, text), Gradient Boosting (specifically XGBoost/LightGBM) dominates structured/tabular data competitions (Kaggle).

---

## 1. The Core Idea: Boosting vs. Bagging

*   **Bagging (Bootstrap Aggregating):**
    *   *Example:* Random Forest.
    *   *Method:* Train $N$ trees independently in parallel on random subsets of data. Average their predictions.
    *   *Goal:* Reduce Variance (Overfitting).
*   **Boosting:**
    *   *Example:* Gradient Boosting Machine (GBM).
    *   *Method:* Train trees **sequentially**. Each new tree tries to correct the errors (residuals) of the previous tree.
    *   *Goal:* Reduce Bias (Underfitting).

---

## 2. How it Works (The Algorithm)

Imagine predicting House Prices.
1.  **Tree 1:** Makes a crude prediction. Average price = $100k.
    *   *True Value:* $150k. *Error (Residual):* +$50k.
2.  **Tree 2:** Predicts the **Residual** of Tree 1. It tries to predict "+$50k".
    *   *Prediction:* +$40k.
    *   *New Combined Prediction:* $100k + $40k = $140k.
    *   *New Residual:* +$10k.
3.  **Tree 3:** Predicts the Residual of Tree 1+2.
    *   *Prediction:* +$8k.
    *   *New Combined Prediction:* $148k.

Mathematically, we move in the direction of the **negative gradient** of the loss function.

---

## 3. Key Implementations

### 3.1. XGBoost (Extreme Gradient Boosting)
The library that popularized GBMs.
*   **Innovations:**
    *   *Regularization:* Adds L1/L2 regularization to the loss function to prevent overfitting.
    *   *Sparsity Awareness:* Handles missing values automatically.
    *   *System Optimization:* Parallelized tree construction (block structure).

### 3.2. LightGBM (Light Gradient Boosting Machine)
Developed by Microsoft. faster and uses less memory than XGBoost.
*   **Innovation:** **GOSS (Gradient-based One-Side Sampling)**. It keeps all instances with large gradients (large errors) and randomly samples instances with small gradients.
*   **Leaf-wise Growth:** Grows the tree by splitting the leaf with the max delta loss, rather than level-wise (balanced tree). Can overfit easier but converges faster.

### 3.3. CatBoost (Categorical Boosting)
Developed by Yandex.
*   **Innovation:** Handles **Categorical Features** natively (no need for One-Hot Encoding). Uses "Ordered Boosting" to prevent data leakage during encoding.

---

## 4. Hyperparameters to Tune

1.  **Learning Rate (Eta):**
    *   Step size shrinkage used in update to prevents overfitting. Lower eta (0.01) requires more trees (n_estimators) but generalizes better.
2.  **Max Depth:**
    *   Maximum depth of a tree. Used to control over-fitting. Typical values: 3-10.
3.  **Subsample:**
    *   Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees.
4.  **Colsample_bytree:**
    *   Subsample ratio of columns (features) when constructing each tree.

---

## 5. Python Example (XGBoost)

```python
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# 1. Prepare Data
X, y = load_data()
dtrain = xgb.DMatrix(X, label=y)

# 2. Set Parameters
param = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

# 3. Train
num_round = 100
bst = xgb.train(param, dtrain, num_round)

# 4. Predict
preds = bst.predict(dtrain)
```

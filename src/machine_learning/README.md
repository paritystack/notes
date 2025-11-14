# Machine Learning

A comprehensive guide to machine learning concepts, algorithms, and implementations.

## Table of Contents

1. [Supervised Learning](./supervised_learning.md) - Classification, regression, and supervised algorithms
2. [Unsupervised Learning](./unsupervised_learning.md) - Clustering and dimensionality reduction
3. [Reinforcement Learning](./reinforcement_learning.md) - RL concepts, Q-learning, and policy gradients
4. [Deep Learning](./deep_learning.md) - Neural networks, CNNs, RNNs, and training techniques
5. [Neural Networks](./neural_networks.md) - Architecture, backpropagation, activation functions
6. [Deep Reinforcement Learning](./deep_reinforcement_learning.md) - DQN, A3C, PPO, and advanced RL
7. [Generative Models](./generative_models.md) - GANs, VAEs, and flow-based models
8. [Deep Generative Models](./deep_generative_models.md) - Advanced generative architectures
9. [Transfer Learning](./transfer_learning.md) - Pre-training, fine-tuning, and domain adaptation
10. [PyTorch](./pytorch.md) - Deep learning framework, tensors, autograd, training
11. [NumPy](./numpy.md) - Foundational numerical computing for ML implementations
12. [Quantization](./quantization.md) - Model compression, INT8/INT4 quantization, GPTQ, AWQ
13. [Transformers](./transformers.md) - Attention mechanisms, BERT, GPT architectures
14. [Hugging Face](./hugging_face.md) - Transformers library, models, and datasets
15. [Interesting Papers](./interesting_papers.md) - Key ML papers and summaries

## Overview

Machine Learning is a field of artificial intelligence that focuses on building systems that learn from data. The field can be broadly categorized into:

### Supervised Learning
Learning from labeled data where each example has an input-output pair. The goal is to learn a mapping from inputs to outputs.
- **Classification**: Predicting discrete categories (e.g., spam/not spam)
- **Regression**: Predicting continuous values (e.g., house prices)

### Unsupervised Learning
Learning patterns from unlabeled data without explicit output labels.
- **Clustering**: Grouping similar data points together
- **Dimensionality Reduction**: Reducing the number of features while preserving information
- **Anomaly Detection**: Identifying outliers in data

### Reinforcement Learning
Learning through interaction with an environment to maximize cumulative rewards.
- **Model-free RL**: Learning without modeling the environment
- **Model-based RL**: Learning a model of the environment
- **Deep RL**: Combining deep learning with reinforcement learning

## Key Concepts

### The Machine Learning Pipeline

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load and prepare data
X, y = load_data()  # Features and labels

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Preprocess data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

### Bias-Variance Tradeoff

The bias-variance tradeoff is a fundamental concept in machine learning:

- **Bias**: Error from overly simplistic assumptions (underfitting)
- **Variance**: Error from sensitivity to small fluctuations in training data (overfitting)
- **Total Error** = Bias² + Variance + Irreducible Error

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, val_mean, label='Validation score')
    plt.fill_between(train_sizes, train_mean - train_std, 
                     train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, val_mean - val_std, 
                     val_mean + val_std, alpha=0.1)
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Learning Curve')
    plt.show()
```

### Cross-Validation

Cross-validation helps assess model performance and reduce overfitting:

```python
from sklearn.model_selection import cross_val_score, KFold

# K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Stratified K-Fold (maintains class distribution)
from sklearn.model_selection import StratifiedKFold
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skfold, scoring='accuracy')
```

### Regularization

Regularization techniques help prevent overfitting:

**L1 Regularization (Lasso)**: Encourages sparsity
```
Loss = MSE + λ * Σ|w_i|
```

**L2 Regularization (Ridge)**: Penalizes large weights
```
Loss = MSE + λ * Σw_i²
```

**Elastic Net**: Combines L1 and L2
```
Loss = MSE + λ₁ * Σ|w_i| + λ₂ * Σw_i²
```

```python
from sklearn.linear_model import Lasso, Ridge, ElasticNet

# L1 Regularization
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# L2 Regularization
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Elastic Net
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train, y_train)
```

### Feature Engineering

Feature engineering is crucial for model performance:

```python
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# One-hot encoding for categorical variables
encoder = OneHotEncoder(sparse=False, drop='first')
X_encoded = encoder.fit_transform(df[['category1', 'category2']])

# Feature scaling
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# Min-Max scaling (0 to 1)
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X)

# Robust scaling (uses median and IQR)
robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(X)
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")

# Randomized Search (faster for large parameter spaces)
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(10, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_distributions,
    n_iter=100,
    cv=5,
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)
```

## Evaluation Metrics

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ROC-AUC
y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

### Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)

# R² Score
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")
```

## Common Pitfalls and Best Practices

### 1. Data Leakage
Ensure test data doesn't leak into training:
```python
# WRONG: Scaling before splitting
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled)

# CORRECT: Fit scaler only on training data
X_train, X_test = train_test_split(X)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 2. Class Imbalance
Handle imbalanced datasets:
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# SMOTE (Synthetic Minority Over-sampling)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Class weights
model = LogisticRegression(class_weight='balanced')
```

### 3. Feature Selection
Remove irrelevant features:
```python
from sklearn.feature_selection import (
    SelectKBest, f_classif, RFE, SelectFromModel
)

# Univariate feature selection
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X_train, y_train)

# Recursive Feature Elimination
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=10)
X_rfe = rfe.fit_transform(X_train, y_train)

# Model-based selection
sfm = SelectFromModel(RandomForestClassifier(), threshold='median')
X_sfm = sfm.fit_transform(X_train, y_train)
```

## Resources

- **Books**:
  - "Pattern Recognition and Machine Learning" by Christopher Bishop
  - "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
  - "Deep Learning" by Goodfellow, Bengio, and Courville

- **Courses**:
  - Andrew Ng's Machine Learning Course (Coursera)
  - Fast.ai Practical Deep Learning
  - Stanford CS229: Machine Learning

- **Libraries**:
  - scikit-learn: Traditional ML algorithms
  - PyTorch: Deep learning framework
  - TensorFlow/Keras: Deep learning framework
  - XGBoost/LightGBM: Gradient boosting
  - Hugging Face: Transformers and NLP

## Quick Reference

### Model Selection Guide

| Problem Type | Recommended Models |
|--------------|-------------------|
| Linear separable data | Logistic Regression, SVM (linear) |
| Non-linear data | Random Forest, XGBoost, Neural Networks |
| High-dimensional data | Ridge/Lasso Regression, SVM |
| Small dataset | SVM, Naive Bayes, Linear Models |
| Large dataset | SGD-based models, Deep Learning |
| Interpretability needed | Linear Models, Decision Trees |
| Image data | CNNs, Vision Transformers |
| Text data | Transformers, RNNs, TF-IDF + Classical ML |
| Time series | RNNs, LSTMs, Transformers, ARIMA |
| Tabular data | XGBoost, LightGBM, Random Forest |

### Performance Optimization

```python
# Use efficient data structures
import pandas as pd
df = pd.read_csv('data.csv', dtype={'col1': 'category'})

# Parallel processing
from joblib import Parallel, delayed
results = Parallel(n_jobs=-1)(delayed(process)(x) for x in data)

# Batch processing for large datasets
def batch_process(data, batch_size=1000):
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        yield process_batch(batch)

# Use generators for memory efficiency
def data_generator(file_path):
    for chunk in pd.read_csv(file_path, chunksize=1000):
        yield chunk
```


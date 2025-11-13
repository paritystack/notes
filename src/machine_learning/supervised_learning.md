# Supervised Learning

Supervised learning is a type of machine learning where the model learns from labeled training data to make predictions on unseen data.

## Table of Contents

1. [Classification](#classification)
2. [Regression](#regression)
3. [Linear Models](#linear-models)
4. [Tree-Based Models](#tree-based-models)
5. [Support Vector Machines](#support-vector-machines)
6. [Ensemble Methods](#ensemble-methods)
7. [Naive Bayes](#naive-bayes)
8. [K-Nearest Neighbors](#k-nearest-neighbors)

## Classification

Classification predicts discrete class labels. The goal is to learn a decision boundary that separates different classes.

### Binary Classification

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### Multi-class Classification

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# Generate multi-class data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=5,
    n_informative=15,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Random Forest for multi-class
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# One-vs-Rest (OvR) strategy
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

ovr = OneVsRestClassifier(SVC(kernel='rbf'))
ovr.fit(X_train, y_train)

# One-vs-One (OvO) strategy
from sklearn.multiclass import OneVsOneClassifier
ovo = OneVsOneClassifier(SVC(kernel='rbf'))
ovo.fit(X_train, y_train)
```

### Imbalanced Classification

```python
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek
from sklearn.utils.class_weight import compute_class_weight

# SMOTE - Synthetic Minority Over-sampling Technique
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# ADASYN - Adaptive Synthetic Sampling
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

# Combined approach
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Custom threshold
y_proba = model.predict_proba(X_test)[:, 1]
threshold = 0.3  # Lower threshold for minority class
y_pred_custom = (y_proba >= threshold).astype(int)
```

## Regression

Regression predicts continuous values. The goal is to learn a function that maps inputs to outputs.

### Linear Regression

Mathematical formulation:
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

Where:
- y is the target variable
- x₁, x₂, ..., xₙ are features
- β₀, β₁, ..., βₙ are coefficients
- ε is the error term

```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Generate regression data
X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predictions
y_pred = lr.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Coefficients
print("\nCoefficients:", lr.coef_)
print("Intercept:", lr.intercept_)
```

### Polynomial Regression

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Create polynomial features
poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features.fit_transform(X_train)

# Using Pipeline
poly_model = Pipeline([
    ('poly_features', PolynomialFeatures(degree=3)),
    ('linear_regression', LinearRegression())
])

poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)

# Compare with linear
from sklearn.metrics import mean_squared_error
print(f"Linear RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"Polynomial RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_poly)):.4f}")
```

### Regularized Regression

**Ridge Regression (L2):**
```
Loss = Σ(y - ŷ)² + λΣβ²
```

**Lasso Regression (L1):**
```
Loss = Σ(y - ŷ)² + λΣ|β|
```

**Elastic Net:**
```
Loss = Σ(y - ŷ)² + λ₁Σ|β| + λ₂Σβ²
```

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LassoCV, RidgeCV

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# Lasso Regression (feature selection)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

# Check which features were selected by Lasso
feature_importance = np.abs(lasso.coef_)
selected_features = np.where(feature_importance > 0)[0]
print(f"Selected features: {selected_features}")

# Elastic Net
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train, y_train)

# Cross-validated alpha selection
ridge_cv = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
ridge_cv.fit(X_train, y_train)
print(f"Best alpha: {ridge_cv.alpha_}")

lasso_cv = LassoCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=5)
lasso_cv.fit(X_train, y_train)
print(f"Best alpha: {lasso_cv.alpha_}")
```

## Linear Models

### Logistic Regression

Binary classification using the sigmoid function:
```
P(y=1|x) = 1 / (1 + e^(-z))
where z = β₀ + β₁x₁ + ... + βₙxₙ
```

```python
from sklearn.linear_model import LogisticRegression

# Binary classification
log_reg = LogisticRegression(
    penalty='l2',
    C=1.0,  # Inverse of regularization strength
    solver='lbfgs',
    max_iter=1000
)
log_reg.fit(X_train, y_train)

# Get probabilities
probabilities = log_reg.predict_proba(X_test)
print("Class probabilities shape:", probabilities.shape)

# Decision boundary
decision_scores = log_reg.decision_function(X_test)
print("Decision scores shape:", decision_scores.shape)

# Multi-class logistic regression
multi_log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
multi_log_reg.fit(X_train, y_train)
```

### Perceptron

```python
from sklearn.linear_model import Perceptron

# Simple perceptron
perceptron = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
perceptron.fit(X_train, y_train)
y_pred = perceptron.predict(X_test)

# Custom perceptron implementation
class CustomPerceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Convert labels to -1 and 1
        y_ = np.where(y <= 0, -1, 1)
        
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = np.sign(linear_output)
                
                # Update weights if misclassified
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)

# Train custom perceptron
custom_perc = CustomPerceptron()
custom_perc.fit(X_train, y_train)
```

## Tree-Based Models

### Decision Trees

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import export_graphviz
import graphviz

# Classification tree
dt_clf = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    criterion='gini'  # or 'entropy'
)
dt_clf.fit(X_train, y_train)

# Regression tree
dt_reg = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5
)
dt_reg.fit(X_train, y_train)

# Feature importance
importances = dt_clf.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for i in range(min(10, len(indices))):
    print(f"{i+1}. Feature {indices[i]} ({importances[indices[i]]:.4f})")

# Visualize tree
dot_data = export_graphviz(
    dt_clf,
    out_file=None,
    feature_names=[f'feature_{i}' for i in range(X_train.shape[1])],
    class_names=['class_0', 'class_1'],
    filled=True,
    rounded=True
)
# graph = graphviz.Source(dot_data)
```

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Random Forest Classifier
rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',  # or 'log2', None
    bootstrap=True,
    oob_score=True,  # Out-of-bag score
    n_jobs=-1,
    random_state=42
)
rf_clf.fit(X_train, y_train)

# Out-of-bag score
print(f"OOB Score: {rf_clf.oob_score_:.4f}")

# Random Forest Regressor
rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    n_jobs=-1,
    random_state=42
)
rf_reg.fit(X_train, y_train)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': [f'feature_{i}' for i in range(X_train.shape[1])],
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(10))
```

### Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
gb_clf.fit(X_train, y_train)

# Gradient Boosting Regressor
gb_reg = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb_reg.fit(X_train, y_train)

# Feature importance
print("Feature importances:", gb_clf.feature_importances_)
```

### XGBoost

```python
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

# XGBoost Classifier
xgb_clf = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    random_state=42
)
xgb_clf.fit(X_train, y_train)

# XGBoost Regressor
xgb_reg = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_reg.fit(X_train, y_train)

# Using DMatrix for better performance
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'max_depth': 6,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc'
}

# Train with early stopping
evals = [(dtrain, 'train'), (dtest, 'test')]
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=False
)

# Predictions
y_pred_proba = bst.predict(dtest)
```

### LightGBM

```python
import lightgbm as lgb

# LightGBM Classifier
lgb_clf = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
lgb_clf.fit(X_train, y_train)

# Using Dataset for better performance
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5
}

# Train
gbm = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[test_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)
```

## Support Vector Machines

SVM finds the hyperplane that maximizes the margin between classes.

**Mathematical Formulation:**
```
Minimize: (1/2)||w||² + C·Σξᵢ
Subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0
```

### Linear SVM

```python
from sklearn.svm import SVC, LinearSVC

# Linear SVM
linear_svm = LinearSVC(C=1.0, max_iter=10000)
linear_svm.fit(X_train, y_train)

# SVC with linear kernel
svc_linear = SVC(kernel='linear', C=1.0)
svc_linear.fit(X_train, y_train)
```

### Non-linear SVM with Kernels

```python
# RBF (Radial Basis Function) kernel
svc_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svc_rbf.fit(X_train, y_train)

# Polynomial kernel
svc_poly = SVC(kernel='poly', degree=3, C=1.0)
svc_poly.fit(X_train, y_train)

# Sigmoid kernel
svc_sigmoid = SVC(kernel='sigmoid', C=1.0)
svc_sigmoid.fit(X_train, y_train)

# Custom kernel
def custom_kernel(X, Y):
    return np.dot(X, Y.T)

svc_custom = SVC(kernel=custom_kernel)
svc_custom.fit(X_train, y_train)
```

### SVM for Regression

```python
from sklearn.svm import SVR

# Support Vector Regression
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)

# Linear SVR
from sklearn.svm import LinearSVR
linear_svr = LinearSVR(epsilon=0.1, C=1.0)
linear_svr.fit(X_train, y_train)
```

## Ensemble Methods

### Bagging

```python
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier

# Bagging with decision trees
bagging_clf = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)
bagging_clf.fit(X_train, y_train)
print(f"OOB Score: {bagging_clf.oob_score_:.4f}")
```

### Boosting

**AdaBoost:**
```python
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

# AdaBoost Classifier
ada_clf = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)
ada_clf.fit(X_train, y_train)

# AdaBoost Regressor
ada_reg = AdaBoostRegressor(
    base_estimator=DecisionTreeRegressor(max_depth=3),
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)
ada_reg.fit(X_train, y_train)
```

### Stacking

```python
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Define base models
base_models = [
    ('lr', LogisticRegression()),
    ('dt', DecisionTreeClassifier()),
    ('svc', SVC(probability=True)),
    ('nb', GaussianNB())
]

# Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5
)
stacking_clf.fit(X_train, y_train)

# Stacking Regressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor

reg_base_models = [
    ('ridge', Ridge()),
    ('lasso', Lasso()),
    ('dt', DecisionTreeRegressor())
]

stacking_reg = StackingRegressor(
    estimators=reg_base_models,
    final_estimator=Ridge(),
    cv=5
)
stacking_reg.fit(X_train, y_train)
```

### Voting

```python
from sklearn.ensemble import VotingClassifier, VotingRegressor

# Hard voting
voting_clf_hard = VotingClassifier(
    estimators=base_models,
    voting='hard'
)
voting_clf_hard.fit(X_train, y_train)

# Soft voting (uses predicted probabilities)
voting_clf_soft = VotingClassifier(
    estimators=base_models,
    voting='soft'
)
voting_clf_soft.fit(X_train, y_train)

# Voting Regressor
voting_reg = VotingRegressor(estimators=reg_base_models)
voting_reg.fit(X_train, y_train)
```

## Naive Bayes

Based on Bayes' theorem with the "naive" assumption of feature independence:
```
P(y|x₁,...,xₙ) = P(y)·P(x₁,...,xₙ|y) / P(x₁,...,xₙ)
```

### Gaussian Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB

# Gaussian NB (assumes features follow normal distribution)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
y_proba = gnb.predict_proba(X_test)
```

### Multinomial Naive Bayes

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Example with text data
texts = ["I love this", "This is bad", "Great product", "Terrible experience"]
labels = [1, 0, 1, 0]

vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(texts)

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_text, labels)
```

### Bernoulli Naive Bayes

```python
from sklearn.naive_bayes import BernoulliNB

# Bernoulli NB (for binary/boolean features)
bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_train, y_train)
```

## K-Nearest Neighbors

KNN is a non-parametric method that classifies based on the k nearest training examples.

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# KNN Classifier
knn_clf = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',  # or 'distance'
    algorithm='auto',  # 'ball_tree', 'kd_tree', 'brute'
    metric='minkowski',
    p=2  # p=2 for Euclidean, p=1 for Manhattan
)
knn_clf.fit(X_train, y_train)

# Distance-weighted KNN
knn_weighted = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_weighted.fit(X_train, y_train)

# KNN Regressor
knn_reg = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn_reg.fit(X_train, y_train)

# Find optimal k
from sklearn.model_selection import cross_val_score

k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

optimal_k = k_range[np.argmax(k_scores)]
print(f"Optimal k: {optimal_k}")
```

## Model Comparison

```python
from sklearn.model_selection import cross_validate
import pandas as pd

# Define models to compare
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'XGBoost': XGBClassifier()
}

# Compare models
results = []
for name, model in models.items():
    cv_results = cross_validate(
        model, X_train, y_train,
        cv=5,
        scoring=['accuracy', 'precision', 'recall', 'f1'],
        return_train_score=True
    )
    
    results.append({
        'Model': name,
        'Train Accuracy': cv_results['train_accuracy'].mean(),
        'Test Accuracy': cv_results['test_accuracy'].mean(),
        'Precision': cv_results['test_precision'].mean(),
        'Recall': cv_results['test_recall'].mean(),
        'F1': cv_results['test_f1'].mean()
    })

# Display results
comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)
print(comparison_df)
```

## Practical Tips

### 1. Data Preprocessing
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# Handle missing values
imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'
X_imputed = imputer.fit_transform(X)

# Feature scaling (important for SVM, KNN, Neural Networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 2. Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE

# Univariate selection
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X_train, y_train)

# Recursive Feature Elimination
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10)
X_rfe = rfe.fit_transform(X_train, y_train)
```

### 3. Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Create pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', LogisticRegression())
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

### 4. Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid search
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
```

## Resources

- scikit-learn documentation: https://scikit-learn.org/
- XGBoost documentation: https://xgboost.readthedocs.io/
- LightGBM documentation: https://lightgbm.readthedocs.io/
- "Introduction to Statistical Learning" by James et al.
- "Pattern Recognition and Machine Learning" by Bishop


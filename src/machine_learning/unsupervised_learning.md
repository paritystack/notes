# Unsupervised Learning

Unsupervised learning discovers hidden patterns in data without labeled outputs.

## Table of Contents

1. [Clustering](#clustering)
2. [Dimensionality Reduction](#dimensionality-reduction)
3. [Anomaly Detection](#anomaly-detection)
4. [Density Estimation](#density-estimation)
5. [Association Rules](#association-rules)

## Clustering

Clustering groups similar data points together without predefined labels.

### K-Means Clustering

K-Means partitions data into k clusters by minimizing within-cluster variance.

**Algorithm:**
1. Initialize k centroids randomly
2. Assign each point to nearest centroid
3. Update centroids as mean of assigned points
4. Repeat steps 2-3 until convergence

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X)

# Predictions
y_pred = kmeans.predict(X)
centers = kmeans.cluster_centers_

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, edgecolors='black')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Cluster characteristics
print(f"Cluster centers:\n{centers}")
print(f"Inertia (sum of squared distances): {kmeans.inertia_:.2f}")
```

### Choosing Optimal K

**Elbow Method:**
```python
from sklearn.metrics import silhouette_score

# Elbow method
inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X, kmeans.labels_))

# Plot elbow curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(K_range, inertias, 'bo-')
ax1.set_xlabel('Number of clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')
ax1.grid(True)

ax2.plot(K_range, silhouettes, 'ro-')
ax2.set_xlabel('Number of clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis')
ax2.grid(True)

plt.tight_layout()
plt.show()
```

### K-Means++

Improved initialization for K-Means:
```python
# K-Means++ (default in scikit-learn)
kmeans_plus = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans_plus.fit(X)

# Mini-batch K-Means (faster for large datasets)
from sklearn.cluster import MiniBatchKMeans

mini_kmeans = MiniBatchKMeans(n_clusters=4, random_state=42, batch_size=100)
mini_kmeans.fit(X)
```

### Hierarchical Clustering

Builds a tree of clusters (dendrogram).

**Agglomerative (Bottom-up):**
```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# Agglomerative clustering
agg_clustering = AgglomerativeClustering(
    n_clusters=4,
    linkage='ward'  # 'complete', 'average', 'single'
)
y_pred_agg = agg_clustering.fit_predict(X)

# Create dendrogram
Z = linkage(X, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Different linkage methods
linkage_methods = ['ward', 'complete', 'average', 'single']
for method in linkage_methods:
    agg = AgglomerativeClustering(n_clusters=4, linkage=method)
    labels = agg.fit_predict(X)
    print(f"{method.capitalize()} linkage - Silhouette: {silhouette_score(X, labels):.3f}")
```

### DBSCAN

Density-Based Spatial Clustering finds core samples of high density.

```python
from sklearn.cluster import DBSCAN

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_pred_dbscan = dbscan.fit_predict(X)

# Number of clusters (excluding noise points labeled as -1)
n_clusters = len(set(y_pred_dbscan)) - (1 if -1 in y_pred_dbscan else 0)
n_noise = list(y_pred_dbscan).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")

# Visualization
plt.figure(figsize=(10, 6))
unique_labels = set(y_pred_dbscan)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]  # Black for noise
    
    class_member_mask = (y_pred_dbscan == k)
    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title(f'DBSCAN Clustering\n{n_clusters} clusters, {n_noise} noise points')
plt.show()

# Grid search for optimal parameters
from sklearn.model_selection import ParameterGrid

param_grid = {
    'eps': [0.3, 0.5, 0.7, 1.0],
    'min_samples': [3, 5, 10]
}

best_score = -1
best_params = None

for params in ParameterGrid(param_grid):
    dbscan = DBSCAN(**params)
    labels = dbscan.fit_predict(X)
    
    # Skip if all points are noise or only one cluster
    if len(set(labels)) <= 1:
        continue
    
    score = silhouette_score(X, labels)
    if score > best_score:
        best_score = score
        best_params = params

print(f"Best parameters: {best_params}")
print(f"Best silhouette score: {best_score:.3f}")
```

### HDBSCAN

Hierarchical DBSCAN with better parameter selection:
```python
# pip install hdbscan
import hdbscan

# HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
y_pred_hdbscan = clusterer.fit_predict(X)

# Cluster probabilities
probabilities = clusterer.probabilities_

print(f"Number of clusters: {len(set(y_pred_hdbscan)) - (1 if -1 in y_pred_hdbscan else 0)}")
print(f"Noise points: {list(y_pred_hdbscan).count(-1)}")
```

### Gaussian Mixture Models

GMM assumes data is generated from a mixture of Gaussian distributions.

```python
from sklearn.mixture import GaussianMixture

# Gaussian Mixture Model
gmm = GaussianMixture(
    n_components=4,
    covariance_type='full',  # 'tied', 'diag', 'spherical'
    random_state=42
)
gmm.fit(X)

# Predictions (hard clustering)
y_pred_gmm = gmm.predict(X)

# Soft clustering (probabilities)
probabilities = gmm.predict_proba(X)
print("Shape of probabilities:", probabilities.shape)

# Model parameters
print(f"Means:\n{gmm.means_}")
print(f"Covariances shape: {gmm.covariances_.shape}")
print(f"Weights: {gmm.weights_}")

# Bayesian Information Criterion (BIC) for model selection
n_components_range = range(2, 11)
bic_scores = []
aic_scores = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))
    aic_scores.append(gmm.aic(X))

plt.figure(figsize=(10, 6))
plt.plot(n_components_range, bic_scores, 'bo-', label='BIC')
plt.plot(n_components_range, aic_scores, 'rs-', label='AIC')
plt.xlabel('Number of components')
plt.ylabel('Information Criterion')
plt.title('GMM Model Selection')
plt.legend()
plt.grid(True)
plt.show()

optimal_components = n_components_range[np.argmin(bic_scores)]
print(f"Optimal number of components: {optimal_components}")
```

### Mean Shift

Finds clusters by locating peaks in density.

```python
from sklearn.cluster import MeanShift, estimate_bandwidth

# Estimate bandwidth
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

# Mean Shift clustering
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
y_pred_ms = ms.labels_
cluster_centers = ms.cluster_centers_

n_clusters = len(np.unique(y_pred_ms))
print(f"Number of clusters: {n_clusters}")
```

### Spectral Clustering

Uses eigenvalues of similarity matrix for clustering.

```python
from sklearn.cluster import SpectralClustering

# Spectral clustering
spectral = SpectralClustering(
    n_clusters=4,
    affinity='rbf',  # 'nearest_neighbors', 'precomputed'
    assign_labels='discretize',  # 'kmeans'
    random_state=42
)
y_pred_spectral = spectral.fit_predict(X)

# Custom affinity matrix
from sklearn.metrics.pairwise import rbf_kernel
affinity_matrix = rbf_kernel(X, gamma=1.0)
spectral_custom = SpectralClustering(n_clusters=4, affinity='precomputed')
y_pred_spectral_custom = spectral_custom.fit_predict(affinity_matrix)
```

## Dimensionality Reduction

Reducing the number of features while preserving important information.

### Principal Component Analysis (PCA)

PCA finds orthogonal directions of maximum variance.

**Mathematical Formulation:**
```
Maximize: Var(Xw) subject to ||w|| = 1
```

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# Load high-dimensional data
digits = load_digits()
X = digits.data  # 64 features (8x8 images)
y = digits.target

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Explained variance
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# Visualization
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(scatter)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA of Digits Dataset')
plt.show()

# Determine number of components
pca_full = PCA()
pca_full.fit(X)

# Cumulative explained variance
cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.argmax(cumsum_var >= 0.95) + 1

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.axvline(x=n_components_95, color='g', linestyle='--', 
            label=f'{n_components_95} components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.legend()
plt.grid(True)
plt.show()

print(f"Components needed for 95% variance: {n_components_95}")

# Incremental PCA for large datasets
from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(n_components=10, batch_size=100)
X_ipca = ipca.fit_transform(X)

# Kernel PCA for non-linear dimensionality reduction
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.04)
X_kpca = kpca.fit_transform(X)
```

### t-SNE

t-Distributed Stochastic Neighbor Embedding for visualization.

```python
from sklearn.manifold import TSNE

# t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter=1000,
    random_state=42
)
X_tsne = tsne.fit_transform(X)

# Visualization
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(scatter)
plt.title('t-SNE of Digits Dataset')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()

# Try different perplexities
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
perplexities = [5, 30, 50, 100]

for ax, perplexity in zip(axes.ravel(), perplexities):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, 
                        cmap='tab10', alpha=0.6)
    ax.set_title(f'Perplexity = {perplexity}')

plt.tight_layout()
plt.show()
```

### UMAP

Uniform Manifold Approximation and Projection (faster than t-SNE).

```python
# pip install umap-learn
import umap

# UMAP
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
    random_state=42
)
X_umap = reducer.fit_transform(X)

# Visualization
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(scatter)
plt.title('UMAP of Digits Dataset')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()

# Compare PCA, t-SNE, and UMAP
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

methods = [
    ('PCA', X_pca),
    ('t-SNE', X_tsne),
    ('UMAP', X_umap)
]

for ax, (name, X_reduced) in zip(axes, methods):
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, 
                        cmap='tab10', alpha=0.6)
    ax.set_title(name)
    ax.set_xlabel(f'{name} 1')
    ax.set_ylabel(f'{name} 2')

plt.tight_layout()
plt.show()
```

### Linear Discriminant Analysis (LDA)

Supervised dimensionality reduction that maximizes class separability.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LDA (requires labels)
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Explained variance ratio
print(f"Explained variance ratio: {lda.explained_variance_ratio_}")

# Visualization
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(scatter)
plt.xlabel(f'LD1 ({lda.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'LD2 ({lda.explained_variance_ratio_[1]:.2%})')
plt.title('LDA of Digits Dataset')
plt.show()
```

### Autoencoders

Neural network-based dimensionality reduction.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

# Prepare data
X_normalized = (X - X.min()) / (X.max() - X.min())
X_tensor = torch.FloatTensor(X_normalized)
dataset = TensorDataset(X_tensor, X_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
input_dim = X.shape[1]
encoding_dim = 2
model = Autoencoder(input_dim, encoding_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
n_epochs = 50
for epoch in range(n_epochs):
    total_loss = 0
    for batch_x, _ in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(dataloader):.4f}')

# Get encoded representations
model.eval()
with torch.no_grad():
    X_encoded = model.encode(X_tensor).numpy()

# Visualization
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_encoded[:, 0], X_encoded[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(scatter)
plt.title('Autoencoder Dimensionality Reduction')
plt.xlabel('Encoded Dimension 1')
plt.ylabel('Encoded Dimension 2')
plt.show()
```

### Non-negative Matrix Factorization (NMF)

Decomposes data into non-negative components.

```python
from sklearn.decomposition import NMF

# NMF (requires non-negative data)
X_nonneg = X - X.min() + 1e-10
nmf = NMF(n_components=10, init='random', random_state=42, max_iter=500)
W = nmf.fit_transform(X_nonneg)  # Coefficient matrix
H = nmf.components_  # Component matrix

print(f"Reconstruction error: {nmf.reconstruction_err_:.2f}")
print(f"W shape: {W.shape}, H shape: {H.shape}")

# Visualize components
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(H[i].reshape(8, 8), cmap='gray')
    ax.set_title(f'Component {i+1}')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

### Truncated SVD

Similar to PCA but works with sparse matrices.

```python
from sklearn.decomposition import TruncatedSVD

# Truncated SVD
svd = TruncatedSVD(n_components=10, random_state=42)
X_svd = svd.fit_transform(X)

print(f"Explained variance ratio: {svd.explained_variance_ratio_}")
print(f"Total explained variance: {svd.explained_variance_ratio_.sum():.3f}")
```

## Anomaly Detection

Identifying unusual patterns that don't conform to expected behavior.

### Isolation Forest

```python
from sklearn.ensemble import IsolationForest

# Isolation Forest
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.1,  # Expected proportion of outliers
    random_state=42
)
y_pred_outliers = iso_forest.fit_predict(X)

# -1 for outliers, 1 for inliers
n_outliers = (y_pred_outliers == -1).sum()
print(f"Number of outliers detected: {n_outliers}")

# Anomaly scores
anomaly_scores = iso_forest.score_samples(X)

# Visualization
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=anomaly_scores, cmap='RdYlGn')
plt.colorbar(scatter, label='Anomaly Score')
plt.title('Isolation Forest Anomaly Scores')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
```

### Local Outlier Factor (LOF)

```python
from sklearn.neighbors import LocalOutlierFactor

# Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred_lof = lof.fit_predict(X)

# Negative outlier factor (lower values = more anomalous)
outlier_scores = lof.negative_outlier_factor_

n_outliers = (y_pred_lof == -1).sum()
print(f"Number of outliers detected: {n_outliers}")
```

### One-Class SVM

```python
from sklearn.svm import OneClassSVM

# One-Class SVM
oc_svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='auto')
y_pred_oc = oc_svm.fit_predict(X)

n_outliers = (y_pred_oc == -1).sum()
print(f"Number of outliers detected: {n_outliers}")
```

### Elliptic Envelope

```python
from sklearn.covariance import EllipticEnvelope

# Elliptic Envelope (assumes Gaussian distribution)
elliptic = EllipticEnvelope(contamination=0.1, random_state=42)
y_pred_elliptic = elliptic.fit_predict(X)

n_outliers = (y_pred_elliptic == -1).sum()
print(f"Number of outliers detected: {n_outliers}")
```

## Density Estimation

Estimating the probability density function of data.

### Kernel Density Estimation

```python
from sklearn.neighbors import KernelDensity

# Kernel Density Estimation
kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
kde.fit(X)

# Score samples (log-likelihood)
log_density = kde.score_samples(X)

# Sample from the learned distribution
samples = kde.sample(100, random_state=42)

# Visualization (for 2D data)
if X.shape[1] == 2:
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
                         np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
    Z = np.exp(kde.score_samples(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, levels=20, cmap='viridis', alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c='red', alpha=0.3, s=10)
    plt.colorbar(label='Density')
    plt.title('Kernel Density Estimation')
    plt.show()
```

## Association Rules

Finding interesting relationships between variables.

### Apriori Algorithm

```python
# pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Example transaction data
transactions = [
    ['milk', 'bread', 'butter'],
    ['milk', 'bread'],
    ['milk', 'butter'],
    ['bread', 'butter'],
    ['milk', 'bread', 'butter', 'cheese'],
    ['milk', 'cheese'],
    ['bread', 'cheese']
]

# Convert to one-hot encoded DataFrame
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

# Find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
print("Frequent Itemsets:")
print(frequent_itemsets)

# Generate association rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Filter interesting rules
interesting_rules = rules[(rules['lift'] > 1) & (rules['confidence'] > 0.6)]
print("\nInteresting Rules:")
print(interesting_rules)
```

## Clustering Evaluation Metrics

```python
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, 
    calinski_harabasz_score, adjusted_rand_score
)

# Silhouette Score (higher is better, range: [-1, 1])
silhouette = silhouette_score(X, y_pred)

# Davies-Bouldin Index (lower is better)
davies_bouldin = davies_bouldin_score(X, y_pred)

# Calinski-Harabasz Index (higher is better)
calinski_harabasz = calinski_harabasz_score(X, y_pred)

# Adjusted Rand Index (if true labels available)
ari = adjusted_rand_score(y_true, y_pred)

print(f"Silhouette Score: {silhouette:.3f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.3f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz:.3f}")
print(f"Adjusted Rand Index: {ari:.3f}")

# Silhouette analysis per sample
from sklearn.metrics import silhouette_samples

silhouette_vals = silhouette_samples(X, y_pred)

# Visualize silhouette scores
fig, ax = plt.subplots(figsize=(10, 6))
y_lower = 10

for i in range(len(set(y_pred))):
    cluster_silhouette_vals = silhouette_vals[y_pred == i]
    cluster_silhouette_vals.sort()
    
    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i
    
    ax.fill_betweenx(np.arange(y_lower, y_upper),
                     0, cluster_silhouette_vals,
                     alpha=0.7)
    
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

ax.set_xlabel("Silhouette Coefficient")
ax.set_ylabel("Cluster")
ax.axvline(x=silhouette, color="red", linestyle="--")
ax.set_title("Silhouette Analysis")
plt.show()
```

## Practical Tips

### 1. Feature Scaling
```python
# Always scale features for distance-based methods
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 2. Handling High-Dimensional Data
```python
# Apply dimensionality reduction before clustering
pca = PCA(n_components=0.95)  # Keep 95% variance
X_reduced = pca.fit_transform(X_scaled)
kmeans = KMeans(n_clusters=4)
kmeans.fit(X_reduced)
```

### 3. Visualizing Clusters
```python
def plot_clusters_3d(X, labels, title='3D Cluster Visualization'):
    from mpl_toolkits.mplot3d import Axes3D
    
    # Reduce to 3D if needed
    if X.shape[1] > 3:
        pca = PCA(n_components=3)
        X = pca.fit_transform(X)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis')
    ax.set_title(title)
    plt.colorbar(scatter)
    plt.show()
```

## Resources

- scikit-learn documentation: https://scikit-learn.org/
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "Introduction to Data Mining" by Tan, Steinbach, Kumar
- UMAP documentation: https://umap-learn.readthedocs.io/


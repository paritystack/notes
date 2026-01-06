# Vector Databases

> **Domain:** AI Infrastructure, Databases
> **Key Concepts:** Embeddings, HNSW, IVF, Cosine Similarity, ANN Search

**Vector Databases** are specialized storage systems designed to handle high-dimensional vector embeddings. Unlike relational databases (rows/columns) or document stores (JSON), vector DBs are optimized for **Approximate Nearest Neighbor (ANN)** search—finding data points that are semantically similar to a query vector, not just exact matches.

---

## 1. Why Vector Databases?

Traditional databases use B-Trees or Hash Indexes for exact keyword matching (`WHERE id = 5`). These fail with unstructured data (images, audio, text) where "similarity" is mathematical, not literal.

*   **The Embedding:** A vector is a dense array of floating-point numbers (e.g., `[0.1, -0.5, 0.8...]`) representing the semantic meaning of content.
*   **The Query:** "King" - "Man" + "Woman" ≈ "Queen".
*   **The Scale:** Searching for the nearest neighbor in a dataset of 100M vectors via brute force is $O(N)$. Vector DBs use indices to make this $O(\log N)$.

---

## 2. Distance Metrics

The definition of "similarity" depends on the math used to compare vectors.

1.  **Cosine Similarity:** Measures the cosine of the angle between two vectors.
    *   *Formula:* $A \cdot B / (||A|| \cdot ||B||)$
    *   *Use Case:* NLP, text semantic similarity. Ignores vector magnitude (length), focusing only on direction/orientation.
2.  **Euclidean Distance (L2):** Measures the straight-line distance between two points.
    *   *Formula:* $\sqrt{\sum(A_i - B_i)^2}$
    *   *Use Case:* Computer Vision, where magnitude matters.
3.  **Dot Product:** Projection of one vector onto another.
    *   *Use Case:* Recommendation systems (Matrix Factorization). Same as Cosine if vectors are normalized.

---

## 3. Indexing Algorithms (The "Secret Sauce")

To search fast, we trade accuracy for speed (Approximate Nearest Neighbor).

### 3.1. HNSW (Hierarchical Navigable Small World)
The industry standard.
*   **Structure:** A multi-layered graph. Top layers have few "long-range" links (like highways). Bottom layers have dense "short-range" links (local roads).
*   **Search:** Start at the top layer, greedy jump to the closest node, drop down a layer, repeat until the bottom layer.
*   **Pros:** Extremely fast query, high recall.
*   **Cons:** High memory usage (graph is stored in RAM), slow build time.

### 3.2. IVF (Inverted File Index)
*   **Concept:** Cluster the vector space into $N$ Voronoi cells (centroids).
*   **Indexing:** Assign every vector to its nearest centroid.
*   **Search:** Determine which centroid the query falls into, and brute-force search *only* the vectors in that cell (and maybe neighbor cells).
*   **Pros:** Low memory (can rely on disk), fast.
*   **Cons:** Lower recall if the "true" neighbor is just across the boundary of a skipped cell.

### 3.3. PQ (Product Quantization)
A compression technique often combined with IVF.
*   **Concept:** Split high-dimensional vectors (e.g., 1024-d) into smaller sub-vectors (e.g., 8 chunks of 128-d) and cluster each chunk separately.
*   **Result:** Reduces memory footprint by 90%+, allowing massive datasets to fit in RAM.

---

## 4. Metadata Filtering

In real apps, you rarely search *just* vectors. You ask: *"Find closest documents to 'Apple' that were published in 2024."*

*   **Post-Filtering:** Search vectors first, then filter results by year.
    *   *Risk:* You retrieve 100 docs, filter them, and end up with 0 because all 100 were from 2023.
*   **Pre-Filtering:** Filter the dataset by year first, then search vectors.
    *   *Risk:* Brutally slow if the filtered set is large but not indexed.
*   **Single-Stage Filtering (Hybrid):** Modern DBs (Milvus, Weaviate) traverse the HNSW graph but only consider nodes that match the metadata filter mask during the traversal.

---

## 5. Popular Tools

| Database | Type | Best For |
| :--- | :--- | :--- |
| **Pinecone** | Managed SaaS | Developers who want zero ops. |
| **Milvus** | Open Source / Go | Massive scale (billions of vectors), cloud-native. |
| **Weaviate** | Open Source / Go | Hybrid search, built-in embedding modules. |
| **pgvector** | Postgres Ext | Simple apps, transactional consistency (ACID) with vector search. |
| **Chroma** | Open Source / Python | Local development, simple RAG apps. |

## 6. Conclusion

Vector Databases are the long-term memory of AI applications. Choosing the right one depends on your scale (10k vs 100M vectors), latency requirements, and whether you need complex metadata filtering.

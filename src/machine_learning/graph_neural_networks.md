# Graph Neural Networks

How neural networks operate on **graph-structured** data — nodes connected by edges, with no
fixed grid or sequence order. GNNs generalize the idea behind [convolution](convolution.md)
(aggregate a local neighborhood) and [attention](attention.md) (weight neighbors by relevance)
to arbitrary connectivity, learning node, edge, and whole-graph representations by passing
messages along edges.

## Table of Contents

1. [Overview](#overview)
2. [Why Graphs Need Their Own Architecture](#why-graphs-need-their-own-architecture)
3. [Message Passing](#message-passing)
4. [GCN](#gcn)
5. [GraphSAGE](#graphsage)
6. [GAT](#gat)
7. [Readout & Graph-Level Tasks](#readout--graph-level-tasks)
8. [Over-Smoothing & Over-Squashing](#over-smoothing--over-squashing)
9. [Comparison](#comparison)
10. [Where this connects](#where-this-connects)
11. [Pitfalls](#pitfalls)

## Overview

A graph `G = (V, E)` is a set of **nodes** `V` connected by **edges** `E`, each optionally
carrying a feature vector. Unlike images (a fixed pixel grid) or text (a 1-D sequence), graphs
have **no canonical ordering** and **variable-size neighborhoods** — a molecule, a social
network, a citation graph, or a knowledge base. A GNN learns representations that respect this
structure and are **permutation-equivariant**: relabeling the nodes relabels the output the
same way, without changing it.

```
   inputs                  GNN                      outputs (pick a level)
   ┌─────────┐                                      ┌──────────────────────┐
   │ node    │   message passing over edges:        │ node classification  │
   │ features│   each node mixes in its neighbors → │ link prediction      │
   │ + edges │   repeat L layers → L-hop context    │ graph classification │
   └─────────┘                                      └──────────────────────┘
```

The three task levels: **node-level** (classify a user, label a protein residue), **edge-level**
(predict a missing link, recommend), and **graph-level** (is this molecule toxic?).

## Why Graphs Need Their Own Architecture

You cannot just flatten a graph and feed it to an MLP or CNN:

```
  CNN          fixed 3×3 grid neighborhood, shared kernel       → needs a grid
  RNN/Transformer  ordered sequence                              → needs an order
  MLP on flattened adjacency  → not permutation-invariant, O(V²) params, no generalization
```

A graph has neither a grid nor an order, and the same graph can be written with its nodes in
any order. The architecture must therefore (a) handle **arbitrary, variable-degree
neighborhoods** and (b) be **invariant to node permutation**. Message passing delivers both.

## Message Passing

Almost every GNN is an instance of the **message-passing** framework. At each layer, every node
**aggregates** transformed messages from its neighbors and **updates** its own state:

```
   for each node v, at layer k:
                                  ┌── neighbor u₁ ─ h_u₁ ─┐
       h_v^(k) = UPDATE( h_v^(k-1),  AGGREGATE({ h_u : u ∈ N(v) }) )
                                  └── neighbor u₂ ─ h_u₂ ─┘
                                       (sum / mean / max / attention)
```

- **AGGREGATE** must be **permutation-invariant** (sum, mean, max, or attention-weighted sum) —
  this is what makes the GNN order-independent.
- **UPDATE** combines the node's previous state with the aggregated message (a linear layer +
  nonlinearity, sometimes a GRU).

Stacking `L` layers gives each node a **receptive field of `L` hops**: after one layer a node
sees its immediate neighbors, after two it sees neighbors-of-neighbors, and so on. The specific
GNN variants differ only in *how* they aggregate.

## GCN

The **Graph Convolutional Network** (Kipf & Welling, 2017) uses a normalized-mean aggregation —
the canonical baseline:

```
   H^(k) = σ( D̂^(-½) Â D̂^(-½) H^(k-1) W^(k) )

   Â = A + I       add self-loops (a node includes itself)
   D̂              degree matrix of Â  → symmetric normalization
   W^(k)           learned weight matrix for layer k
```

In message-passing terms: each node averages its neighbors' features (degree-normalized so
high-degree hubs don't dominate), applies a shared linear map, and a nonlinearity. Simple,
fast, and a strong baseline — but **transductive** in its basic form (it operates on one fixed
graph) and treats all neighbors with equal, structurally-fixed weight.

## GraphSAGE

**GraphSAGE** (Hamilton et al., 2017) makes GNNs **inductive** — able to generalize to nodes or
graphs unseen at training time — via two changes:

```
  1. neighbor SAMPLING   sample a fixed number of neighbors per node (not all)
                         → bounds compute, scales to huge graphs
  2. generalized AGGREGATE  mean / LSTM / max-pool, then CONCAT with self, then project
                         h_v = σ( W · CONCAT(h_v,  AGG({h_u})) )
```

Because the aggregator and weights are shared across all nodes and don't depend on the global
graph structure, a trained GraphSAGE can embed a brand-new node from its features and local
neighborhood — essential for production graphs that grow (new users, new items).

## GAT

The **Graph Attention Network** (Veličković et al., 2018) replaces fixed averaging with
**learned attention weights** over neighbors — the same idea as [transformer](transformers.md)
[attention](attention.md), but restricted to the graph's edges:

```
   α_vu = softmax_u( LeakyReLU( aᵀ [W h_v ‖ W h_u] ) )    attention over neighbors u ∈ N(v)
   h_v' = σ( Σ_u  α_vu · W h_u )                           weighted neighbor sum
```

Each node decides *how much* to attend to each neighbor, and **multi-head attention** stabilizes
training (just like in transformers). GAT is effectively a transformer whose attention mask is
the adjacency matrix — which is the cleanest way to see the connection: a standard transformer
is a GNN on a **fully-connected** graph with [positional encodings](positional_encoding.md)
standing in for missing structure.

## Readout & Graph-Level Tasks

Node-level outputs come straight from the final node embeddings. For **graph-level** tasks you
need a single vector per graph, produced by a permutation-invariant **readout** (pooling):

```
   h_G = READOUT({ h_v : v ∈ V })      sum / mean / max  → graph embedding → classifier

   sum    sensitive to graph size, most expressive
   mean   size-invariant
   max    captures salient nodes
```

More expressive **hierarchical pooling** (DiffPool, Top-K) coarsens the graph in stages,
learning to merge nodes into clusters. The expressive power of message-passing GNNs is bounded
by the **Weisfeiler-Lehman** graph-isomorphism test; **GIN** (Graph Isomorphism Network) uses a
sum aggregator plus an MLP to match that bound exactly.

## Over-Smoothing & Over-Squashing

Two failure modes limit how deep GNNs can go — the reason most GNNs are only 2–4 layers, in
sharp contrast to very deep CNNs/transformers:

```
  OVER-SMOOTHING   stack too many layers → every node averages over the whole graph →
                   all node embeddings converge to the SAME vector → features become useless

  OVER-SQUASHING   information from an exponentially growing receptive field is crushed
                   into a fixed-size vector → distant signals can't pass through bottlenecks
```

Mitigations borrowed from the rest of deep learning: **residual/skip connections**,
[normalization](normalization.md) (PairNorm), [dropout](deep_learning.md) on edges (DropEdge),
and **jumping-knowledge** connections that combine representations from multiple layers. Graph
**rewiring** addresses over-squashing by adding edges that shorten long-range paths.

## Comparison

| Model | Aggregation | Inductive? | Neighbor weighting | Note |
|---|---|---|---|---|
| **GCN** | degree-normalized mean | no (basic form) | fixed by structure | strong simple baseline |
| **GraphSAGE** | mean / max-pool / LSTM + concat | **yes** | uniform over sample | sampling → scales to huge graphs |
| **GAT** | attention-weighted sum | yes | **learned** per edge | attention restricted to edges |
| **GIN** | sum + MLP | yes | fixed | maximally expressive (WL bound) |

## Where this connects

- [Convolution](convolution.md) — a CNN is a GNN on a regular grid; graph convolution generalizes
  the local-neighborhood aggregation to arbitrary connectivity
- [Attention](attention.md) and [transformers](transformers.md) — GAT is edge-masked attention; a
  transformer is a GNN on a fully-connected graph
- [Positional encodings](positional_encoding.md) — graph/Laplacian positional encodings give
  transformers the structure GNNs get from edges
- [Normalization](normalization.md) — PairNorm and friends fight over-smoothing
- [Neural networks](neural_networks.md) and [deep learning](deep_learning.md) — the MLP update
  steps, residuals, and dropout reused inside message passing
- [Embeddings & reranking](../ai/embeddings.md) and [GraphRAG](../ai/graphrag.md) — node/graph
  embeddings used downstream for retrieval and recommendation

## Pitfalls

- **Too many layers → over-smoothing** — node embeddings collapse to one vector; keep GNNs
  shallow (2–4 layers) or add residual/jumping-knowledge connections.
- **Over-squashing on long-range tasks** — message passing can't move information across
  bottlenecks; consider rewiring or a graph transformer.
- **Transductive vs. inductive mismatch** — basic GCN can't embed unseen nodes; use GraphSAGE/GAT
  for graphs that grow.
- **Full-neighborhood aggregation on huge graphs** — explodes memory; sample neighbors
  (GraphSAGE) or use cluster/subgraph batching.
- **Mean/max readout loses size information** — for size-sensitive graph tasks, sum-pool or add
  explicit size features.
- **Ignoring edge features and direction** — many real graphs are directed or have typed edges;
  a plain symmetric GCN throws that signal away.
- **Weak aggregator under-fits structure** — mean/max can't distinguish some graphs WL can; use a
  sum-based GIN when structural expressiveness matters.

# Discrete Mathematics

## Overview

Discrete mathematics is the study of countable, distinct structures — sets, logic,
counting, integers, and graphs — as opposed to the continuous world of
[calculus](calculus.md). It is the native language of computer science: it underlies
[algorithms](../algorithms/index.html) and their correctness proofs,
[data structures](../data_structures/index.html), [databases](../databases/index.html)
(relational algebra is set theory), [cryptography](../security/encryption.md) (number
theory), and digital logic.

## Logic

Propositional logic — combining true/false statements:

```
¬  NOT      ∧  AND      ∨  OR      →  IMPLIES      ↔  IFF

p → q  is FALSE only when p is true and q is false.
Contrapositive:  p → q   ≡   ¬q → ¬p        (logically identical — basis of proofs)
Converse:        q → p                       (NOT equivalent to p → q)

De Morgan:  ¬(p ∧ q) ≡ ¬p ∨ ¬q      ¬(p ∨ q) ≡ ¬p ∧ ¬q
```

**Predicate logic** adds quantifiers over variables:

```
∀x P(x)   "for all x, P holds"        ∃x P(x)   "there exists an x with P"
Negation flips them:  ¬∀x P(x) ≡ ∃x ¬P(x)
```

This is the formal basis of [SQL](../databases/sql.md) `WHERE`/`EXISTS`, type systems,
and formal verification.

## Sets

```
∈ member   ⊆ subset   ∪ union   ∩ intersection   \ difference   |A| cardinality
Power set 𝒫(A): all subsets, |𝒫(A)| = 2ⁿ
Cartesian product A × B: all ordered pairs, |A×B| = |A|·|B|

De Morgan (sets):  (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ      (A ∩ B)ᶜ = Aᶜ ∪ Bᶜ
```

Sets are the foundation of relational databases (a table is a set of tuples; joins are
products + selection) and of [probability](probability.md) (events are sets).

## Relations and functions

```
Relation R ⊆ A × B.  Properties of a relation on a set:
  Reflexive   ∀a: aRa
  Symmetric   aRb → bRa
  Transitive  aRb ∧ bRc → aRc
  Equivalence = reflexive + symmetric + transitive  → partitions a set into classes
  Partial order = reflexive + antisymmetric + transitive  → DAGs, dependency ordering
```

Equivalence relations underpin hashing buckets and union-find; partial orders underpin
topological sort and build systems.

## Combinatorics — counting

```
Sum rule        either A or B (disjoint):   |A| + |B|
Product rule    A then B:                    |A| · |B|

Permutations (order matters)    P(n,k) = n! / (n−k)!
Combinations (order ignored)    C(n,k) = n! / (k!(n−k)!) = "n choose k"

Binomial theorem:  (x+y)ⁿ = Σ C(n,k) xᵏ yⁿ⁻ᵏ
Pigeonhole:        n items in m<n boxes → some box has ≥2 items
Inclusion–exclusion: |A∪B| = |A|+|B|−|A∩B|
```

Counting bounds the size of state spaces, hash collisions (birthday paradox), and the
complexity analyses in [algorithms](../algorithms/big_o.md).

## Number theory

The arithmetic of integers — and the engine of public-key [cryptography](../security/encryption.md).

```
Divisibility, primes, GCD (Euclid's algorithm: gcd(a,b)=gcd(b, a mod b))

Modular arithmetic:  a ≡ b (mod n)  means n | (a−b)
  (a+b) mod n = ((a mod n)+(b mod n)) mod n      (same for ×)

Modular inverse:  a·a⁻¹ ≡ 1 (mod n)  exists iff gcd(a,n)=1  (extended Euclid)
Fermat's little theorem:  aᵖ⁻¹ ≡ 1 (mod p)   for prime p, gcd(a,p)=1
Euler's theorem:          a^φ(n) ≡ 1 (mod n)  — the basis of RSA
Chinese Remainder Theorem: solve simultaneous congruences (speeds up RSA)
```

RSA's security rests on integer factorization being hard; Diffie–Hellman and ECC on the
discrete-log problem. See [public-key encryption](../security/encryption.md) and
[post-quantum crypto](../security/post_quantum_crypto.md) for what changes when these
assumptions break.

## Proof techniques

```
Direct           assume hypothesis, derive conclusion
Contrapositive   prove ¬q → ¬p instead of p → q
Contradiction    assume ¬conclusion, derive an impossibility (√2 irrational, ∞ primes)
Induction        base case P(0); inductive step P(k) → P(k+1)  ⟹  ∀n P(n)
  Strong induction: assume P(0..k) to prove P(k+1)  — natural for recursion/divide&conquer
```

Induction is the formal twin of [recursion](../algorithms/recursion.md): the same
structure that proves a recursive algorithm correct.

## Recurrences

Equations defining a sequence in terms of earlier terms — the runtime model for recursive
[algorithms](../algorithms/divide_and_conquer.md).

```
Fibonacci:  F(n) = F(n−1) + F(n−2)
Mergesort:  T(n) = 2·T(n/2) + O(n)

Master theorem for  T(n) = a·T(n/b) + f(n):
  compare f(n) with n^(log_b a):
    f smaller  → T = Θ(n^(log_b a))         (e.g. binary search)
    f equal    → T = Θ(n^(log_b a) · log n) (e.g. mergesort → n log n)
    f larger   → T = Θ(f(n))
```

## Graph theory

The mathematics of networks — a set of vertices `V` and edges `E`. The structural basis
of [graph algorithms](../algorithms/graph_algorithms.md) and [graph data structures](../data_structures/graphs.md).

```
Directed / undirected, weighted / unweighted, cyclic / acyclic (DAG)
Degree: # edges at a vertex.  Σ degrees = 2|E|  (handshake lemma)

Key structures:
  Tree         connected, acyclic, |E| = |V|−1
  DAG          directed acyclic → topological order exists (scheduling, build deps)
  Bipartite    2-colourable, no odd cycles → matching problems
  Complete Kₙ  every pair connected, |E| = n(n−1)/2

Euler path   uses every EDGE once    → exists iff 0 or 2 odd-degree vertices
Hamiltonian  visits every VERTEX once → NP-hard (TSP)
```

Graph coloring → register allocation and scheduling; bipartite matching → assignment
problems; spanning trees → network design.

## Boolean algebra

The algebra of digital logic and bit manipulation:

```
Identities:  a∧1=a   a∨0=a   a∧0=0   a∨1=1   a∧¬a=0   a∨¬a=1
Any function expressible via {AND, OR, NOT}; {NAND} alone is universal.
```

Directly relevant to [bit manipulation](../algorithms/bit_manipulation.md), digital
circuits, and processor [ISA](../embedded/isa.md) design.

## Where this shows up

- **Algorithms & DS** — complexity (counting), correctness (induction), structure (graphs).
- **Databases** — relational algebra is set theory; constraints are predicate logic.
- **Cryptography** — modular arithmetic, primes, discrete log; see [encryption](../security/encryption.md).
- **Compilers / verification** — logic, lattices, DAGs.

## Pitfalls

- **Confusing converse with contrapositive** — only the contrapositive is equivalent.
- **Off-by-one in induction** — forgetting or misstating the base case.
- **0! = 1 and C(n,0) = 1** — empty-product conventions that trip up counting.
- **Modular division** — you multiply by the modular *inverse*, not ordinary division.

# Transformers Architecture

> **Domain:** Deep Learning, NLP, Generative AI
> **Key Concepts:** Attention Mechanism, Self-Attention, Encoder-Decoder, Positional Encoding

The **Transformer**, introduced in the 2017 paper *"Attention Is All You Need"*, revolutionized NLP. It replaced Recurrent Neural Networks (RNNs/LSTMs) by enabling **parallel processing** of sequences and handling long-range dependencies perfectly via the **Attention** mechanism.

---

## 1. The High-Level Architecture

The original Transformer has two stacks:
1.  **Encoder:** Processes the input text to understand it. (Used in BERT).
2.  **Decoder:** Generates output text based on the Encoder's understanding. (Used in GPT).

*   **Encoder-Only Models (BERT):** Good for classification, sentiment analysis.
*   **Decoder-Only Models (GPT, Llama):** Good for text generation.
*   **Encoder-Decoder Models (T5, Bart):** Good for translation, summarization.

---

## 2. The Core Mechanism: Scaled Dot-Product Attention

This is the mathematical heart of the model.

**The Concept:** When processing the word "Bank" in "Bank of the river", the model needs to know that "river" is relevant context, but "money" is not.

**The Inputs:**
*   **Query ($Q$):** What I am looking for.
*   **Key ($K$):** What I have to offer.
*   **Value ($V$):** The actual content I hold.

**The Formula:**
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

1.  **$QK^T$ (Similarity):** Calculate the dot product of the Query with every Key. High dot product = High relevance.
2.  **Divide by $\sqrt{d_k}$:** Scaling factor to keep gradients stable.
3.  **Softmax:** Convert scores to probabilities (sum to 1). This is the "Attention Map".
4.  **Multiply by $V$:** Create a weighted sum of Values. If "river" has high attention, its value dominates the sum.

---

## 3. Multi-Head Attention

One attention head focuses on one type of relationship (e.g., syntax). To understand language, we need to focus on multiple things at once (syntax, semantics, tone).

*   **Mechanism:** Run the Attention step $h$ times in parallel with different weight matrices ($W_Q, W_K, W_V$).
*   **Result:** Concatenate the outputs and project them back to the original dimension.

---

## 4. Positional Encoding

Transformers processing the whole sentence at once (parallel). They have no concept of "order" or "sequence" by default. "Man bites Dog" looks the same as "Dog bites Man".

*   **Solution:** Add a vector to each input embedding that represents its position.
*   **Method:** Sine and Cosine functions of different frequencies.
    $$ PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}}) $$

---

## 5. Layer Components (The Block)

Each Encoder/Decoder layer consists of:

1.  **Multi-Head Attention**
2.  **Add & Norm:** Residual Connection (Input + Output) followed by Layer Normalization. This allows deep networks to train without vanishing gradients.
3.  **Feed-Forward Network (FFN):** A simple MLP applied to each token independently.
4.  **Add & Norm:** Again.

---

## 6. Decoder Nuances

The Decoder has two differences:
1.  **Masked Attention:** It cannot peek at future tokens. When generating token 5, it can only attend to tokens 1-4. The attention matrix is masked with $-\infty$ above the diagonal.
2.  **Cross-Attention:** In Encoder-Decoder models, the Decoder queries the Encoder's output. ($Q$ comes from Decoder, $K$ and $V$ come from Encoder).

---

## 7. Evolution

*   **Self-Attention:** $O(N^2)$ complexity. Long sequences are expensive.
*   **FlashAttention:** An IO-aware optimization that makes attention faster and memory-efficient by tiling GPU memory operations.
*   **RoPE (Rotary Positional Embeddings):** A modern replacement for sinusoidal encodings, used in Llama/PaLM.

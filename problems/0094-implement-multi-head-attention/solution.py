import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    return Q, K, V

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    # Step 1: Scaled dot-product attention scores
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)

    # Step 2: Numerically stable softmax
    scores = scores - np.max(
        scores, axis=-1, keepdims=True
    )

    exp_scores = np.exp(scores)

    attention_weights = exp_scores / np.sum(
        exp_scores, axis=-1, keepdims=True
    )

    # Step 3: Weighted sum of values
    output = attention_weights @ V

    return output

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    seq_len, d_model = Q.shape

    if n_heads <= 0:
        raise ValueError("n_heads must be positive")

    if d_model % n_heads != 0:
        raise ValueError(
            "d_model must be divisible by n_heads"
        )

    if K.shape != Q.shape or V.shape != Q.shape:
        raise ValueError("Q, K, and V must have the same shape")

    # Dimension of each head
    head_dim = d_model // n_heads

    head_outputs = []

    # Split features into independent heads
    for i in range(n_heads):
        start = i * head_dim
        end = start + head_dim

        Q_head = Q[:, start:end]
        K_head = K[:, start:end]
        V_head = V[:, start:end]

        # Compute attention independently for this head
        head_output = self_attention(
            Q_head, K_head, V_head
        )

        head_outputs.append(head_output)

    # Concatenate outputs along the feature dimension
    output = np.concatenate(head_outputs, axis=-1)

    return output
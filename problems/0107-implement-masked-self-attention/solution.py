import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Compute masked self-attention.
	"""
	d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)

    # Step 2: Apply mask before softmax
    scores = scores + mask

    # Step 3: Numerically stable softmax, row by row
    scores = scores - np.max(
        scores, axis=-1, keepdims=True
    )

    exp_scores = np.exp(scores)

    attention_weights = exp_scores / np.sum(
        exp_scores, axis=-1, keepdims=True
    )

    # Step 4: Weighted sum of values
    output = attention_weights @ V

    return output
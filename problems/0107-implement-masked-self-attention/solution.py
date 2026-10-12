import torch
import math

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return torch.matmul(X, W_q), torch.matmul(X, W_k), torch.matmul(X, W_v)

def masked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute masked self-attention.
    """
    # Your code here
    # Step 1: Compute attention scores
    d_k = K.shape[-1]
    scores = Q @ K.T / math.sqrt(d_k)
    scores+=mask

    # Step 2: Softmax
    attention_weights = torch.softmax(
        scores, dim=-1
    )

    # Step 3: Weighted sum of values
    output = attention_weights @ V

    return output

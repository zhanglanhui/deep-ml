import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
    n_samples, n_features = X.shape

    weights = np.zeros(n_features)
    bias = 0.0
    losses = []

    for _ in range(iterations):
        # 1. Forward propagation
        z = X @ weights + bias
        probabilities = 1.0 / (1.0 + np.exp(-z))

        # 2. Binary cross-entropy loss (sum, not mean)
        p = np.clip(probabilities, 1e-15, 1.0 - 1e-15)

        loss = -np.sum(
            y * np.log(p) + (1 - y) * np.log(1 - p)
        )
        losses.append(round(float(loss), 4))

        # 3. Calculate gradients (sum, not mean)
        errors = probabilities - y
        dw = X.T @ errors
        db = np.sum(errors)

        # 4. Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db

    # Return bias first, followed by weights
    coefficients = [bias, *weights]

    coefficients = [
        round(float(value), 4)
        for value in coefficients
    ]

    return coefficients, losses
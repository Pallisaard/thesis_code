import torch


def rbf_kernel(X: torch.Tensor, Y: torch.Tensor, sigma=1.0):
    """
    Computes the RBF kernel matrix between two sets of vectors. Assumes inputs are series of vectors (B, D).
    """
    X_norm = (X**2).sum(dim=1, keepdim=True)
    Y_norm = (Y**2).sum(dim=1, keepdim=True)
    dists = X_norm - 2 * X @ Y.T + Y_norm.T  # Squared Euclidean distances
    return torch.exp(-dists / (2 * sigma**2))


def compute_mmd(X, Y, sigma=1.0):
    """Computes the squared Maximum Mean Discrepancy sqrt(MMD^2) between two distributions."""
    K_XX = rbf_kernel(X, X, sigma)
    K_YY = rbf_kernel(Y, Y, sigma)
    K_XY = rbf_kernel(X, Y, sigma)

    mmd2 = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return torch.sqrt(mmd2)  # Optional: take square root for interpretability


class MMDMetric:
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def __call__(self, X, Y):
        return compute_mmd(X, Y, sigma=self.sigma)


if __name__ == "__main__":
    # Example Usage
    X = torch.randn(500, 512)  # 500 generated feature vectors (512D each)
    Y = torch.randn(500, 512)  # 500 real feature vectors (512D each)

    mmd_value = compute_mmd(X, Y, sigma=1.0)
    print("MMD:", mmd_value.item())

import torch
import numpy as np
from scipy import linalg


def compute_statistics(features: torch.Tensor):
    """
    Compute mean and covariance statistics for a batch of features.
    Args:
        features: Tensor of shape (N, D) where N is batch size and D is feature dimension
    Returns:
        mu: Mean vector
        sigma: Covariance matrix
    """
    mu = torch.mean(features, dim=0)
    sigma = torch.cov(features.T, correction=0)  # Use N instead of N-1
    return mu, sigma


def compute_fid(real_features: torch.Tensor, generated_features: torch.Tensor):
    """
    Compute Fréchet Inception Distance between two sets of features.
    Args:
        real_features: Tensor of shape (N, D) containing features from real data
        generated_features: Tensor of shape (M, D) containing features from generated data
    Returns:
        fid_score: The Fréchet Inception Distance
    """
    mu1, sigma1 = compute_statistics(real_features)
    mu2, sigma2 = compute_statistics(generated_features)

    # Convert to numpy for scipy operations
    mu1, mu2 = mu1.cpu().numpy(), mu2.cpu().numpy()
    sigma1, sigma2 = sigma1.cpu().numpy(), sigma2.cpu().numpy()

    # Calculate FID score with numerical stability
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    # Ensure covmean is real and positive semi-definite
    covmean = covmean.real  # type: ignore
    covmean = np.maximum(covmean, 0)  # Avoid negative values from numerical errors

    # Ensure non-negative trace
    trace_term = np.trace(sigma1 + sigma2 - 2 * covmean)
    trace_term = np.maximum(trace_term, 1e-6)  # Prevent negative values

    fid_score = (diff @ diff) + trace_term
    return torch.tensor(float(fid_score), dtype=torch.float32)


class FIDMetric:
    def __init__(self):
        pass

    def __call__(self, real_features: torch.Tensor, generated_features: torch.Tensor):
        return compute_fid(real_features, generated_features)


if __name__ == "__main__":
    # Example Usage
    real_features = torch.randn(500, 512)  # 500 real feature vectors (512D each)
    generated_features = torch.randn(
        500, 512
    )  # 500 generated feature vectors (512D each)

    fid_score = compute_fid(real_features, generated_features)
    print("FID:", fid_score.item())

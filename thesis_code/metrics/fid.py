import torch


def compute_fid(vec1, vec2):
    mean1 = vec1.mean(dim=0)
    mean2 = vec2.mean(dim=0)
    cov1 = torch.cov(vec1)
    cov2 = torch.cov(vec2)
    diff = mean1 - mean2
    covmean = torch.sqrt(cov1 @ cov2)
    return diff @ diff + torch.trace(cov1 + cov2 - 2 * covmean)


def compute_fid_from_volumes(vol1, vol2):
    flat_vol1 = vol1.view(vol1.size(0), -1)
    flat_vol2 = vol2.view(vol2.size(0), -1)
    return compute_fid(flat_vol1, flat_vol2)

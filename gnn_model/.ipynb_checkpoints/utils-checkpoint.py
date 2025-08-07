import numpy as np
import torch

# [preprocessing] - normalization function
def normalize_node_features(x, mean, std):
    return (x - mean) / (std + 1e-8)

def normalize_global_features(x, mean, std):
    return (x - mean) / (std + 1e-8)

# [training] - mixup
def mixup_embeddings(h, y, alpha=0.3):
    """ h: (B, D), y: (B,) """
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(h.size(0))
    h2 = h[index]
    y2 = y[index]
    h_mix = lam * h + (1 - lam) * h2
    y_mix = lam * y + (1 - lam) * y2
    return h_mix, y_mix
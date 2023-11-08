import torch
import numpy as np

""" Code was adapted from https://github.com/ndahlquist/pytorch-fourier-feature-networks/blob/master/fourier_feature_transform.py. """

class GaussianFourierFeatureTransform(torch.nn.Module):
    """ Gaussian Fourier Feature Transform

    Input: H,W,C
    Returns: H,W,mapping_size*2
    """
    
    def __init__(self, in_features, mapping_size=256, scale=5, device='cpu'):
        super().__init__()

        self.in_features = in_features
        self.mapping_size = mapping_size
        self.B = torch.randn((in_features, mapping_size)).to(device) * scale

    def forward(self, x):
        x = (np.pi * 2 * x) @ self.B
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
    
class CauchyFourierFeatureTransform(torch.nn.Module):
    """ Cauchy Fourier Feature Transform

    Input: H,W,C
    Returns: H,W,mapping_size*2
    """
    
    def __init__(self, in_features, mapping_size=256, scale=1, location=0.0, device='cpu'):
        super().__init__()

        self.in_features = in_features
        self.mapping_size = mapping_size

        # Create a Cauchy distribution for generating random features
        cauchy_dist = torch.distributions.Cauchy(location, scale)
        self.B = cauchy_dist.sample((in_features, mapping_size)).to(device)

    def forward(self, x):
        x = (np.pi * 2 * x) @ self.B
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
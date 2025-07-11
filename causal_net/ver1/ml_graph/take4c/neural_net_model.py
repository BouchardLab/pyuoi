#!/usr/bin/env python3
import torch
import torch.nn as nn

class SparseNetworkModel(nn.Module):
    def __init__(self, W_sparse, tau):
        super(SparseNetworkModel, self).__init__()
        self.W_model = nn.Parameter(W_sparse.to_dense()) #Store as dense for optimization
        self.register_buffer('tau', torch.tensor(float(tau)))

    def forward(self, x):
        # The equation is: X_t+1 = X_t + (-X_t + W*X_t)/tau
        # Simplified: X_t+1 = (1 - 1/tau) * X_t + (1/tau) * W * X_t
        
        # apply W to all samples in batch
        Wx = torch.matmul(x, self.W_model.T) # (batch,M) x (M,M).T -> (batch,M)
        
        # Note: x corresponds to X_t
        x_next = x + (-x + Wx) / self.tau
        return x_next 
#!/usr/bin/env python3
import torch
import torch.nn as nn

class SparseNetworkModel(nn.Module):
    def __init__(self, V_dense, E_mask, tau, num_neuron, num_excite):
        super(SparseNetworkModel, self).__init__()
        self.V_model = nn.Parameter(V_dense)
        self.register_buffer('E_mask', E_mask)
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32), requires_grad=False)
        self.num_neuron = num_neuron
        self.num_excite = num_excite

        # Define constant masks based on row index, for all columns
        self.diag_mask = torch.eye(num_neuron, dtype=torch.bool, device=V_dense.device)
        offdiag_mask = ~self.diag_mask
        
        excite_mask = torch.zeros((num_neuron, num_neuron), dtype=torch.bool, device=V_dense.device)
        excite_mask[:num_excite, :] = True
        
        self.excit_off_diag_mask = excite_mask & offdiag_mask
        self.inhib_off_diag_mask = ~excite_mask & offdiag_mask

    def get_w(self):
        """Constructs W from V based on Dale's Principle."""
        # Enforce sparsity by applying the mask at the beginning
        V_sparse_effective = self.V_model * self.E_mask
        
        V2 = V_sparse_effective ** 2
        W = torch.clone(V2)
        
        # Apply Dale's rules using broadcasted masks
        W[self.diag_mask] = -V2[self.diag_mask]
        W[self.inhib_off_diag_mask] = -V2[self.inhib_off_diag_mask]
        
        return W

    def forward(self, x):
        W = self.get_w()
        # Note: The model equation uses X @ W.T, which is equivalent to W @ X if X is a column vector.
        # Here, x has shape (batch, features), so we need W @ x.T, then transpose back.
        delta_x = (-x + (W @ x.T).T) / self.tau
        return x + delta_x 
# PoissonGML refers to a Poisson Generalized Linear Model (GLM)
import torch
import torch.nn as nn


class PoissonGLModel(nn.Module):
    """
    Generalized Linear Model for multivariate Poisson spike counts.
    rate_t = exp(A @ Y_prev + B) * dt

    Constructor:
      PoissonGLM(n_neurons)
        - Stage A: random initialization of A and B
      PoissonGLM(n_neurons, A_init, B_init, trainable_mask)
        - Stage B: efficient parameterization with only trainable weights in a single tensor
    """
    def __init__(self, n_neurons, A_init=None, B_init=None, trainable_mask=None, noise_scale=0.0):
        super().__init__()
        self.n_neurons = n_neurons
        
        if A_init is None:  # Stage A: standard parameterization
            self.A = nn.Parameter(torch.randn(n_neurons, n_neurons) * 0.1)
            self.B = nn.Parameter(torch.randn(n_neurons) * 0.1)
            self.is_stage_b = False
        else:  # Stage B: efficient parameterization
            self.is_stage_b = True
            
            # Store fixed components as buffers
            A_t = torch.tensor(A_init, dtype=torch.float32)
            B_t = torch.tensor(B_init, dtype=torch.float32)
            mask = torch.tensor(trainable_mask, dtype=torch.bool)
            
            self.register_buffer('A_fixed', A_t)  # Full A matrix for initialization
            self.register_buffer('B_fixed', B_t)  # Fixed B values (will be overridden)
            self.register_buffer('trainable_mask', mask)
            
            # Find trainable positions
            trainable_A_indices = torch.nonzero(mask, as_tuple=True)
            self.register_buffer('trainable_A_row_idx', trainable_A_indices[0])
            self.register_buffer('trainable_A_col_idx', trainable_A_indices[1])
            
            # Count trainable parameters
            n_trainable_A = mask.sum().item()
            n_trainable_B = n_neurons  # All B elements are trainable
            total_trainable = n_trainable_A + n_trainable_B
            
            # Single parameter tensor for all trainable weights
            self.C = nn.Parameter(torch.zeros(total_trainable))
            
            # Initialize C with current values + optional noise
            with torch.no_grad():
                # First part: trainable A elements
                trainable_A_values = A_t[trainable_A_indices]
                self.C[:n_trainable_A] = trainable_A_values
                
                # Second part: B elements
                self.C[n_trainable_A:] = B_t
                # Add random noise if requested
                if noise_scale > 0:
                    noise = torch.randn_like(self.C) * noise_scale
                    self.C += noise
                    #print(f"Added random noise with scale {noise_scale} to Stage B initialization")
            
            self.n_trainable_A = n_trainable_A

    @property
    def A(self):
        if not self.is_stage_b:
            return self._parameters['A']
        else:
            # Reconstruct A matrix from C and fixed values
            A_reconstructed = self.A_fixed.clone()
            A_reconstructed[self.trainable_A_row_idx, self.trainable_A_col_idx] = self.C[:self.n_trainable_A]
            return A_reconstructed
    
    @property 
    def B(self):
        if not self.is_stage_b:
            return self._parameters['B']
        else:
            # B elements are stored in the second part of C
            return self.C[self.n_trainable_A:]

    def forward(self, Y_prev, dt=0.01):
        linear = torch.addmm(self.B.unsqueeze(0), Y_prev, self.A.t()).clamp(min=-10, max=10)
        return torch.exp(linear) * dt


def poisson_nll_loss(spikes, targets, firing_rates):
    eps = 1e-8
    firing_rates_safe = torch.maximum(firing_rates, torch.tensor(0.1, device=firing_rates.device))
    weights = 1.0 / firing_rates_safe
    weights /= torch.mean(weights)
    weights = weights.unsqueeze(0)
    loss = -weights * targets * torch.log(spikes + eps) + weights * spikes
    return loss.mean()

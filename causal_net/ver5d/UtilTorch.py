import torch
import numpy as np

def check_gpu_availability():
    """Check GPU availability and set up device configuration."""
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA-enabled GPU environment.")
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    return device

def preprocess_data(Y, args):
    T, M = Y.shape
    
    # Apply decorrelation if requested
    if hasattr(args, 'desync_time') and args.desync_time > 0:
        print(f"\\n=== Applying Time Decorrelation ===")
        print(f"Shifting consecutive neurons by {args.desync_time} time bins")

        Y_decorr = Y.copy()
        for neuron_idx in range(M):
            shift_amount = neuron_idx * args.desync_time
            if shift_amount > 0:
                # Circular shift: move data to the right, wrap around
                Y_decorr[:, neuron_idx] = np.roll(Y[:, neuron_idx], shift_amount)

        print(f"Applied time shifts from 0 to {(M-1) * args.desync_time} bins")
        print(f"This destroys temporal correlations between neurons")
        Y = Y_decorr
        
    max_pairs = T - 1
    num_samples = args.num_samples
    if num_samples is None or num_samples > max_pairs:
        num_samples = max_pairs
    return Y[:num_samples], Y[1:num_samples + 1]

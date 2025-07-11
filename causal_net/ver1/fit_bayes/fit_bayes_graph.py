#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""
Bayesian Sparse Regression for Neural Connectivity Inference

This script performs Bayesian sparse regression to infer neural connectivity graphs 
from time-series neural activity data using the Pyro probabilistic programming library.
It implements a hierarchical Bayesian model with sparse priors to discover which 
neurons are connected and whether connections are excitatory or inhibitory.

Key Features:
- Hierarchical sparse priors (horseshoe-like) to encourage sparse connectivity
- Stochastic Variational Inference (SVI) with linear learning rate decay
- Batched training for scalability with large time-series datasets
- Posterior sampling with configurable timepoint subsampling for efficiency
- Probability thresholding for edge selection based on sign confidence
- Comprehensive evaluation against ground truth with confidence analysis
- GPU monitoring and performance tracking

The Bayesian Model:
- Time-series data: X[t] → Y[t] = X[t+1] - X[t] (discrete derivatives)
- Connectivity matrix A governs dynamics: Y[t] = A @ X[t] + noise
- Hierarchical priors: A[i,j] ~ Normal(0, λ[i,j] * τ) with sparse λ, τ
- Sparse connections inferred with excitatory/inhibitory classification
- Gaussian noise with learnable variance σ

Training Process:
- Linear learning rate schedule from initial to final learning rate
- Batched processing of random time windows for computational efficiency
- Posterior sampling using subset of timepoints for faster inference
- Edge selection based on posterior probability thresholds (not arbitrary sparsity)

Output:
- Inferred connectivity matrix with confidence scores per connection
- Detailed performance metrics (precision, recall, F1) by connection type
- Row-level neuron type classification (excitatory vs inhibitory)
- Training diagnostics and GPU utilization statistics

Usage: ./bayes_sparse_regression.py --input datafile --num_epochs 100 --batch_size 10000
"""

import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoNormal
import pyro.optim as optim
import time,os,hashlib
import subprocess
import argparse
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from pprint import pprint

# Import evaluation functions from separate module
from eval_bayes_graph import report_results, _print_matrix

# ==================================
#  HELPER & SETUP FUNCTIONS
# ==================================

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--final_lr", type=float, default=1e-4, help="Final learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=10_000, help="Batch size for training (time steps per batch)")
    parser.add_argument("--edge_probability", type=float, default=0.55, help="Posterior probability threshold for edge sign confidence [0-1]")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of posterior samples")
    parser.add_argument("--sample_timepoints", type=int, default=20_000, help="Number of timepoints to use for posterior sampling")
    parser.add_argument('-v',"--verb", type=int, default=1, help="Verbosity level")
    parser.add_argument("--fitName",  default=None,help='fit name')

    parser.add_argument('--time_range' , default=[50, 40_000],  nargs=2,   type=int, help='fit data time range')

    parser.add_argument("--basePath",default='out',help="head dir for any results")
    parser.add_argument("--inpName",  required=True,help='name of input data')

    args = parser.parse_args()

    args.dataPath=os.path.join(args.basePath,'input_uoi')    
    args.modelPath=os.path.join(args.basePath,'model_bayes')
   
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.dataPath)
    assert os.path.exists(args.modelPath)
    if args.time_range!=None: assert args.time_range[0] < args.time_range[1] 
    
    return args

def load_data(args,bigD,md,device="cuda"):
    
    pmd=md['payload']
    sem=md['selector']
    dmm=md['dale_truth']
    #pprint(md)    
  
    featData=bigD['all_features']
        
    #.... clip data
    tL,tR= args.time_range 
    print('FUV tbinLR:',tL,tR)
    assert tR < featData.shape[0]
    featData=featData[tL:tR].astype(np.float32)
    W=bigD['true_network_matrix']
    # Create the ternary ground truth edge type matrix E
    E = np.sign(W).astype(np.float32)
    bigD['true_edge_matrix']=E
    
    sem['time_range']=[args.time_range[0], args.time_range[1]]          
    sem['num_feature']=featData.shape[1]
    sem['num_time_bin']=featData.shape[0]
    
    bigD['fit_inp_data']=featData
   
    data = torch.tensor(featData.T, dtype=torch.float32).to(device)
    truth_E = torch.tensor(E, dtype=torch.float32).to(device)
    sparsity_frac = float(dmm['prob_synaptic_conn'])

    return data, truth_E, sparsity_frac

#...!...!....................
def buildBayesMeta(args,md):
    fim={};  md['fit_bayes']=fim
    
    fim['hash']=hashlib.md5(os.urandom(32)).hexdigest()[:6]
    if args.fitName==None:
        md['short_name']='fitb-%s'%(fim['hash'])
    else:
        md['short_name']=args.fitName
    pprint(fim); print( flush=True)
   
def make_regression(data):
    X = data[:, :-1]
    Y = data[:, 1:] - data[:, :-1]
    return X, Y

# ==================================
#  MODEL AND CORE PIPELINE
# ==================================

def model(X, Y):
    M, N = X.shape
    tau = pyro.sample("tau", dist.HalfCauchy(torch.tensor(1.0, device=X.device)))
    lambda_local = pyro.sample("lambda_local", dist.HalfCauchy(torch.ones(M, M, device=X.device)).to_event(2))
    A = pyro.sample("A", dist.Normal(0.0, lambda_local * tau).to_event(2))
    sigma = pyro.sample("sigma", dist.HalfCauchy(torch.tensor(1.0, device=X.device)))
    mean = A.matmul(X)
    pyro.sample("Y", dist.Normal(mean, sigma).to_event(2), obs=Y)

def create_batches(X, Y, batch_size):
    """Create batches from X and Y tensors along the time dimension."""
    _, N = X.shape
    N_Y = Y.shape[1]
    
    # Create indices for batching
    indices = torch.randperm(min(N, N_Y))  # Shuffle indices
    
    batches_X = []
    batches_Y = []
    
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        
        # For X, we can use the indices directly
        if len(batch_indices) <= N:
            batch_X = X[:, batch_indices]
        else:
            batch_X = X[:, batch_indices[:N]]
            
        # For Y, we need to be careful about the size difference
        if len(batch_indices) <= N_Y:
            batch_Y = Y[:, batch_indices]
        else:
            batch_Y = Y[:, batch_indices[:N_Y]]
            
        batches_X.append(batch_X)
        batches_Y.append(batch_Y)
    
    return batches_X, batches_Y

def run_training(model, X, Y, args, device):
    print(f"run_training, X:{X.shape}, batch_size={args.batch_size}, lr={args.lr}->{args.final_lr}, epochs={args.num_epochs}, edge_prob_thresh={args.edge_probability:.3f}")
    pyro.clear_param_store()
    guide = AutoNormal(model)

    # Create optimizer with initial learning rate
    optimizer = optim.Adam({'lr': args.lr})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    
    # Calculate linear learning rate schedule
    lr_decay = (args.lr - args.final_lr) / (args.num_epochs - 1) if args.num_epochs > 1 else 0

    start_time = time.time()
    gpu_temp, gpu_power, gpu_util = "N/A", "N/A", "N/A"
    
    if args.verb > 0:
        print("Starting training...")

    for epoch in range(args.num_epochs):
        # Calculate current learning rate
        current_lr = args.lr - epoch * lr_decay
        current_lr = max(current_lr, args.final_lr)

        # Create batches for this epoch
        batches_X, batches_Y = create_batches(X, Y, args.batch_size)
        
        # Process all batches and accumulate loss
        total_loss = 0.0
        for batch_X, batch_Y in zip(batches_X, batches_Y):
            loss = svi.step(batch_X, batch_Y)
            total_loss += loss
        
        # Average loss over batches
        avg_loss = total_loss / len(batches_X) if batches_X else 0.0
        
        # Update learning rate for next epoch (after optimizers are created)
        if len(optimizer.optim_objs) > 0:
            # Calculate next epoch learning rate
            next_epoch_lr = args.lr - (epoch + 1) * lr_decay
            next_epoch_lr = max(next_epoch_lr, args.final_lr)
            
            for optim_obj in optimizer.optim_objs.values():
                for param_group in optim_obj.param_groups:
                    param_group['lr'] = next_epoch_lr

        if epoch == args.num_epochs // 2 and device.type == 'cuda':
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,power.draw,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
                gpu_temp, gpu_power, gpu_util = result.stdout.strip().split(', ')
            except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
                gpu_temp, gpu_power, gpu_util = "Error", "Error", "Error"
        
        if epoch % 10 == 0:
            if args.verb > 0:
                elapsed_seconds = time.time() - start_time
                print(f"[ELBO] epoch {epoch:3d}  loss = {avg_loss:.0f}  lr={current_lr:.2e}  time={elapsed_seconds:.0f}s")
    
    elapsed_minutes = (time.time() - start_time) / 60
    train_stats = {"elapsed_minutes": elapsed_minutes, "gpu_temp": gpu_temp, "gpu_power": gpu_power, "gpu_util": gpu_util}
    return guide, train_stats

def build_graph_from_posterior(model, guide, X, args, device, file_sparsity):
    # Use a random subset of timepoints for posterior sampling (much faster)
    M, N = X.shape
    sample_size = min(args.sample_timepoints, N)
    if sample_size < N:
        # Randomly sample timepoints
        indices = torch.randperm(N, device=device)[:sample_size]
        X_sample = X[:, indices]
        if args.verb > 0:
            print(f"Using {sample_size:,} random timepoints (out of {N:,}) for posterior sampling")
    else:
        X_sample = X
        if args.verb > 0:
            print(f"Using all {N:,} timepoints for posterior sampling")
    
    predictive = Predictive(model, guide=guide, return_sites=["A", "tau"], num_samples=args.num_samples)
    samples = predictive(X_sample, None)
    A_samps, tau_samps = samples["A"], samples["tau"]
    S, M, _ = A_samps.shape

    I = torch.eye(M, device=device)
    W_samps = tau_samps.view(S, 1, 1) * A_samps + I.unsqueeze(0)
    idx = torch.arange(M, device=device)
    W_samps[:, idx, idx] = -1.0

    mask = ~torch.eye(M, dtype=torch.bool, device=device)
    
    # Use probability threshold for sign confidence
    prob_threshold = args.edge_probability
    if file_sparsity is not None and args.verb > 0:
        print(f"File contains sparsity={file_sparsity:.3f}, but using probability threshold={prob_threshold:.3f}")

    # Compute posterior probabilities for positive and negative edges
    prob_pos = (W_samps > 0).float().mean(dim=0)
    prob_neg = (W_samps < 0).float().mean(dim=0)
    
    # Keep edges where we're confident about the sign (either positive or negative)
    max_sign_prob = torch.max(prob_pos, prob_neg)
    keep = max_sign_prob >= prob_threshold
    
    # Assign sign based on higher probability (positive vs negative)
    edge_sign = torch.where(prob_pos > prob_neg, 1.0, -1.0)
    
    # Apply both presence and sign decisions
    W_est = keep.float() * edge_sign
    W_est[idx, idx] = -1.0  # Set diagonal to -1
    
    # Compute confidence based on the chosen sign for kept edges
    W_confidence = torch.zeros_like(max_sign_prob)
    W_confidence[W_est > 0] = prob_pos[W_est > 0]
    W_confidence[W_est < 0] = prob_neg[W_est < 0]
    
    # Compute actual sparsity for reporting
    num_nonzero = (W_est[mask] != 0).sum().item()
    total_offdiag = mask.sum().item()
    actual_sparsity = num_nonzero / total_offdiag if total_offdiag > 0 else 0.0

    return W_est, W_confidence, mask, actual_sparsity

# ==================================
#  MAIN
# ==================================
def main():
    args = setup_args()
    np.set_printoptions(linewidth=200, precision=2, suppress=True)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.verb > 0:
        print(f"Using device: {device}")

    inpF=args.inpName+'.act.h5'        
    expD,expMD=read4_data_hdf5(os.path.join(args.dataPath,inpF))
    if args.verb>=2:
        print('M:expMD:');  pprint(expMD)
        stop2


    data, truth_E, file_sparsity = load_data(args,expD,expMD, device=device)
    
    X, Y = make_regression(data)
    buildBayesMeta(args,expMD)
    
    guide, train_stats = run_training(model, X, Y, args, device)
    
    W_est, W_confidence, mask, used_sparsity = build_graph_from_posterior(model, guide, X, args, device, file_sparsity)

    report_results(W_est, W_confidence, truth_E, mask, train_stats, args, device, used_sparsity, data.shape[1])

    expD['bayes_network_matrix']=W_est.cpu().numpy()
    expD['bayes_edge_matrix']=np.sign(expD['bayes_network_matrix']).astype(np.float32)
    expD['bayes_matrix_confidence']=W_confidence.cpu().numpy()
         
    #...... WRITE   OUTPUT .........
    outF=os.path.join(args.modelPath,expMD['short_name']+'.fitBayes.h5')
    write4_data_hdf5(expD,outF,expMD)
    #pprint(expMD)    

    print(' ./postproc_bayes.py --basePath $basePath -e %s  -p e  -Y '%expMD['short_name'])
    print(' ./fit_graph_weights.py --basePath $basePath --inpName %s  -p e  -Y '%expMD['short_name'])
    
if __name__ == "__main__":
    main()

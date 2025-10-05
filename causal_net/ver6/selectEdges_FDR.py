#!/usr/bin/env python3
"""
This script loads LASSO connectivity matrices from K bootstraps of real and desync data.
Diagonal elements are separated to compute mean and std across real-data bootstraps.
For off-diagonal edges, median magnitudes from real bootstraps are compared to a pooled null distribution
from desync bootstraps within each row to compute empirical p-values.
Row-wise Benjamini–Hochberg FDR is applied to select statistically significant edges at the given alpha level.
FDR = False Discovery Rate - a statistical method for controlling errors when testing many hypotheses simultaneously.
"""
import argparse
import os
import numpy as np
from pprint import pprint
from toolbox.Util_NumpyIO import write_data_npz
from PlotterFitEval import Plotter
from UtilSelectFDR import (summary_reco_neuronNet, compare_triplets, eval_tagged_edges_4_simu, 
                           get_offdiag_triplets, edge_selector_fdr, load_bootstrap_data, 
                           load_auxiliary_plotting_data)

def print_table_4_Yao(evalD,fitMD):
    
    Nn=fitMD['fit_lasso'] ['num_neurons']
    Nedg=Nn*(Nn-1)
    recL='#L,Nn,Nedg,'
    recV='#prob,%d,%d,'%(Nn,Nedg)
    recM='#M,'
    recR='#rmse,'
    for etype in ['neg','pos']:
        tripV=evalD[etype]
        TP,FP,FN=tripV
        
        p=TP.shape[0]/Nedg
        if p>0 and p<1 : std=np.sqrt(p*(1-p)/Nedg)
        else: std=1./Nedg
        recL+=etype+'TP,std,'
        recV+='%.4f,%.4f,'%(p,std)

        res=TP[:,3]-TP[:,2]
        mean_val = np.mean(res)
        std_val = np.std(res)
        recM+=etype+'TP,std,'
        recR+='%.2e,%.2e,'%(mean_val,std_val)
        
        p=FP.shape[0]/Nedg
        if p>0 and p<1 : std=np.sqrt(p*(1-p)/Nedg)
        else: std=1./Nedg
        recL+=etype+'FP,std,'
        recV+='%.4f,%.4f,'%(p,std)
        
        p=FN.shape[0]/Nedg
        if p>0 and p<1 : std=np.sqrt(p*(1-p)/Nedg)
        else: std=1./Nedg
        recL+=etype+'FN,std,'
        recV+='%.4f,%.4f,'%(p,std)

    for xx in ['diag','bterm']:
        V=evalD[xx]
        res=V[:,1]-V[:,0]
        mean_val = np.mean(res)
        std_val = np.std(res)
        recM+=xx+',std,'
        recR+='%.2e,%.2e,'%(mean_val,std_val)
       
    print(recL)
    print(recV)
    print(recM)
    print(recR)


#########################
#  MAIN
#########################

def main():
    parser = argparse.ArgumentParser(description="Row-wise FDR edge selection from LASSO bootstraps")
    parser.add_argument("--dataName", type=str, required=True, help="Base name for the dataset")
    parser.add_argument("--dataPath", type=str, default="/pscratch/sd/b/balewski/2025_causalNet_tmp/")
    parser.add_argument("--verb", "-v", type=int, default=1, help="Verbosity level")
    parser.add_argument("--num_bootstraps", type=int, nargs='+', required=True, help="Number of bootstraps: 1 value (duplicated for real/desync) or 2 values [Kreal, Kdesync]")
    parser.add_argument("--alphaFDR", type=float, default=0.002, help="FDR significance level")
    
    parser.add_argument('-p',"--showPlots", type=str,nargs='+', default="f", help="Plot types to show: a=structure, e=experiment_eigen, f=freqSortA_histos")
    parser.add_argument("--outPath", type=str, default=None, help="Output path for plots (defaults to dataPath)")
    parser.add_argument('-X',"--noXterm", action="store_true", help="Disable X terminal for plotting")
    args = parser.parse_args()
    print(vars(args))
        
    if args.outPath is None:   
        args.outPath = args.dataPath
    args.showPlots=''.join(args.showPlots)
 
    dataName = args.dataName
    dataPath = args.dataPath
    # Handle 1 or 2 values for num_bootstraps
    if len(args.num_bootstraps) == 1:
        K = [args.num_bootstraps[0], args.num_bootstraps[0]]  # Duplicate single value
    elif len(args.num_bootstraps) == 2:
        K = args.num_bootstraps
    else:
        raise ValueError("--num_bootstraps must have 1 or 2 values")
    
    Kreal, Kdesync = K
    alpha = args.alphaFDR

    # Load all bootstrap data
    A_edges_real_list, A_edges_desync_list, A_real_list, B_real_list, output_meta,output_big1 = load_bootstrap_data(
        dataName, dataPath, K, args.verb
    )

    # Apply row-wise FDR to edges
    W_mask, W_pval, summary = edge_selector_fdr(A_edges_real_list, A_edges_desync_list, alpha=alpha)

    print("FDR selection summary:"); pprint(summary)
    
    # Compute averages and standard deviations from real data bootstraps
    A_stack = np.stack(A_real_list, axis=0)  # shape (K, N, N)
    B_stack = np.stack(B_real_list, axis=0)  # shape (K, N)
    
    A_avr = np.mean(A_stack, axis=0)
    A_std = np.std(A_stack, axis=0)
    B_avr = np.mean(B_stack, axis=0)
    B_std = np.std(B_stack, axis=0)
    
    # Create full W_mask that applies only to off-diagonal elements
    N = A_avr.shape[0]
    W_mask_full = np.eye(N, dtype=bool)  # Start with diagonal = True (keep diagonal)
    # Apply FDR mask to off-diagonal elements only
    off_diag_mask = ~np.eye(N, dtype=bool)
    W_mask_full[off_diag_mask] = W_mask[off_diag_mask]

    A_avr[~W_mask_full]=0.  # now none-existing edges are 0
    
    print(f"Computed averages and std from {K} real bootstraps")
    print(f"A_avr shape: {A_avr.shape}, B_avr shape: {B_avr.shape}")
    print(f"Mask preserves diagonal and selects {W_mask.sum()} significant off-diagonal edges")
    
    # Prepare output data
    output_data = {
        'W_mask': W_mask_full,
        'A_avr': A_avr,
        'A_std': A_std,
        'B_avr': B_avr,
        'B_std': B_std,
        'W_pval': W_pval,
        'summary': summary
    }
    for xx in [ 'losses_total', 'losses_epochs', 'losses_wo_L1']:
        output_data[xx]=output_big1[xx]

    output_meta['edge_selector']={'selector_type':'FDR', 'alpha':args.alphaFDR}
 
    # Save results
    output_file = os.path.join(dataPath, f"{dataName}.FDRselected.npz")
    write_data_npz(output_data, output_file, metaD=output_meta)
    print(f"FDR results saved to: {output_file}")

    print("\nNext step commands:")
    print(f"  ./fit_regressPoisson.py  --dataName {outName}  --num_epochs 50")
 
    
    # Generate plots if requested
    if args.showPlots:
        print(f"\nGenerating plots: {args.showPlots}")
        
        # Prepare plotting data (compatible with eval_fitLasso.py structure)
        fitD = output_data.copy()  # Use our processed output data as fitD
        fitMD = output_meta
   
        # Rename records so select_edges_from_fitLasso() has the expected names
        fitD['A_lasso'] = A_avr.copy()  # tmp
        fitD['B_lasso'] = B_avr.copy()
        #fitD['A_lasso'] = fitD.pop('A_avr')
        #fitD['B_lasso'] = fitD.pop('B_avr')
                
        # Load auxiliary data needed for plotting
        #maskMD={}
        #XspikeD, trueD, MD = load_auxiliary_plotting_data(fitMD, maskMD, dataName, dataPath, alpha)
        spikeD, trueD, MD = load_auxiliary_plotting_data(fitMD,  dataName, dataPath)
        # Add mask metadata and FDR method info
        
        MD['edge_selection_method'] = 'fdr'
        MD['fdr_alpha'] = alpha
    
        if fitMD['data_type']=='simDale':
            evalD=eval_tagged_edges_4_simu(fitD,trueD)            
            print_table_4_Yao(evalD,fitMD)
        
             
        edgeD=summary_reco_neuronNet(A_avr, A_std)
        
        # adjustment for plotting

        fitD['single_rates']=spikeD['single_rates']
        
        # Setup plotter
        args.prjName = dataName 
        plot = Plotter(args)
        
        # Generate plots based on showPlots argument
        if 'a' in args.showPlots:
            plot.summary_fitLasso(fitD,MD,figId=1)

        if 'b' in args.showPlots:
            assert  fitMD['data_type']=='simDale'
            plot.residuals(evalD,MD,figId=2)

        if 'c' in args.showPlots:            
            plot.summary_network(fitD, edgeD,MD, figId=3)

        if 'd' in args.showPlots:
            plot.freqSortA_histos(fitD, MD, spikeD, figId=4)
      

        plot.display_all()
        print("Plotting completed.")

if __name__ == "__main__":
    main()

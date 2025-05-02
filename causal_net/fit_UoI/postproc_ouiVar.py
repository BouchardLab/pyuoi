#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Analyze  
./postproc_ouiVar.py 

'''

import os
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from time import time
from pprint import pprint
import numpy as np
from PlotterModelFit import Plotter


import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3,4],  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-p", "--showPlots",  default='a b', nargs='+',help="abcd-string listing shown plots")
    
    parser.add_argument( "-Y","--noXterm", dest='noXterm',  action='store_false', default=True, help="enables X-term for interactive mode")
    parser.add_argument("--basePath",default='out',help="head dir for any results")
                        
    parser.add_argument('-e',"--expName",  default='fit-b7018a',help='UoI-VAR fitted model')

    parser.add_argument('-m', '--max_feature', default=20, type=int, help='max num of analyzed features')

    args = parser.parse_args()
    # make arguments  more flexible 
    
    args.modelPath=os.path.join(args.basePath,'model_uoi')
    args.outPath=os.path.join(args.basePath,'postproc')
    args.showPlots=''.join(args.showPlots)
  
    
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    
    assert os.path.exists(args.modelPath)
    assert os.path.exists(args.outPath)
    return args


#...!...!.................... 
def nice_print_model(bigD,md,mxFeat=None):
    #pprint(md)
    pmd=md['payload']
    sem=md['selector']
    fim=md['fit_uoi']
    lag=fim['lag_depth']
    ntime,nfeat=fim['data_shape']
    inpN=md['input_name']
    print('Postproc UoI-VAR  input=%s  nfeat=%d  ntime=%d lag=%d  fitTime=%.1f sec  ranks=%d'%(inpN,nfeat,ntime,lag,fim['fit_time'],fim['num_rank']))
    AV=bigD['fit_A_model']
    #1freqData=bigD['sel_feat_freq']
    freqData=[i for i in range(AV[0].shape[0])]
    bigD['sel_feat_freq']=freqData  # tmp
    
    if mxFeat!=None:
        nfeat=min(mxFeat,nfeat)
    
    # Function to format values
    def format_value(val):
        if abs(val) < 0.01:
            return "  .  "  # Represent zero as '-'
        return f"{val:+5.2f}"  # Format as +0.12 or -0.23

    
    col_indices = "feat " + "     ".join(f"{i:2d}" for i in range(nfeat))

    #..... print frequencies
    print('Frequencies per feature')
    print(col_indices)
    freq_txt=" Hz " + "  ".join('%5.1f'%freqData[i] for i in range(nfeat))
    print(freq_txt)
    
    for il in range(lag):
        print('\nA_model[%d]'%(il))
        A=AV[il]
        # Print column indices
        
        print(col_indices)
        # Print row index and formatted values
        for i in range(nfeat):
            row=A[i]
            formatted_row = "  ".join(format_value(row[j]) for j in range(nfeat) )
            print(f"{i:2d}  {formatted_row}")  # Row index + formatted values
        
 
#...!...!.................... 
def XXXpostproc_polyEH(expD,md):
    
    pmd=md['payload']
    smd=md['submit']
    nImg=pmd['num_sample']
    shots=smd['num_shots']
    assert pmd['inp_size']==1
      
  
    
    expD['rec_poly']=rec_poly

        
#...!...!.................... 
def XXXresidual_ana(expD,md):
    rdata=expD['rec_poly']
    tdata=expD['true_poly']
    res_data = rdata[0] - tdata
    mean = np.mean(res_data)
    std = np.std(res_data)
    # assuming normal distribution, compute std error of std estimator
    # SE_s=std/sqrt(2(n-1)), where n is number of samples
    N=res_data.shape[0]
    se_s=std/np.sqrt(2*(N-1))
    pom=md['postproc']
    pom['res_mean']=float(mean)
    pom['res_std']=float(std)
    pom['res_SE_s']=float(se_s)
  


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()
    np.set_printoptions(precision=3)
                    
    inpF=args.expName+'.fitUoI.h5'
    expD,expMD=read4_data_hdf5(os.path.join(args.modelPath,inpF))
    
    if args.verb>=2:
        print('M:expMD:');  pprint(expMD)
        stop2
 
    if 0: # older data fix 
        pmd=expMD['payload']
        cad=expMD['canned']
       

    nice_print_model(expD,expMD,mxFeat=args.max_feature)
    
    
    #postproc_polyEH(expD,expMD)
    #expMD['postproc']={'hw_calib':False}

    
    #...... WRITE  OUTPUT
    outF=os.path.join(args.outPath,expMD['short_name']+'.post.h5')
    write4_data_hdf5(expD,outF,expMD)

    
    #--------------------------------
    # ....  plotting ........
    args.prjName=expMD['short_name']
    #expMD['plot']={'resid_max_range':0.4}
    
        
    plot=Plotter(args)  
    #1expMD['truth_rangeLR']=[-0.3,0.5]

    if 'a' in args.showPlots:
        plot.A_matrix(expD,expMD,figId=1,lag=0)
    if 'b' in args.showPlots:
        plot.Aper_row(expD,expMD,figId=1,lag=0)

    plot.display_all()
    print('M:done')
    #pprint(expMD) #tmp

#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''

test tool for selecting data subsets
Dependency: pyplot & numpy

Perlmutter
dataPath=/global/cfs/cdirs/m2043/causal_inference/DIV13/features 

  --time_range 7 9 --num_feature 20 

CHANGES:
pre-proc
read obj: spike_freq (545, 710) float32
--> mon_freq_time

'qa_twindow_sec' --> mon_freq_twindow


''' 

import os,sys,hashlib
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from time import time
from pprint import pprint
import numpy as np
from toolbox.PlotterBackbone import PlotterBackbone


import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3,4],  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-p", "--showPlots",  default='a', nargs='+',help="abcd-string listing shown plots")

    parser.add_argument( "-Y","--noXterm", dest='noXterm',  action='store_false', default=True, help="enables X-term for interactive mode")

    
    parser.add_argument("--basePath",default='out',help="head dir for any results")
    parser.add_argument("--dataPath",default='/global/cfs/cdirs/m2043/causal_inference/DIV13/features' ,help="direct input path")
                        
    parser.add_argument("--inpName",  default='HET_80k_1_samp1kHz',help='name of input data')
    parser.add_argument("--fitName",  default=None,help='fit name')

    #.... selection params
    parser.add_argument('--num_feature', default=8, type=int, help='num of from full dataset')
    parser.add_argument('--time_range' , default=[0., 1.0],  nargs=2,   type=float, help='fit data time range')
    
    args = parser.parse_args()
    # make arguments  more flexible
    if args.dataPath==None:
        args.dataPath=os.path.join(args.basePath,'features')
    args.outPath=os.path.join(args.basePath,'postproc')
    args.showPlots=''.join(args.showPlots)
    
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    
    
    assert os.path.exists(args.dataPath)
    assert os.path.exists(args.outPath)
    if args.time_range!=None: assert args.time_range[0] < args.time_range[1] 
    return args


#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)

#...!...!..................
    def selector1(self,auxD,md,figId=1):
        pprint(md)
        
        pmd=md['payload']
        mxfeat=pmd['num_feature']
        ntime=pmd['num_time_bin']

        tit=md['short_name']

        figId=self.smart_append(figId)
        nrow,ncol=2,2
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,7))

        #tmpD={'mon_freq_data':freqData,'feat_idx_time':featIdxV}
        featIdxV=auxD['feat_idx_time']

        featCntV=[ len(x) for x in featIdxV]
        

        ax = self.plt.subplot(nrow,ncol,1)
        ax.plot(featCntV)
        xLab ='session time bins (%.1f sec/bin)'%pmd['qa_twindow_sec']
        ax.set(xlabel=xLab,ylabel='num features', title=tit)

        ax = self.plt.subplot(nrow,ncol,2)
        ax.hist(featCntV, bins=30, color='salmon', alpha=0.7)
        ax.set(xlabel='num features')
        
        return
        y1V=bigD['tot_spike_vs_fid']
        y2V=bigD['min_spike_delT']
        xLab='input feature index'
#............................
#............................
#............................


        
#...!...!....................
def downselect_data(bigD,md,args):
    freqThr1=4.0
    pmd=md['payload']
    pprint(pmd)
    sem={}  # data-selector metadata
    
    freqData=bigD['spike_freq']  # 2D  [features, time

    nfeat,ntbin=freqData.shape

    # Boolean mask: True where A > freqThr1
    mask = freqData > freqThr1

    # Create 2D list: Outer index = ntbin, Inner list = feature indices
    featIdxV = [list(np.where(mask[:, i])[0]) for i in range(ntbin)]

    for it in range(ntbin):
        idxL=featIdxV[it]
        print('it=%d  nfeat=%d  sample:%s'%(it,len(idxL),freqData[idxL[:10],it]))
        if it >10: break

    tmpD={'mon_freq_data':freqData,'feat_idx_time':featIdxV}
    return tmpD #freqData, featIdx

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()
    np.set_printoptions(precision=3)

    inpF=args.inpName+'.act.h5'
    expD,expMD=read4_data_hdf5(os.path.join(args.dataPath,inpF))
    if args.verb>=2:
        print('M:expMD:');  pprint(expMD)
        stop2

    tmpD=downselect_data(expD,expMD,args)
    
    
    #--------------------------------
    # ....  plotting ........
    args.prjName=expMD['short_name']
    expMD['plot']={}
    #expMD['plot']['time_rangeLR']=[0.,10.]

    plot=Plotter(args)
   
    if 'a' in args.showPlots:
        plot.selector1(tmpD,expMD,figId=1)

    plot.display_all()
    print('M:done')
    exit(0)
    #...... WRITE   OUTPUT .........
    outF=os.path.join(args.modelPath,expMD['short_name']+'.fit.h5')
    write4_data_hdf5(expD,outF,expMD)
    pprint(expMD)
    


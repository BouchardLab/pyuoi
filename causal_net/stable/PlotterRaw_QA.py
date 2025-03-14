__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from toolbox.PlotterBackbone import PlotterBackbone
from matplotlib import cm as cmap
import matplotlib.ticker as ticker
from pprint import pprint
import numpy as np
import matplotlib.gridspec as gridspec

from matplotlib.colors import LinearSegmentedColormap

#...!...!....................
def summary_column(md):
    #pprint(md)
    pmd=md['payload']
    smd=md['submit']
    tmd=md['transpile']
    pom=md['postproc']
    txt=md['short_name']
    txt+='\nback: %s'%smd['backend']
    txt+='\nshots/addr : %d'%(smd['num_shots']/pmd['num_addr'])
    txt+='\nshots/img : %d k'%(smd['num_shots']/1000)
    txt+='\nnum sample %d'%(pmd['num_sample'])
    txt+='\nsample size: %d'%(pmd['seq_len'])
    txt+='\nnum addr: %d'%pmd['num_addr']
    txt+='\nqubits: %d'%pmd['num_qubit']
    if 'ibm' in smd['backend']:  txt+='  RC: %r'%smd['random_compilation']
    txt+='\nnum 2q gates: %d'%tmd['2q_gate_count']
    txt+='\n2q gates depth: %d'%tmd['2q_gate_depth']

    #txt+='\nhwCalib: %s'%pom['hw_calib']
    #if pom['hw_calib']: txt+=' fac: %.2f'%pom['ampl_fact']
    return txt
    if 'noise_model' in smd:
        txt+='\nfake : %s'%(smd['noise_model'])       
 
   
  
#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)
        
#...!...!..................
    def spikes_general(self,bigD,md,figId=1):
        pprint(md)
        '''
        pmd=md['payload']
        plm=md['plot']
        nfeat=min(10,pmd['num_feature'])
        ntime=pmd['num_time_bin']
        '''
        tit='session '+md['short_name']
        
        figId=self.smart_append(figId)        
        nrow,ncol=2,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(10,4*nrow))

        y1V=bigD['tot_spike_vs_fid']
        y2V=bigD['min_spike_delT']
        xLab='input feature index'
        
        #....... spike count
        ax = self.plt.subplot(nrow,ncol,1)
        ax.plot(y1V)
        ax.set_yscale('log')
        ax.grid()
        ax.set(xlabel=xLab, ylabel='spike count',title=tit)

        #....... min time between spikes
        ax = self.plt.subplot(nrow,ncol,2)
        ax.plot(y2V)
        ax.set_yscale('log')
        
        ax.grid()
        ax.set(xlabel=xLab, ylabel='min spike del time (10 kHz bin) ',title=tit)

        return
      
#...!...!..................
    def xyz(self,bigD,md,figId=3):
        #pprint(md)
        pmd=md['payload']
        smd=md['submit']
        tmd=md['transpile']

        figId=self.smart_append(figId)        
        nrow,ncol=1,2
        fig=self.plt.figure(figId,facecolor='white', figsize=(12,4))

        make_it_work

#...!...!..................
    def input_features_dense(self,bigD,md,figId=1):
        pprint(md)
        pmd=md['payload']
        plm=md['plot']
        nfeat=min(9,pmd['num_feature'])
        ntime=pmd['num_time_bin']
        nrow,ncol=nfeat,1
         
        #axes=self.blank_share2D(nrow=nrow,ncol=ncol, figsize=(12,1.5*nrow),figId=figId)
        #axes=self.blank_share2D(nrow=nrow,ncol=ncol, figsize=(20,0.6*nrow),figId=figId)  

        timeV=bigD['time']
        featIdL=bigD['feature_id']
        width =0.0005 
        for k in range(nrow):
            ax = axes[k]
            j=k+1
            featV=bigD['feature'][j]
            fid=featIdL[k]
            spikeV=bigD['spike'][j].astype(float)
            # Plot Exponential Decay as Filled Area
            ax.fill_between(timeV, 0,featV , color='red', alpha=0.3, label='feature=%d'%fid)

            # .... decorations ....
            ax.legend()
            ax.set_ylim(0,2.1)
            ax.set_ylabel('Ampl')
            if k==0:ax.set_title('Spikes with Exponential Decay, data=%s'%md['short_name'] )
            
            
        # common
        if 'time_rangeLR' in plm:  ax.set_xlim(tuple(plm['time_rangeLR']))
        ax.set_xlabel('Time (s)')

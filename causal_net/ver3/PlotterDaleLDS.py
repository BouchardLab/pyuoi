__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from toolbox.PlotterBackbone import PlotterBackbone
from matplotlib import cm as cmap
import matplotlib.ticker as ticker
from pprint import pprint
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors

    
#...!...!....................
def summary_column(md):
    #pprint(md)
    pmd=md['payload']
    sem=md['selector']
    
    txt=md['short_name']
    txt+='\nsession '+pmd['session_name']
    txt+='\nsampFreq %d Hz'%(pmd['sampling_freq'])
    txt+='\ndecay:%d ms,  len:%d ms '%(pmd['tau_decay'][0]*1000., pmd['tau_decay'][1]*1000.)
    txt+='\nsel time [%.1f %.1f] s'%(sem['time_range'][0],sem['time_range'][1])
    txt+='\nsel features %d'%(sem['num_feature'])
    
    return txt


#...!...!..................
def plot_dale_matrix(fig,ax,W):
    normMap = colors.TwoSlopeNorm(vmin=W.min(), vcenter=0, vmax=W.max())
    
    im=ax.imshow(W, aspect='auto', origin='lower', cmap='bwr', norm=normMap, interpolation='nearest')
    ax.set( xlabel='presyn. node index, source', ylabel='postsyn. node index, target')

    ax.set_aspect(1.0)
    ax.grid()
    # Create the colorbar.
    cbar = fig.colorbar(im, ax=ax, extend="both")
    cbar.set_label('Dal-Matrix: coupling strength')

#...!...!..................
def plot_dale_eigen(fig,ax,Eigen,tit="Eigenvalue Spectrum"):
    real_parts = np.real(Eigen)
    imag_parts = np.imag(Eigen)
    ax.scatter(real_parts, imag_parts, color='blue', marker='o')
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.set_title(tit)
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.grid(True)
    
    ax.axvline(0,color='red', linestyle='--')
        
#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)

#...!...!..................
    def Dale_matrix_and_eigen(self,bigD,md,figId=3):
        dmm=md['dale_truth']    
        figId=self.smart_append(figId)        
        nrow,ncol=1,2
        fig=self.plt.figure(figId,facecolor='white', figsize=(10,5))

        #.... left ......
        ax = self.plt.subplot(nrow,ncol,1)
        W=bigD['Wtrue'].T

        plot_dale_matrix(fig,ax,W)
    
        tit='True Dale, M%d,%s'%(W.shape[0],md['short_name'])
        ax.set(title=tit)
        numExc=dmm['num_excit_neur']
        ax.axvline(numExc-0.5,color='k',ls='--')
        ax.text(0.1, 0.92, 'Excitatory', size=18,color='r',transform=ax.transAxes)
        ax.text(0.6, 0.92, 'Inhibitory', size=18,color='b',transform=ax.transAxes)
        #..... right......
        ax = self.plt.subplot(nrow,ncol,2)
        Eigen=bigD['Weigen']
        plot_dale_eigen(fig,ax,Eigen)
        
        
        
#...!...!..................
    def Dale_stats(self,bigD,md,figId=3):
        dmm=md['dale_truth']        
        tit=md['short_name']

        figId=self.smart_append(figId)        
        nrow,ncol=2,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(6,5))        

        W=bigD['Wtrue'].T
        nnExcit=dmm['num_excit_neur']
        nnAny=dmm['num_any_neur']
        nnInhib=nnAny-nnExcit

        # Sanity check
        assert W.shape == (nnAny, nnAny)

        if 0:  # Build mask to skip diagonal
            diag_mask = ~np.eye(nnAny, dtype=bool)
            # Apply diagonal mask
            W1 = W[diag_mask].reshape(nnAny, nnAny - 1)
        else:  # show all
            W1=W
        
        # Mask excitatory and inhibitory columns
        W_excit = W1[:, :nnExcit].flatten()
        W_inhib = W1[:, nnExcit:].flatten()

        # skip 0's
        W_excit = W_excit[W_excit != 0]
        W_inhib = W_inhib[W_inhib != 0]

        
        ax = self.plt.subplot(nrow,ncol,1)
        ax.hist(W_excit, bins=100, color='tab:red')
        ax.set_title(tit+'  Excitatory Weights')
        ax.set_yscale('log')
        ax.grid()
        ax.text(0.1, 0.6, 'diagonal',transform=ax.transAxes,rotation=45)
        ax.text(0.7, 0.8, 'off-diagonal',transform=ax.transAxes)
        
        ax = self.plt.subplot(nrow,ncol,2)
        ax.hist(W_inhib, bins=100, color='tab:blue')
        ax.set_title('Inhibitory Weights')
        ax.set_xlabel('Synaptic Weight Value')
        ax.grid()
        ax.set_yscale('log')
        ax.text(0.1, 0.6, 'diagonal',transform=ax.transAxes,rotation=45)
        ax.text(0.4, 0.8, 'off-diagonal',transform=ax.transAxes)

        
#...!...!..................
    def rate_sample(self,bigD,md,nidxL,obsN='rate',figId=3):
        nn=min(10,len(nidxL))
        sim=md['simu']
        dmm=md['dale_truth']
        pom=md['postproc']
        
        figId=self.smart_append(figId)        
        nrow,ncol=nn,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(10,8))        

        tit='sim=%s'%(md['short_name'])
        timeV=bigD['evol_time']
        
        #print(sorted(bigD))
        obsV=bigD['simu_state']
        
        if obsN=='state':   
            tit+=' obs=state'
            yLab='state (a.u.)'
            dCol='darkorange'
        
        if obsN=='rate':
            tBin_sec=sim['step_duration']
            obsV=np.exp(obsV)*tBin_sec
            tit+=', obs= spike prob/Tbin' 
            yLab='spke prob/bin'
            dCol='b'
            
        if obsN=='spikes':
            obsV=bigD['simu_spikes'].astype(np.float32)
            tit+=' obs=spikes'
            yLab='count'
            dCol='red'

        tit+=', sig_noise=%.1f'%(sim['sigma_noise'])
        tit+=', dt=%d msec'%(sim['step_duration']*1000.)
        tit+=', tau=%d msec'%(sim['tau_response']*1000.)
        
        #print('tt',obsN,timeV.shape,obsV.shape)
        for n in range(nn):
            k=nidxL[n]
            ax = self.plt.subplot(nrow,ncol,n+1)
            if timeV.shape[0]>1001:                
                ax.plot(timeV,obsV[k],color=dCol)
            else:  # make it look more realistic like a histogram
                ax.step(timeV, obsV[k], color=dCol, where='mid')
            
            ax.set(ylabel=yLab)
            ax.text(0.05, 0.8, 'neuron %d'%k,color='r',transform=ax.transAxes)
            #if obsN=='rate': ax.axhline(1,lw=1,ls='--',c='k')
            if obsN=='state': ax.axhline(0,lw=1,ls='--',c='k')
            if n>0: continue
            ax.set(title=tit)
            
        ax.set(xlabel='evolution time (time bins)')
        
                   
#...!...!..................
    def evoked_energy(self,bigD,md,figId=3):
        figId=self.smart_append(figId)        
        nrow,ncol=1,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,3))
        ax = self.plt.subplot(nrow,ncol,1)

 
        dmm=md['dale_truth']
        sim=md['simu']
        eneV=bigD['raw_energy']
        timeV=bigD['evol_time']

        if 'time_0' in md['plot']:
            t0=md['plot']['time_0']
            it=int(t0/sim['time_step'])
            ene=ene[it:]
            timeV=timeV[it:]
        #print('tt2',timeV.shape,eneV.shape)
        ax.plot(timeV,eneV,'darkviolet')
        
        tit='sim=%s , Energy not normalized,  sigma=%.1f'%(md['short_name'],md['simu']['sigma_noise'])
        ax.set(xlabel='evolution time (msec)', ylabel='Evoked energy (a.u.)',title=tit)
        ax.set_yscale('log')
        ax.grid()
        
        

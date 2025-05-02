__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from toolbox.PlotterBackbone import PlotterBackbone
from matplotlib import cm as cmap
import matplotlib.ticker as ticker
from pprint import pprint
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors

#from matplotlib.colors import LinearSegmentedColormap


    
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

 
#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)

#...!...!..................
    def Dale_matrix(self,bigD,md,figId=3):

        figId=self.smart_append(figId)        
        nrow,ncol=1,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,7))
        ax = self.plt.subplot(nrow,ncol,1)
        W=bigD['Wtrue'].T
        nfeat=W.shape[0]
        
        # Create a normalization that centers at 0.
        #print('wmax=',W.max())
        normMap = colors.TwoSlopeNorm(vmin=W.min(), vcenter=0, vmax=W.max())

        
        im=ax.imshow(W, aspect='auto', origin='lower', cmap='bwr', norm=normMap, interpolation='nearest')
        tit='True Dale-matrix, %d neurons, name=%s'%(nfeat,md['short_name'])
        ax.set(title=tit, xlabel='presyn. node index, source', ylabel='postsyn. node index, target')
        # Create the colorbar.
        cbar = fig.colorbar(im, ax=ax, extend="both")
        cbar.set_label('coupling strength')
        # Define five ticks: min, midpoint (min to 0), 0, midpoint (0 to max), and max.
        tick_min = W.min()
        tick_max = W.max()
        tick_mid_left = (tick_min + 0) / 2
        tick_mid_right = (0 + tick_max) / 2
        ticks = [tick_min, tick_mid_left, 0, tick_mid_right, tick_max]
        
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
        ax.grid()

        ax.set_xlim(-0.5,nfeat+0.5)
        ax.set_ylim(-0.5,nfeat+0.5)
        ax.set_aspect(1.0)
        
#...!...!..................
    def Dale_eigen(self,bigD,md,figId=3):

        #pmd=md['payload']
        #tit=md['short_name']+' M-matrix[lag=%d]'%(lag)

        figId=self.smart_append(figId)        
        nrow,ncol=1,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(7,6))
        ax = self.plt.subplot(nrow,ncol,1)

        Eigen=bigD['Weigen']
        
        real_parts = np.real(Eigen)
        imag_parts = np.imag(Eigen)
        ax.scatter(real_parts, imag_parts, color='blue', marker='o')
        ax.set_xlabel("Real Part")
        ax.set_ylabel("Imaginary Part")
        ax.set_title("Eigenvalue Spectrum")
        ax.axhline(0, color='black', lw=0.5)
        ax.axvline(0, color='black', lw=0.5)
        ax.grid(True)

        ax.axvline(0,color='red', linestyle='--')
        
#...!...!..................
    def Dale_stats(self,bigD,md,figId=3):
        dmm=md['dale_truth']        
        tit=md['short_name']

        figId=self.smart_append(figId)        
        nrow,ncol=2,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(6,6))        

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
        dmm=md['dale_truth']        
        nn=min(10,len(nidxL))        
        
        figId=self.smart_append(figId)        
        nrow,ncol=nn,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(10,8))        

        tit='sim=%s'%(md['short_name'])
        timeV=bigD['evol_time']
        obsV=bigD['evol_state']
                    
        if obsN=='rate':
            obsV=np.exp(obsV)
            max_rate = 130  # max limit for rate
            obsV = np.minimum(obsV , max_rate)
            tit+=', obs=rate' # clip %d (Hz)'%(max_rate)
            yLab='rate (Hz)'
        if obsN=='state':
            tit+=' obs=state'
            yLab='state (a.u.)'

        
        print('tt',timeV.shape,obsV.shape)
        for n in range(nn):
            k=nidxL[n]
            ax = self.plt.subplot(nrow,ncol,n+1)
            ax.plot(timeV,obsV[:,k])
            ax.set(ylabel=yLab)
            #ax.set_ylim(-0.5,)
            ax.text(0.05, 0.8, 'neuron %d'%k,color='r',transform=ax.transAxes)
            #if t0!=None: ax.set_xlim(t0,)
            if obsN=='state': ax.axhline(0,lw=1,ls='--',c='k')
            if n>0: continue
            ax.set(title=tit)
            
        ax.set(xlabel='evolution time (sec)')
        
                   
#...!...!..................
    def evoked_energy(self,bigD,md,figId=3):
        figId=self.smart_append(figId)        
        nrow,ncol=1,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,3))
        ax = self.plt.subplot(nrow,ncol,1)

 
        dmm=md['dale_truth']
        smd=md['simu']
        ene=bigD['raw_energy']
        timeV=bigD['evol_time']

        if 'time_0' in md['plot']:
            t0=md['plot']['time_0']
            it=int(t0/smd['time_step'])
            ene=ene[it:]
            timeV=timeV[it:]
            #print('it:',it)
            #ax.set_xlim(t0,)
            
        ax.plot(timeV,ene,'g')
        
        tit='sim=%s , Energy not normalized,  sigma=%.1f'%(md['short_name'],md['simu']['sigma_noise'])
        ax.set(xlabel='evolution time (sec)', ylabel='Evoked energy (a.u.)',title=tit)
        ax.set_yscale('log')
        ax.grid()
        
        

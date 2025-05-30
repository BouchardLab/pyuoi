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

 
#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)

#...!...!..................
    def Dale_matrix(self,bigD,md,figId=3):
        dmm=md['dale_truth']    
        figId=self.smart_append(figId)        
        nrow,ncol=1,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(7,6))
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
        cbar.set_label('Dal-Matrix: coupling strength')
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

        numExc=dmm['num_excit_neur']
        ax.axvline(numExc-0.5,color='k',ls='--')
        ax.text(0.1, 0.92, 'Excitatory', size=18,color='r',transform=ax.transAxes)
        ax.text(0.6, 0.92, 'Inhibitory', size=18,color='b',transform=ax.transAxes)
        
#...!...!..................
    def Dale_eigen(self,bigD,md,figId=3):
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
        nn=min(10,len(nidxL))
        smd=md['simu']
        dmm=md['dale_truth']
        pom=md['postproc']
        
        figId=self.smart_append(figId)        
        nrow,ncol=nn,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(10,8))        

        tit='sim=%s'%(md['short_name'])
        timeV=bigD['evol_time']
        obsV=bigD['evol_state']
                    
        if obsN=='rate':
            obsV=np.exp(obsV)
            #max_rate = 130  # max limit for rate
            #obsV = np.minimum(obsV , max_rate)
            tit+=', obs=rate' # clip %d (Hz)'%(max_rate)
            yLab='rate (Hz)'
            dCol='b'
        if obsN=='state':
            tit+=' obs=state'
            yLab='state (a.u.)'
            dCol='darkorange'

        tit+=', sig_noise=%.1f'%(smd['sigma_noise'])
        if pom['time_rebin'] >1 : tit+=', dt=%.3f sec'%pom['time_step']
        
        #print('tt',timeV.shape,obsV.shape)
        for n in range(nn):
            k=nidxL[n]
            ax = self.plt.subplot(nrow,ncol,n+1)
            if timeV.shape[0]>100:
                ax.plot(timeV,obsV[:,k],color=dCol)
            else:  # make it look more realistic like a histogram
                ax.step(timeV, obsV[:, k], color=dCol, where='mid')
            
            ax.set(ylabel=yLab)
            ax.text(0.05, 0.8, 'neuron %d'%k,color='r',transform=ax.transAxes)
            if obsN=='rate': ax.axhline(1,lw=1,ls='--',c='k')
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
            
        ax.plot(timeV,ene,'darkviolet')
        
        tit='sim=%s , Energy not normalized,  sigma=%.1f'%(md['short_name'],md['simu']['sigma_noise'])
        ax.set(xlabel='evolution time (sec)', ylabel='Evoked energy (a.u.)',title=tit)
        ax.set_yscale('log')
        ax.grid()
        
        
#...!...!..................
    def rate_correl(self,bigD,md,nidxL,obsN='state',figId=3):
        assert obsN in ['state','rate']
        nn=min(8,len(nidxL))
        nidxL=nidxL[:nn]
        smd=md['simu']
        dmm=md['dale_truth']
        pom=md['postproc']

        obsV=bigD['evol_state']
        valV=obsV[:,nidxL]  # select channels

        # add text above all plots
        tit='sim=%s'%(md['short_name'])
                
        if obsN=='rate':
            obsV=np.exp(obsV)
            max_rate = 10  # max limit for rate
            obsV = np.minimum(obsV , max_rate)
            xxVal=1  # baseline rate
            tit+=', obs=rate (Hz)'
        else:
            xxVal=0 # baseline state
            tit+=', obs=state=log(rate/Hz)'

        tit+=', sig_noise=%.1f'%(smd['sigma_noise'])
        
        '''
        Parameters:
        -----------
        valV : array-like, shape (nSamp, nFeat)
        Each column is one feature; rows are samples.
        '''

        figId=self.smart_append(figId)
        nFeat = valV.shape[1]
        fig, axes = self.plt.subplots(
            nFeat, nFeat,
            figsize=(2 * nFeat, 2 * nFeat),
            gridspec_kw={'wspace': 0, 'hspace': 0},
            num=figId
        )

        
        if pom['time_rebin'] >1 : tit+=', dt=%.3f sec'%pom['time_step']
        
        fig.suptitle(tit, fontsize=16)
        # adjust layout to make room for the super-title
        fig.subplots_adjust(top=0.93)

        for i in range(nFeat):
            for j in range(nFeat):
                # invert vertical index so feature 0 is at bottom
                row = nFeat - 1 - i
                ax = axes[row][j]
                if i == j:
                    # diagonal: 1D histogram of feature i
                    ax.hist(valV[:, i], bins='auto', edgecolor='none',color='g')
                    ax.axvline(xxVal,c='k',ls='--')
                else:
                    # off-diagonal: scatter feature j vs feature i
                    x = valV[:, j]
                    y = valV[:, i]
                    ax.scatter(x, y, s=5, alpha=0.4,c='grey')

                    # compute correlation coefficient
                    r = np.corrcoef(x, y)[0, 1]
                    mean_x, mean_y = x.mean(), y.mean()
                    std_x, std_y = x.std(ddof=0), y.std(ddof=0)

                    # regression‐style slope through the centroid
                    if std_x > 0:
                        slope = r * (std_y / std_x)
                    else:
                        slope = 0.0

                    tCol='m' if r>0 else 'b'
                    
                    # draw dashed line through (mean_x, mean_y)
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    xs = np.array(xlim)
                    ys = slope * (xs - mean_x) + mean_y
                    ax.plot(xs, ys, '--', color=tCol, linewidth=1)
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    # add cross at origin
                    ax.plot(xxVal,xxVal, marker='x', color=tCol)
                    # annotate r in bottom-left corner                    
                    ax.text(xlim[0], ylim[0], ' r=%.2f' % r,
                            va='bottom', ha='left', fontsize=10,c=tCol)

                
                # hide inner x-axes
                if row < nFeat - 1:
                    ax.xaxis.set_visible(False)
                else:
                    ax.set_xlabel('Feat %d'%nidxL[j], fontsize=8)

                # hide inner y-axes
                if j > 0:
                    ax.yaxis.set_visible(False)
                else:
                    ax.set_ylabel('Feat %d'%nidxL[i], fontsize=8)
        self.plt.tight_layout(pad=0)




       

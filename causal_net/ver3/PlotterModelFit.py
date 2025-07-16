#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
from matplotlib import cm as cmap
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from pprint import pprint

from toolbox.PlotterBackbone import PlotterBackbone
from PlotterDaleLDS import plot_dale_matrix, plot_dale_eigen

def summary_column(md):
    pmd=md['dataset']
    sem=md['selector']
    txt=md['short_name']
    txt+='\ninput '+sem['input_name']
    #txt+='\nsampFreq %d Hz'%(sem['sampling_freq'])
    txt+='\ndecay:%d ms '%(pmd['simu']['tau_discount'])
    txt+='\nsel time [%.1f %.1f] s'%(sem['time_range'][0],sem['time_range'][1])
    txt+='\nsel features %d'%(sem['num_feature'])
    return txt

def plot_diagonal_and_violins(A,plt,figId,tit0,eps=1e-5):
    nfeat=A.shape[0]
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,7),gridspec_kw={'height_ratios':[1,2]},num=figId)
    
    # Top Row: Diagonal elements
    diagV=np.diag(A)
    ax1.plot(diagV,marker='o',linestyle='-',color='blue',label='Diagonal Elements')
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Diagonal Value')
    ax1.grid(True)
    ax1.set_xlim(-0.5,nfeat+0.5)
    ax1.set_title('%s  diagonal'%(tit0))

    # Customizing x-axis
    tickL=5
    x_ticks=np.arange(0,nfeat,tickL)
    ax1.set_xticks(x_ticks)
    
    # Bottom Row: Violin plots
    violin_data=[]
    for i in range(nfeat):
        row_values=np.delete(A[i,:],i)
        row_values=row_values[np.abs(row_values)>=eps]
        violin_data.append(row_values if len(row_values)>0 else [0])

    ax2.violinplot(violin_data,positions=np.arange(nfeat),showmeans=True,showmedians=True)
    ax2.set_xlim(-0.5,nfeat+0.5)
    ax2.set_xticks(x_ticks)
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('Off-diagonal Values')
    ax2.set_title('Vertical Violin Plots for Each Row of A')
    ax2.grid(True)
    ax2.axhline(0,lw=1.,ls='--',c='k')
    return ax2

def draw_correlation_plot(ax,XY,stD,dCol):
    ax.scatter(XY[:,0],XY[:,1],facecolors='none',edgecolors=dCol,label='all')
    ax.plot(stD['mu_X'],stD['mu_Y'],'+',color='#00ff00',markersize=20, markeredgewidth=3 )
    ax.text(0.1,0.92,'Corr=%.2f'%(stD['rho']),transform=ax.transAxes,color='r')
    ax.set_xlabel('true')
    ax.set_ylabel('UoI ADMM fit')
    #ax.plot([0],[0])
    ax.grid()
    #return
    
    ax.set_aspect(1.0)    
    # Set grid only at even integer multiples
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # draw x=y line
    ax.plot([x_min, x_max],[x_min, x_max],'--',color='cyan')
    
    return
    
def add_histogram(ax,data,dLab0,dCol):
    stdX=np.std(data)
    dLab='%s std=%.3f, n=%d'%(dLab0,stdX,data.shape[0])
    ax.hist(data,bins=30,color=dCol,histtype='step',alpha=0.7,label=dLab,linewidth=1.5)
    ax.axvline(x=0,color='#00ff00',linestyle='-',linewidth=1.5,alpha=0.8)
    ax.legend()
    ax.set_xlabel('fit residuals')

class Plotter(PlotterBackbone):
    def __init__(self,args):
        PlotterBackbone.__init__(self,args)

            
#...!...!..................
    def W_and_eigen(self,bigD,md,tf, figId=3):
        figId=self.smart_append(figId)        
        nrow,ncol=1,2
        fig=self.plt.figure(figId,facecolor='white', figsize=(10,5))

        if tf==True:  # Dale-truth
            A0=bigD['true_network_matrix'].astype(np.float32)
            W=np.copy(A0).T
            tit='True daleM, M%d, %s'%(W.shape[0],md['selector']['input_name'])
            tit2='True eigen values'
        else: # UoI fit
            W=bigD['fit_W_matrix'].T
            tit='Fit daleM, M%d, %s'%(W.shape[0],md['selector']['input_name'])
            tit2='Fit eigen values'
        
        #.... left ......
        ax = self.plt.subplot(nrow,ncol,1)
        plot_dale_matrix(fig,ax,W)
        ax.set(title=tit)

        #..... right......
        ax = self.plt.subplot(nrow,ncol,2)
        Eigen=np.linalg.eigvals(W)
        plot_dale_eigen(fig,ax,Eigen,tit=tit2)
       

        
    def Aper_row(self,bigD,md,tf,figId=3):
        figId=self.smart_append(figId)
        nrow,ncol=2,1
        
        if tf==True:  # Dale-truth
            A0=bigD['true_network_matrix']
            W=np.copy(A0)
            tit='True daleM, M%d, %s'%(W.shape[0],md['selector']['input_name'])
        else: # UoI fit
            W=bigD['fit_W_matrix']
            tit='Fit daleM, M%d, %s'%(W.shape[0],md['selector']['input_name'])
            
        ax=plot_diagonal_and_violins(W,self.plt,figId,tit)
        txt=summary_column(md)
        ax.text(0.6,0.95,txt,fontsize=10,color='blue',ha='left',va='top',transform=ax.transAxes)
        
    def weigh_correl(self,bigD,md,figId=4):
        fim=md['fit_uoi']
        pof=md['post_fit_residuals']
        mxs=md['dale_truth']['5index']
        pmd=md['dataset']
        sem=md['selector']
        dmm=md['dale_truth']
                
        figId=self.smart_append(figId)        
        nrow,ncol=2,3
        fig=self.plt.figure(figId,facecolor='white',figsize=(12,7))

        zEps=1e-4
        print('PWC: zEps=%.1e'%(zEps))
        print('PWC: dale partition size: diag:%d  exc:%d  zexc:%d  inh:%d  zinh:%d'%(len(bigD['post_Ydiag']), len(bigD['post_Yexc']), len(bigD['post_Yzexc']), len(bigD['post_Yinh']), len(bigD['post_Yzinh'])))
        # Diagonal elements
        ax=self.plt.subplot(nrow,ncol,1)
        Ydiag=bigD['post_Ydiag']  # fit values
        ResDia=Ydiag[:,0]-Ydiag[:,1]
        #Rdia1=bigD['post_Rdia1'] #  regressed fit by rotation
        #Rdia2=bigD['post_Rdia2'] # residuals after rotation
        #Ydia[:,1]=Rdia1
        
        dCol='darkorange'
        obsN='diagonal'
        draw_correlation_plot(ax,Ydiag,pof['diag'],dCol)
        ax.set(xlabel='true '+obsN,title= '%d neurons, UoI=%s'%(dmm['num_any_neur'],md['short_name']))
                       
        ax=self.plt.subplot(nrow,ncol,4)
        tit2='%d neur, %s residuals'%(dmm['num_any_neur'],obsN)
        ax.set_title(tit2)
        add_histogram(ax,ResDia,'diag',dCol)

        # Excitatory weights
        ax=self.plt.subplot(nrow,ncol,2)
        Yexc=bigD['post_Yexc']
        ResExc=Yexc[:,0]-Yexc[:,1]
        #Rexc=bigD['post_Rexc']
        Yzexc=bigD['post_Yzexc']
        dCol='darkred'
        obsN='excitatory '
        draw_correlation_plot(ax,Yexc,pof['exc'],dCol)
        ax.set(xlabel='true '+obsN,title= 'input %s'%(fim['data_name']))
        ax.axhline(0, color='k', linestyle='--', lw=0.8)
        ax.axvline(0, color='k', linestyle='--', lw=0.8)
        
        ax=self.plt.subplot(nrow,ncol,5)
        tit2='UoI=%s, %s residuals'%(md['short_name'],obsN)
        ax.set_title(tit2)        
        
        add_histogram(ax,ResExc,'exc',dCol)
        add_histogram(ax,Yzexc,'zexc','dimgray')

        # Inhibitory weights
        ax=self.plt.subplot(nrow,ncol,3)
        Yinh=bigD['post_Yinh']
        ResInh=Yinh[:,0]-Yinh[:,1]
        #Rinh=bigD['post_Rinh']
        Yzinh=bigD['post_Yzinh']
        dCol='blue'
        obsN='inhibitory'
        draw_correlation_plot(ax,Yinh,pof['inh'],dCol)
        ax.set(xlabel='true '+obsN,title= 'input [nT,nF]=%s'%(fim['data_shape']))
        ax.axhline(0, color='k', linestyle='--', lw=0.8)
                
        ax=self.plt.subplot(nrow,ncol,6)
        tit2=' %s residuals'%(obsN)
        ax.set_title(tit2)        
      
        add_histogram(ax,ResInh,'inh',dCol)
        add_histogram(ax,Yzinh,'zinh','dimgray')

      
        print('Num of non zero-values in Yzexc: %d of %d -->frac=%.2f'%( Yzexc.size,mxs['exc_zero'],Yzexc.size/mxs['exc_zero']))
        print('Num of non zero-values in Yzinh: %d of %d -->frac=%.2f'%( Yzinh.size,mxs['inh_zero'],Yzinh.size/mxs['inh_zero']))
        print('Num of zero-values in Yexc: %d of %d -->frac=%.2f'%(np.sum(np.abs(Yexc[:,1])<=zEps), Yexc.shape[0],np.sum(np.abs(Yexc[:,1])<=zEps)/Yexc.shape[0]))
        print('Num of zero-values in Yinh: %d of %d -->frac=%.2f'%(np.sum(np.abs(Yinh[:,1])<=zEps), Yinh.shape[0],np.sum(np.abs(Yinh[:,1])<=zEps)/Yinh.shape[0]))
        #pprint(mxs)

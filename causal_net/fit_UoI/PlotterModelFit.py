#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
from matplotlib import cm as cmap
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from pprint import pprint

from toolbox.PlotterBackbone import PlotterBackbone

def summary_column(md):
    pmd=md['payload']
    sem=md['selector']
    txt=md['short_name']
    txt+='\ninput '+sem['input_name']
    txt+='\nsampFreq %d Hz'%(sem['sampling_freq'])
    txt+='\ndecay:%d ms '%(pmd['tau_response']*1000.)
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
    ax1.set_title('%s   Auto-correlation'%(tit0))

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

def draw_correlation_plot(ax,XY,stD,tit,dCol):
    ax.scatter(XY[:,0],XY[:,1],facecolors='none',edgecolors=dCol,label='all')
    ax.plot(stD['mu_X'],stD['mu_Y'],'+',color='#00ff00',markersize=20)     
    ax.text(0.1,0.9,'Correl=%.2f'%(stD['rho']),transform=ax.transAxes)
    ax.set_xlabel('true')
    ax.set_ylabel('UoI ADMM fit')
    ax.set_title(tit)

def add_histogram(ax,data,dLab0,dCol):
    stdX=np.std(data)
    dLab='%s std=%.3f'%(dLab0,stdX)
    ax.hist(data,bins=30,color=dCol,histtype='step',alpha=0.7,label=dLab)
    ax.axvline(x=0,color='#00ff00',linestyle='-',linewidth=1.5,alpha=0.8)
    ax.legend()
    ax.set_xlabel('fit residuals')

class Plotter(PlotterBackbone):
    def __init__(self,args):
        PlotterBackbone.__init__(self,args)

    def A_matrix(self,bigD,md,figId=3,lag=0):
        figId=self.smart_append(figId)        
        fig=self.plt.figure(figId,facecolor='white',figsize=(8,7))
        ax=self.plt.subplot(1,1,1)

        if lag>=0:
            A0=bigD['fit_A_model'][lag]
            A=A0-np.eye(A0.shape[0])
            tit='UoI fit matrix, %d neurons, name=%s'%(A0.shape[0],md['short_name'])
        else:
            A0=bigD['true_network_matrix']
            A=np.copy(A0)
            tit='Dale true matrix, %d neurons, name=%s'%(A0.shape[0],md['selector']['input_name'])
        
        nfeat=A.shape[0]
        max_val=np.max(np.abs(A))/2.
        
        im=ax.imshow(A.T,aspect='auto',origin='lower',cmap='bwr',vmin=-max_val,vmax=max_val)
        ax.grid()
        ax.plot([0,nfeat],[0,nfeat],'--',lw=0.5)
        ax.set_xlim(-0.5,nfeat+0.5)
        ax.set_ylim(-0.5,nfeat+0.5)
        ax.set_aspect(1.0)

        cbar=fig.colorbar(im,ax=ax,extend="both")
        cbar.set_label('UoI coupling strength')
        ax.set(title=tit,xlabel='presyn. node index, source',ylabel='postsyn. node index, target')
        
    def Aper_row(self,bigD,md,figId=3,lag=0):
        figId=self.smart_append(figId)
        
        if lag>=0:
            A0=bigD['fit_A_model'][lag]
            A=A0-np.eye(A0.shape[0])
            tit='UoI fit matrix, %d neurons, name=%s'%(A0.shape[0],md['short_name'])
        else:
            A0=bigD['true_network_matrix']
            A=np.copy(A0)
            tit='Dale true matrix, %d neurons, name=%s'%(A0.shape[0],md['selector']['input_name'])

        ax=plot_diagonal_and_violins(A,self.plt,figId,tit)
        txt=summary_column(md)
        ax.text(0.6,0.95,txt,fontsize=10,color='blue',ha='left',va='top',transform=ax.transAxes)
        
    def weigh_correl(self,bigD,md,figId=4):
        pof=md['post_fit_residual']
        lag=0
        figId=self.smart_append(figId)        
        fig=self.plt.figure(figId,facecolor='white',figsize=(12,7))

        # Diagonal elements
        ax=self.plt.subplot(2,3,1)
        Ydia=bigD['post_Ydia']
        Rdia=bigD['post_Rdia']
        draw_correlation_plot(ax,Ydia,pof['diag'],'diagonal elements','blue')
        
        ax=self.plt.subplot(2,3,4)        
        add_histogram(ax,Rdia,'diag','blue')

        # Excitatory weights
        ax=self.plt.subplot(2,3,2)
        Yexc=bigD['post_Yexc']
        Rexc=bigD['post_Rexc']
        Yzexc=bigD['post_Yzexc']
        draw_correlation_plot(ax,Yexc,pof['exc'],'Excitatory weights','darkorange')
        
        ax=self.plt.subplot(2,3,5)        
        add_histogram(ax,Rexc,'true exc','darkorange')
        add_histogram(ax,Yzexc,'zero exc','dimgray')

        # Inhibitory weights
        ax=self.plt.subplot(2,3,3)
        Yinh=bigD['post_Yinh']
        Rinh=bigD['post_Rinh']
        Yzinh=bigD['post_Yzinh']
        draw_correlation_plot(ax,Yinh,pof['inh'],'Inhibitory weights','darkred')
        
        ax=self.plt.subplot(2,3,6)        
        add_histogram(ax,Rinh,'true inh','darkred')
        add_histogram(ax,Yzinh,'zero inh','dimgray')
    

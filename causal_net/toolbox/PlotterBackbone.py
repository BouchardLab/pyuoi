import os
import numpy as np

import time
#from QubicUtil import repack1D_cent2edge, measureTime_to_string

#...!...!..................
def roys_fontset(plt):
    print('load Roys fontest')
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    #plt.rcParams['text.usetex'] = True  #Needs new Docker image

    tick_major = 6
    tick_minor = 4
    plt.rcParams["xtick.major.size"] = tick_major
    plt.rcParams["xtick.minor.size"] = tick_minor
    plt.rcParams["ytick.major.size"] = tick_major
    plt.rcParams["ytick.minor.size"] = tick_minor

    font_small = 12
    font_medium = 13
    font_large = 14
    plt.rc('font', size=font_small)          # controls default text sizes
    plt.rc('axes', titlesize=font_medium)    # fontsize of the axes title
    plt.rc('axes', labelsize=font_medium)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_small)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_small)    # fontsize of the tick labels
    
    plt.rc('figure', titlesize=font_large)   # fontsize of the figure title

    # legend box
    plt.rc('legend', frameon=False)  # remove it the frame
    plt.rc('legend', fontsize=font_small)    # legend fontsize
    
#............................
#............................
#............................
class PlotterBackbone(object):
    def __init__(self, args):
        self.jobName=args.prjName
        try:
            self.venue=args.formatVenue
        except:
            self.venue='prod'
        
        import matplotlib as mpl
        if args.noXterm:
            if args.verb>0: print('disable Xterm')
            mpl.use('Agg')  # to plot w/o X-server
        else:
            mpl.use('TkAgg')
        import matplotlib.pyplot as plt

        if args.verb>0: print(self.__class__.__name__,':','Graphics started')
        plt.close('all')
        self.plt=plt
        self.args=args
        self.figL=[]
        self.outPath=args.outPath+'/'
        assert os.path.exists(self.outPath)
        if 'paper' in self.venue:
            roys_fontset(plt)

    #............................
    def figId2name(self, fid):
        figName='%s_f%d'%(self.jobName,fid)
        return figName

    #............................
    def clear(self):
        self.figL=[]
        self.plt.close('all')
    #............................
    def display_all(self, png=1):
        if len(self.figL)<=0: 
            print('display_all - nothing top plot, quit')
            return
        
        for fid in self.figL:
            self.plt.figure(fid)
            self.plt.tight_layout()
            figName=self.outPath+self.figId2name(fid)
            if png: figName+='.png'
            else: figName+='.pdf'
            print('Graphics saving to ',figName)
            self.plt.savefig(figName)
        self.plt.show()

# figId=self.smart_append(figId)
#...!...!....................
    def smart_append(self,id): # increment id if re-used
        while id in self.figL: id+=1
        self.figL.append(id)
        return id

#............................
    def blank_share2D(self,nrow=2,ncol=2, figsize=(6,6),figId=10):
        figId=self.smart_append(figId)
        kwargs={'num':figId,'facecolor':'white', 'figsize':figsize}
        fig, axs = self.plt.subplots(nrow,ncol, sharex='col', sharey='row',
                                     gridspec_kw={'hspace': 0, 'wspace': 0}, **kwargs)
        return axs
   
#............................
    def blank_separate2D(self,nrow=2,ncol=2, figsize=(6,6),figId=10):
        figId=self.smart_append(figId)
        kwargs={'num':figId,'facecolor':'white', 'figsize':figsize}
        fig, axs = self.plt.subplots(nrow,ncol, **kwargs)
        return axs
   

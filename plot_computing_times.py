from numpy import *
from pylab import *
import h5py
import os,pdb
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit


mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10

def spine_selector(ax,locs=['left','bottom'],out_points=10,lw=1):
    for loc, spine in ax.spines.iteritems():
        if loc in locs:
            spine.set_position(('outward',out_points))
            spine.set_smart_bounds(True)
            spine.set_linewidth(lw)
        else:
            spine.set_color('none')
    if 'bottom' in locs:
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.xaxis.set_ticks_position('none')
    if 'left' in locs:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks_position('none')

def bigger_ticklines(ax,ms=8,mew=1.35,direction='out',pad=10,color='k'):
    for tl in ax.yaxis.get_ticklines():
        tl.set_markersize(ms)
        tl.set_markeredgewidth(mew)
        tl.set_color(color)
    for tl in ax.xaxis.get_ticklines():
        tl.set_markersize(ms)
        tl.set_markeredgewidth(mew)
    ax.tick_params(direction=direction,pad=pad)


class FixedOrderFormatter(mpl.ticker.ScalarFormatter):
    """Formats axis ticks using scientific notation with a constant order of 
    magnitude"""
    def __init__(self, order_of_mag=0, useOffset=True, useMathText=True):
        self._order_of_mag = order_of_mag
        mpl.ticker.ScalarFormatter.__init__(self, useOffset=useOffset, 
                                 useMathText=useMathText)
    def _set_orderOfMagnitude(self, range):
        """Over-riding this to avoid having orderOfMagnitude reset elsewhere"""
        self.orderOfMagnitude = self._order_of_mag


experiment = 'size'
machine = 'cori'


if experiment=='size':
    #n = [10,100,1000]
    n = [8,10,22,32,72,100,224,318,708,1000,2236]
    m = [10]*len(n)
    l = [10000]*len(n)
    labels = ['0.4','0.8','4','8','40','80','400','800','4000','8000','40000']
elif experiment=='samples':
    n = [10,10,10]
    m = [10,100,1000]
    l = [10000,10000,10000]
    labels = ['1e2','1e3','1e4']
elif experiment=='params':
    n = [1000,100,10]
    m = [1,10,100]
    l = [10000,10000,10000]
    labels = ['1e3','1e4','1e5']

save = False 
if save:
    figurePath ='/global/project/projectdirs/m2043/BRAINgraph/results/scalability_test/figures/%s/%s'%(machine,experiment)
    if not os.path.exists(figurePath):
        os.makedirs(figurePath)

resultsPath = '/global/project/projectdirs/m2043/BRAINgraph/results/scalability_test'

res = zeros((2,2,len(n)))
w = ['w','wo']
k = ['serial','distributed']

for i in xrange(2):
    for ii in xrange(2):
        for iii in xrange(len(n)):
            try:
                if i==1:
                    fname='MPI_BoLBO'
                else:
                    fname = 'BoLBO'
                filename = '%s/%s/%s/%s/%s_ExpI_%i_0.0_%i.0_%i.0_f8.h5'%(resultsPath,k[i],experiment,machine,fname,n[iii],l[iii],m[iii])
                with h5py.File(filename,'r') as f:
                    if ii==0:
                        if iii<=7 or i==1:
                            res[i,ii,iii]=f.attrs['compTime']
                        else:
                            res[i,ii,iii]= 10*48*f.attrs['las1Time']+\
                                           48**2*f.attrs['las2Time']+\
                                           10*48**2*f.attrs['bolsTime']
                    elif ii==1:
                        res[i,ii,iii]=f.attrs['loadTime']+f.attrs['saveTime']
            except:
                print '\n%s could not be loaded!'%filename
                pass

print res[0,0]/res[1,0]

col = np.linspace(.25,1,len(n))

fig =figure('logplot',figsize=(4,4))
fig.clf()
ax = fig.add_axes([.15,.15,.8,.8])

ax.loglog(logspace(-1,8,10),logspace(-1,8,10),color='gray',alpha=.8)

def func(x,m,b):
    return 10**(m*np.log10(x)+b)

m,b = curve_fit(func,res[1,0,:7],res[0,0,:7])[0]

print "\nSlope:\t%.4f\t\tIntercept:\t%.4f"%(m,b)

ax.loglog(res[1,0,:-1],func(res[1,0,:-1],m,b), 'r',lw=1.5,alpha=.2)

marker = ['s','s','s','s','s','s','s','s','o','o','o']
mec = 10*['k']+['r']
for i in xrange(len(n)):
    c = cm.binary(col[i])
    loglog(res[1,0,i],res[0,0,i],marker[i],mfc=c,mec=mec[i],mew=1)

#legend(labels,loc='lower right',frameon=False)
labels = ['4e2','8e2','4e3','8e3','4e4','8e4','4e5','8e5','4e6','8e6','4e7']

axCB = plt.axes([.45,.3,.4,.04])
axCB.imshow(np.vstack((col,col)),aspect='auto',interpolation='nearest',cmap=cm.binary,vmin=0,vmax=1)
axCB.set_yticks([])
axCB.set_xticks(np.arange(len(n)))
#axCB.set_xticklabels(['1e2','','1e3','','1e4','','1e5','','1e5','','1e7'],fontsize=6)
axCB.set_xticklabels([])


axCB.set_xlabel('Data size [bytes]',fontsize=10)


spine_selector(axCB,['bottom'],out_points=0)
bigger_ticklines(axCB,ms=2,mew=1,direction='in',pad=1)

ax.set_ylabel('Time [s] - serial',labelpad=0,fontsize=10)
ax.set_xlabel('Time [s] - distributed',labelpad=0,fontsize=10)


spine_selector(ax,out_points=4)
bigger_ticklines(ax,ms=2,mew=1,direction='out',pad=1)
ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())

if save:
    for p in ['png','eps','pdf']:
        savefig('%s/bolbo_MPI_serial_comp_size.%s'%(figurePath,p),dpi=300)
else:
    show()

fig2 = plt.figure('bar_plot_dist',figsize=(4,4))
fig2.clf()
ax2 = fig2.add_axes([.15,.15,.8,.8])

p1 = ax2.bar(np.arange(1,len(n)+1), res[1,1], .35, color='r',edgecolor='none')
p2 = ax2.bar(np.arange(1,len(n)), res[1,0,:-1], .35, color='pink',edgecolor='none',
             bottom=res[1,1,:-1])

ax2.bar(11, res[1,0,-1], .35, color='white',edgecolor='pink',
             bottom=res[1,1,-1],hatch='//')
             
p1 = ax2.bar(np.arange(1,len(n)+1)+.37, res[0,1], .35, color='k',edgecolor='none')
p2 = ax2.bar(np.arange(1,len(n)-2)+.37, res[0,0,:-3], .35, color='gray',edgecolor='none',
             bottom=res[0,1,:-3])

ax2.bar(np.arange(9,len(n)+1)+.37, res[0,0,-3:], .35, color='white',edgecolor='gray',
             bottom=res[0,1,-3:],hatch='//')


plt.xticks(np.arange(1,len(n)+1)+.36,labels,fontsize=6)
ax2.set_yscale('log')
ax2.legend((p1[0], p2[0]), ('IO', 'compute'),loc='best',frameon=False,fontsize=8)
ax2.set_xlabel('Data size [bytes]',labelpad=0,fontsize=10)
ax2.set_ylabel('Time [s]',labelpad=0,fontsize=10)
ax2.set_xlim(0.8,len(n)+1+.2)
ax2.set_ylim(1e-1,1e8)

spine_selector(ax2,out_points=4)
bigger_ticklines(ax2,ms=2,mew=1,direction='out',pad=1)
ax2.yaxis.set_minor_locator(mpl.ticker.NullLocator())

if save:
    for p in ['png','eps','pdf']:
        savefig('%s/bolbo_serial_iocomp_size.%s'%(figurePath,p),dpi=300)
else:
    show()





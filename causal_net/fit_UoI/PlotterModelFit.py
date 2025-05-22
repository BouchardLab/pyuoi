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
    sem=md['selector']
    
    txt=md['short_name']
    txt+='\ninput '+sem['input_name']
    txt+='\nsampFreq %d Hz'%(sem['sampling_freq'])
    txt+='\ndecay:%d ms '%(pmd['tau_response']*1000.)
    txt+='\nsel time [%.1f %.1f] s'%(sem['time_range'][0],sem['time_range'][1])
    txt+='\nsel features %d'%(sem['num_feature'])
    
    return txt

    
#...!...!....................
def plot_diagonal_and_violins(A,plt,figId,tit0, eps=1e-5):
    """
    Plots the diagonal elements of the matrix A as a 1D line plot (top row) and 
    vertical violin plots for each row (bottom row), excluding diagonal elements 
    and values with amplitude smaller than eps.

    Parameters:
        A (2D sparse matrix or array): Sparse matrix of size (nfeat, nfeat).
        eps (float): Minimum value threshold to include in violin plots.
    """
    nfeat = A.shape[0]

    # Create the canvas with two rows
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [1, 2]},   num=figId)
    
    
    # Top Row: Plotting the diagonal elements
    diagV = np.diag(A)
    ax1.plot(diagV, marker='o', linestyle='-', color='blue', label='Diagonal Elements')
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Diagonal Value')
    ax1.set_title('Diagonal Elements of A')
    ax1.grid(True)
  
    ax1.set_xlim(-0.5,nfeat+0.5)
    ax1.set_title('%s   Auto-correlation' % (tit0))

    # Customizing x-axis
    tickL=5
    x_ticks = np.arange(0, nfeat, tickL)
    ax1.set_xticks(x_ticks)
    
    # Bottom Row: Vertical violin plots
    violin_data = []
    for i in range(nfeat):
        # Exclude diagonal and values with amplitude smaller than eps
        row_values = np.delete(A[i, :], i)  # Exclude diagonal element
        row_values = row_values[np.abs(row_values) >= eps]  # Filter values by amplitude
        if len(row_values) > 0:
            violin_data.append(row_values)
        else:
            violin_data.append([0])  # Add a placeholder to maintain alignment

    # Plotting vertical violin plots
   
    ax2.violinplot(violin_data, positions=np.arange(nfeat) , showmeans=True, showmedians=True)
    ax2.set_xlim(-0.5, nfeat + 0.5)
    ax2.set_xticks(x_ticks )
    
    # Customize the x-axis and labels
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('Off-diagonal Values')
    ax2.set_title('Vertical Violin Plots for Each Row of A')
    ax2.grid(True)
    
    ax2.axhline(0,lw=1.,ls='--',c='k')
    return ax2

 
#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)

#...!...!..................
    def A_matrix(self,bigD,md,figId=3,lag=0):

        pmd=md['payload']
       
        figId=self.smart_append(figId)        
        nrow,ncol=1,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,7))
        ax = self.plt.subplot(nrow,ncol,1)

        if lag >=0:
            A0=bigD['fit_A_model'][lag]
            A = A0 - np.eye(A0.shape[0])  # subtract 1 from diagonal elements
            tit='UoI fit matrix, %d neurons, name=%s'%(A0.shape[0],md['short_name'])
        else:
            A0=bigD['true_network_matrix']
            A=np.copy(A0)
            tit='Dale true matrix, %d neurons, name=%s'%(A0.shape[0],md['selector']['input_name'])
        nfeat = A.shape[0]
        
        # Set diagonal values of A to 0
        #np.fill_diagonal(A, 0)
                
        # Find the maximum absolute value in A
        max_val = np.max(np.abs(A))/2.
        
        # Plot the sparse matrix using imshow
        im = ax.imshow(A.T, aspect='auto', origin='lower', cmap='bwr', vmin=-max_val, vmax=max_val)
        ax.grid()
        ax.plot([0,nfeat],[0,nfeat],'--',lw=0.5)
        ax.set_xlim(-0.5,nfeat+0.5)
        ax.set_ylim(-0.5,nfeat+0.5)
        ax.set_aspect(1.0)

        # Create the colorbar.
        cbar = fig.colorbar(im, ax=ax, extend="both")
        cbar.set_label('UoI coupling strength')
        
        
        ax.set(title=tit, xlabel='presyn. node index, source', ylabel='postsyn. node index, target')
        
#...!...!..................
    def Aper_row(self,bigD,md,figId=3,lag=0):
        figId=self.smart_append(figId)
        pmd=md['payload']
    
        if lag >=0:
            A0=bigD['fit_A_model'][lag]
            A = A0 - np.eye(A0.shape[0])  # subtract 1 from diagonal elements           
            tit='UoI fit matrix, %d neurons, name=%s'%(A0.shape[0],md['short_name'])
        else:
            A0=bigD['true_network_matrix']
            A=np.copy(A0)
            tit='Dale true matrix, %d neurons, name=%s'%(A0.shape[0],md['selector']['input_name'])


        ax=plot_diagonal_and_violins( A,self.plt,figId,tit)
        
        txt=summary_column(md)
        ax.text(0.6, 0.95, txt, fontsize=10, color='blue', ha='left', va='top',transform=ax.transAxes)
        
#...!...!..................
    def weigh_correl(self,bigD,md,figId=4):

        pmd=md['payload']
        lag=0
        Af=bigD['fit_A_model'][lag].T
        At=bigD['true_network_matrix'].T

        figId=self.smart_append(figId)        
        nrow,ncol=2,3
        fig=self.plt.figure(figId,facecolor='white', figsize=(12,7))

        # -----  TOP ROW -----
        # .... diagonal
        ax = self.plt.subplot(nrow,ncol,1)
        Df= np.diag(Af)[1:];        Dt= np.diag(At)[1:]  # skip (0,0) element for UoI-ADMM
        Dres=draw_correlation_plot(ax,Dt, Df,'diagonal elements')

        #... Excitatory
        MEf=get_non_diagonal_elements(Af, position='first')
        MEt=get_non_diagonal_elements(At, position='first')
        mask=MEt!=0
        MEn=MEf[~mask]; MEf=MEf[mask]; MEt=MEt[mask]
        
        ax = self.plt.subplot(nrow,ncol,2)
        MEres=draw_correlation_plot(ax,MEt, MEf,'Excitatory weights')
        ax.axhline(0,linestyle='--',c='k')
        
        #... inhibitory
        MIf=get_non_diagonal_elements(Af, position='last')
        MIt=get_non_diagonal_elements(At, position='last')
        mask=MIt!=0
        MIf=MIf[mask] ;  MIt=MIt[mask]
        ax = self.plt.subplot(nrow,ncol,3)
        MIres=draw_correlation_plot(ax,MIt, MIf,'Inhibitory weights')
        ax.axhline(0,linestyle='--',c='k')
         

        # -----  BOTTOM ROW -----
        # .... diagonal
        ax = self.plt.subplot(nrow,ncol,4)
        plot_histogram(ax,Dres)

        #... Excitatory
        ax = self.plt.subplot(nrow,ncol,5)
        plot_histogram(ax,MEres)
        #plot_histogram(ax,MEn) 

        #... inhibitory
        ax = self.plt.subplot(nrow,ncol,6)
        plot_histogram(ax,MIres)
        
def calc_stats_and_whiten(A: np.ndarray, B: np.ndarray):
    """
    Compute
      • mean of A  (μ_A)
      • mean of B  (μ_B)
      • Pearson correlation ρ(A,B)
      • demeaned-and-rotated data  (A′, B′)

    The returned (A′, B′) satisfy
      – their means are zero, and
      – the horizontal axis (A′) is the direction of maximal variance,
        so the vertical variance (along B′) is minimal.

    Parameters
    ----------
    A, B : 1-D numpy arrays of equal length.

    Returns
    -------
    μ_A, μ_B, ρ, A_prime, B_prime
    """
    # 1. basic statistics
    μ_A = A.mean()
    μ_B = B.mean()
    ρ   = np.corrcoef(A, B)[0, 1]

    # 2. centre the data
    X = np.stack([A - μ_A, B - μ_B], axis=0)      # shape (2, N)

    # 3. rotate so that the 1st principal component is horizontal
    C = np.cov(X)                                 # 2×2 covariance
    eigvals, eigvecs = np.linalg.eigh(C)          # sorted ascending
    v_max = eigvecs[:, eigvals.argmax()]          # eigen-vector of max var.
    θ = np.arctan2(v_max[1], v_max[0])#+np.pi            # angle of that vector

    R = np.array([[ np.cos(-θ), -np.sin(-θ)],     # negative θ → align horiz.
                  [ np.sin(-θ),  np.cos(-θ)]])

    X_rot = R @ X                                 # 2×N  rotated
    A_prime, B_prime = X_rot                      # unpack

    return μ_A, μ_B, ρ, A_prime, B_prime

#...!...!..................
def draw_correlation_plot(ax,A, B,tit):
    muA,muB,rho,Ar,Br=calc_stats_and_whiten(A, B)
    
    ax.scatter(A, B, facecolors='none', edgecolors='b', label='all')     
    ax.plot(muA, muB, 'r+', markersize=20)     
    ax.text(0.1, 0.9, 'Correl=%.2f'%( rho), transform=ax.transAxes)
    
    ax.set_xlabel('true')
    ax.set_ylabel('UoI ADMM fit')
    ax.set_title(tit)
    return Br
                       
       

#...!...!..................
def filter_outliers(A, B, eps=0.05):
    # Find the median of vector B
    median_B = np.median(B)
    percL,percH= eps*100,100 - eps*100
    print('percentiles:',percL,percH)
    print('median_B',median_B,B)
    # Find values in B that are eps% away from the median
    thrL = np.percentile(B, percL)
    thrH = np.percentile(B, percH)
    print('filter_outliers  bounds:',thrL,thrH)
    # Filter out the outlier pairs from A and B
    mask = np.logical_and(B >= thrL, B <= thrH)
    A_filtered = A[mask]
    B_filtered = B[mask]
    
    return A_filtered, B_filtered

#...!...!..................
def get_non_diagonal_elements(C, position='first'):
    """
    Extracts all elements from either the first N or last N columns of matrix C
    (size 2N x 2N), excluding the diagonal elements in those columns,
    and returns them as a flattened 1D array.

    Parameters:
    C (np.ndarray): Square matrix of size 2N x 2N.
    position (str): 'first' for first N columns, 'last' for last N columns.

    Returns:
    np.ndarray: 1D array of selected elements.
    """
    # Verify input dimensions
    rows, cols = C.shape
    if rows != cols or rows % 2 != 0:
        raise ValueError("Input matrix must be square with even dimensions (2N x 2N).")
    
    N = rows // 2

    elements = []
    if position == 'first':
        col_range = range(N)
    elif position == 'last':
        col_range = range(N, 2 * N)
    else:
        raise ValueError("Invalid position value. Use 'first' or 'last'.")

    for i in range(2 * N):
        for j in col_range:
            if i != j:
                elements.append(C[i, j])
    return np.array(elements)

#...!...!..................
def plot_histogram(ax, data, percentile_low=30, percentile_high=70):
    """Plot histogram of the difference and annotate mean, median, and percentiles."""
    
    ax.hist(data, bins=30, color='salmon', alpha=0.7)

    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    
    # Compute standard error of the standard deviation estimator
    N = data.shape[0] 
    se_s = std / np.sqrt(2 * (N - 1))

    # Compute percentiles
    p_low = np.percentile(data, percentile_low)
    p_high = np.percentile(data, percentile_high)
    
    
    # Annotate statistics
    txt = f"Mean: {mean:.3f}\nMedian: {median:.3f}\nRMSE: {std:.3f} ± {se_s:.3f}\n"

    ax.annotate(txt, xy=(0.35, 0.75), color='black', xycoords='axes fraction')
    ax.set_xlabel('fit residuals')
    #ax.legend()


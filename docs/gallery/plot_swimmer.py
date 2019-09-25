"""
.. _swimmer:

UoI-NMF for parts based decomposition
=====================================

This example will demonstrate parts-based decomposition with UoI-NMF on the swimmer dataset.
The swimmer dataset is the canonical example of separable data.

"""

####################
# Swimmer dataset
# ---------------
#
#


import matplotlib.pyplot as plt

from datetime import datetime
import warnings
import numpy as np

from sklearn.preprocessing import minmax_scale
#from sklearn.exceptions import ConvergenceWarning
from sklearn.manifold import TSNE

from pyuoi.decomposition import UoI_NMF
from pyuoi.datasets import load_swimmer

np.random.seed(10)

Swimmers = load_swimmer()
Swimmers = minmax_scale(Swimmers, axis=1)

import matplotlib
matplotlib.rcParams['figure.figsize'] = [4, 4]


####################
# Original Swimmer samples
# =========================

fig, ax = plt.subplots(4, 4, subplot_kw={'xticks': [], 'yticks': []})
indices = np.random.randint(16, size=16) + np.arange(0, 256, 16)
ax = ax.flatten()
for i in range(len(indices)):
    ax[i].imshow(Swimmers[indices[i]].reshape(32,32).T, aspect='auto', cmap='gray')


####################
# Swimmer samples corrupted with Absolute Gaussian noise
# ======================================================
#
# Corrupt samples with with absolute Gaussian noise drawn from a Normal(0, 0.25)


reps = 1
corrupted = np.zeros((Swimmers.shape[0]*reps, Swimmers.shape[1]))
for r in range(reps):
    corrupted[r*Swimmers.shape[0]:(r+1)*Swimmers.shape[0]] = Swimmers + np.abs(np.random.normal(scale=0.25, size=Swimmers.shape))

fig, ax = plt.subplots(4, 4, subplot_kw={'xticks': [], 'yticks': []})
ax = ax.flatten()
for i in range(len(indices)):
    ax[i].imshow(corrupted[indices[i]].reshape(32,32).T, aspect='auto', cmap='gray')

####################
# Run UoI NMF on corrupted Swimmer data
# =====================================
#
# Twenty bootstraps should be enough. Min_pts should be half of the bootstraps
#

nboot = 20
min_pts = nboot/2
ranks = [16]

shape = corrupted.shape

uoinmf = UoI_NMF(n_boots=nboot, ranks=ranks,
                db_min_samples=min_pts,
                nmf_max_iter=400,)

uoinmf.cons_meth = np.mean   # intersect and mean give too much noise

transformed = uoinmf.fit_transform(corrupted)
recovered = transformed @ uoinmf.components_

####################
# NMF Swimmer bases
# =================

order = np.argsort(np.sum(uoinmf.components_, axis=1))

fig, ax = plt.subplots(4, 4, subplot_kw={'xticks': [], 'yticks': []})
ax = ax.flatten()
for i in range(uoinmf.components_.shape[0]):
    ax[i].imshow(uoinmf.components_[order[i]].reshape(32,32).T, aspect='auto', cmap='gray')


####################
# Recovered Swimmers
# ==================


fig, ax = plt.subplots(4, 4, subplot_kw={'xticks': [], 'yticks': []})
ax = ax.flatten()
for i in range(len(indices)):
    ax[i].imshow(recovered[indices[i]].reshape(32,32).T, aspect='auto', cmap='gray')


################################################################
# Plot them all together so we can see how well we recovered
# the original swimmer data.
#
#


fig, ax = plt.subplots(3, 16, figsize=(27, 5), subplot_kw={'xticks': [], 'yticks': []})
indices = np.random.randint(16, size=16) + np.arange(0, 256, 16)
ax = ax.flatten()

# plot Original
ax[0].set_ylabel('Original', rotation=0, fontsize=25, labelpad=40)
ax[0].yaxis.set_label_coords(-1.0, 0.5)
for i in range(len(indices)):
    ax[i].imshow(Swimmers[indices[i]].reshape(32,32).T, aspect='auto', cmap='gray')

# plot Corrupted
ax[16].set_ylabel('Corrupted', rotation=0, fontsize=25, labelpad=40)
ax[16].yaxis.set_label_coords(-1.1, 0.5)
for i in range(len(indices)):
    ax[16+i].imshow(corrupted[indices[i]].reshape(32,32).T, aspect='auto', cmap='gray')

# plot Recovered
ax[32].set_ylabel('Recovered', rotation=0, fontsize=25, labelpad=40)
ax[32].yaxis.set_label_coords(-1.1, 0.5)
for i in range(len(indices)):
    ax[32+i].imshow(recovered[indices[i]].reshape(32,32).T, aspect='auto', cmap='gray')

################################################################
#
# To see what DBSCAN is doing, lets look at a the bases samples:
#
#

plt.figure()
embedding = TSNE(n_components=2).fit_transform(uoinmf.bases_samples_)
sc = plt.scatter(embedding[:,0], embedding[:,1], c=uoinmf.bases_samples_labels_, s=80, cmap="nipy_spectral")
sc.set_facecolor('none')

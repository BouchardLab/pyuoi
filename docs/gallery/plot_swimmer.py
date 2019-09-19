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
from sklearn.exceptions import ConvergenceWarning
from sklearn.manifold import TSNE

from umap import UMAP

from pyuoi.decomposition import UoI_NMF
from pyuoi.datasets import load_swimmer

np.random.seed(10)

Swimmers = load_swimmer()
Swimmers = minmax_scale(Swimmers, axis=1)


####################
# Original Swimmer samples
# =========================

#fig, ax = plt.subplots(4, 4, figsize=(10,10), subplot_kw={'xticks': [], 'yticks': []})
fig, ax = plt.subplots(4, 4, subplot_kw={'xticks': [], 'yticks': []})
indices = np.random.randint(16, size=16) + np.arange(0, 256, 16)
ax = ax.flatten()
for i in range(len(indices)):
    ax[i].imshow(Swimmers[indices[i]].reshape(32,32).T, aspect='auto', cmap='gray')
fig.suptitle("Original Swimmer samples", fontsize='xx-large', verticalalignment='center')
ret = fig.text(.5, .05, "Features were translated and scaled to be between [0,1]", ha='center', fontsize=12)


####################
# Swimmer samples corrupted with Absolute Gaussian noise
# ======================================================
# Noise was randomly sampled from |N(0,0.25)|


reps = 1
corrupted = np.zeros((Swimmers.shape[0]*reps, Swimmers.shape[1]))

for r in range(reps):
    corrupted[r*Swimmers.shape[0]:(r+1)*Swimmers.shape[0]] = Swimmers + np.abs(np.random.normal(scale=0.25, size=Swimmers.shape))

fig, ax = plt.subplots(4, 4, figsize=(10,10), subplot_kw={'xticks': [], 'yticks': []})
ax = ax.flatten()
for i in range(len(indices)):
    ax[i].imshow(corrupted[indices[i]].reshape(32,32).T, aspect='auto', cmap='gray')
fig.suptitle("Corrupted Swimmer samples ", fontsize='xx-large', verticalalignment='center')
fig.text(.5, .05, "%d samples were corrupted with absolute Gaussian noise drawn from a Normal(0, 0.25)" % corrupted.shape[0], ha='center', fontsize=12)

nboot = 20
min_pts = nboot/2
ranks = [16]

shape = corrupted.shape

uoinmf = UoI_NMF(n_boots=nboot, ranks=ranks,
                db_min_samples=min_pts,
                nmf_max_iter=400,)

uoinmf.cons_meth = np.mean   # intersect and mean give too much noise

#transformed = None
#before = datetime.now()
#with warnings.catch_warnings(record=True) as w:
#    warnings.simplefilter('ignore', ConvergenceWarning)
#    print("Caught %d ConvergenceWarnings" % len(w))
#after = datetime.now()

transformed = uoinmf.fit_transform(corrupted)
recovered = transformed @ uoinmf.components_


####################
# NMF Swimmer bases
# =================

plt.figure()
order = np.argsort(np.sum(uoinmf.components_, axis=1))

nrow = 5 if uoinmf.components_.shape[0] > 16 else 4

fig, ax = plt.subplots(nrow,4, figsize=(10,10), subplot_kw={'xticks': [], 'yticks': []})
ax = ax.flatten()
for i in range(uoinmf.components_.shape[0]):
    ax[i].imshow(uoinmf.components_[order[i]].reshape(32,32).T, aspect='auto', cmap='gray')
fig.suptitle("Learned swimmer bases", fontsize='xx-large', verticalalignment='center')

caption = ("UoINMF was run with 20 bootstrap replicates.\n"
           "NMF bases were clustered using %s with minPts=%d and eps=0.5\n"
           "consensus bases were computing by taking the median of all members in each cluster")

caption = caption % ('DBSCAN', min_pts)

fig.text(.5, .05, caption, ha='center', fontsize=12)


####################
# Recovered Swimmers
# ==================


plt.figure()
fig, ax = plt.subplots(4, 4, figsize=(10,10), subplot_kw={'xticks': [], 'yticks': []})
ax = ax.flatten()
for i in range(len(indices)):
    ax[i].imshow(recovered[indices[i]].reshape(32,32).T, aspect='auto', cmap='gray')

ret = fig.suptitle("Recovered Swimmers", fontsize='xx-large', verticalalignment='center')

################################################################
# Plot them all together so we can see how well we recovered
# the original swimmer data.
#
#


plt.figure()
fig, ax = plt.subplots(3, 16, figsize=(80, 15), subplot_kw={'xticks': [], 'yticks': []})
indices = np.random.randint(16, size=16) + np.arange(0, 256, 16)
ax = ax.flatten()

# plot Original
ax[0].set_ylabel('Original', rotation=0, fontsize=50, labelpad=20)
ax[0].yaxis.set_label_coords(-0.5, 0.5)
for i in range(len(indices)):
    ax[i].imshow(Swimmers[indices[i]].reshape(32,32).T, aspect='auto', cmap='gray')

# plot Corrupted
ax[16].set_ylabel('Corrupted', rotation=0, fontsize=50, labelpad=20)
ax[16].yaxis.set_label_coords(-0.6, 0.5)
for i in range(len(indices)):
    ax[16+i].imshow(corrupted[indices[i]].reshape(32,32).T, aspect='auto', cmap='gray')

# plot Recovered
ax[32].set_ylabel('Recovered', rotation=0, fontsize=50, labelpad=20)
ax[32].yaxis.set_label_coords(-0.6, 0.5)
for i in range(len(indices)):
    ax[32+i].imshow(recovered[indices[i]].reshape(32,32).T, aspect='auto', cmap='gray')

ret = fig.suptitle("Original, corrupted, and recoverd Swimmers", fontsize=50, verticalalignment='center')


#plt.figure(figsize=(7,7))
embedding = TSNE(n_components=2).fit_transform(uoinmf.bases_samples_)
sc = plt.scatter(embedding[:,0], embedding[:,1], c=uoinmf.bases_samples_labels_, s=80, cmap="nipy_spectral")
sc.set_facecolor('none')

ret = plt.title("TSNE plot of Swimmer bases samples", fontsize='large', verticalalignment='bottom')

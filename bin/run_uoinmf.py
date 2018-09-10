from PyUoI.UoINMF import UoINMF

from sklearn.decomposition import NMF
from hdbscan import HDBSCAN
import h5py

from datetime import  datetime
import sys
import os
import argparse
from time import time
import logging

root = logging.getLogger()
root.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stderr)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

def log(msg):
    root.info(msg)

def h5_path(string):
    if ':' not in string:
        raise argparse.ArgumentTypeError("please provide a path to an HDF5 file and the path to the datast in the file")
    fpath, dset_path = string.split(":")
    ret = None
    with h5py.File(fpath, 'r') as f:
        ret = f[dset_path][:]
    return ret


parser = argparse.ArgumentParser(usage="%(prog)s [options] <hdf5_path>:<dataset_path> <output_hdf5_path>")
parser.add_argument('input', type=h5_path, help='the HDF5 file and dataset to use')
parser.add_argument('-b', '--bootstraps', type=int, help='the number of bootstraps to run', default=50)
parser.add_argument('-m', '--min_cluster_size', type=int, help='the minimum number samples for a cluster', default=20)
parser.add_argument('-s', '--seed', type=int, help='the seed to use for random number generation', default=-1)
parser.add_argument('-o', '--output', type=str, help='the HDF5 file to save results to', default='uoinmf.h5')
parser.add_argument('-n', '--no_norm', action='store_true', help='do not normalize data to [0,1]', default=False)
parser.add_argument('-f', '--force', action='store_true', help='force overwrite of output file', default=False)

args = parser.parse_args()

if os.path.exists(args.output):
    if not args.force:
        sys.stderr.write("%s exists... cautiously exiting\n" % args.output)
        sys.exit(2)
    else:
        os.remove(args.output)

output = args.output

seed = args.seed if args.seed >= 0 else int(round(time() * 1000) % 2**32)
log('using seed %s' % seed)

start = datetime.now()
#log('loading data')
data = args.input
log('normalizing data')

if not args.no_norm:
    data = data_normalization(data, 'positive')

uoinmf = UoINMF(n_bootstraps=args.bootstraps, ranks=list(range(2,20)), random_state=seed,
                    dbscan=HDBSCAN(min_cluster_size=args.min_cluster_size, core_dist_n_jobs=1))
log('running UoINMF')
log(repr(uoinmf))
W = uoinmf.fit_transform(data)

log('writing results to %s' % output)

f = h5py.File(output, 'w')
f.attrs['explanation'] = 'UoINMF decomposes nonnegative matrix A into the matrices W ("weights") and H ("bases") i.e. A = W*H'

dset = f.create_dataset('weights', data=W)
dset.attrs['explanation'] = 'The W component of the factorization'

dset = f.create_dataset('bases', data=uoinmf.components_)
dset.attrs['explanation'] = 'The H component of the factorization'

dset = f.create_dataset('bases_samples', data=uoinmf.bases_samples_)
dset.attrs['explanation'] = 'Samples for the rows of H. H is computed by building clusters from these samples'

dset = f.create_dataset('normalized_input', data=data)
dset.attrs['explanation'] = 'UoINMF (and all NMF algorithms) require the input matrix to be nonnegative. This requirement is ensured by normalizing all data to values between 0 and 1'
f.close()

end = datetime.now()
log('done - found %d bases' % uoinmf.components_.shape[0])
log('reconsruction error: %s' % uoinmf.reconstruction_err_)
log('time elapsed: %s' % str(end-start))

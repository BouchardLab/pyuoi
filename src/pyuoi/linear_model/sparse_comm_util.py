import numpy as np
import scipy.sparse as sp
from mpi4py import MPI

def send_sparse_matrix(comm, sparse_matrix, dest, tag=0):
    """Send a sparse matrix to a destination process."""
    # Extract the components of the sparse matrix
    data = sparse_matrix.data.astype(np.float64)  # Ensure consistent dtype
    indices = sparse_matrix.indices.astype(np.int32)
    indptr = sparse_matrix.indptr.astype(np.int32)
    shape = np.array(sparse_matrix.shape, dtype=np.int64)
    
    # Send shape first so destination knows what to expect
    comm.Send([shape, MPI.INT64_T], dest=dest, tag=tag)
    
    # Send data array size, then data
    data_size = np.array([data.size], dtype=np.int64)
    comm.Send([data_size, MPI.INT64_T], dest=dest, tag=tag+1)
    comm.Send([data, MPI.DOUBLE], dest=dest, tag=tag+2)
    
    # Send indices size, then indices
    indices_size = np.array([indices.size], dtype=np.int64)
    comm.Send([indices_size, MPI.INT64_T], dest=dest, tag=tag+3)
    comm.Send([indices, MPI.INT32_T], dest=dest, tag=tag+4)
    
    # Send indptr size, then indptr
    indptr_size = np.array([indptr.size], dtype=np.int64)
    comm.Send([indptr_size, MPI.INT64_T], dest=dest, tag=tag+5)
    comm.Send([indptr, MPI.INT32_T], dest=dest, tag=tag+6)

def recv_sparse_matrix(comm, source, tag=0):
    """Receive a sparse matrix from a source process."""
    # Receive shape first
    shape = np.empty(2, dtype=np.int64)
    comm.Recv([shape, MPI.INT64_T], source=source, tag=tag)
    
    # Receive data array size, then data
    data_size = np.empty(1, dtype=np.int64)
    comm.Recv([data_size, MPI.INT64_T], source=source, tag=tag+1)
    data = np.empty(data_size[0], dtype=np.float64)
    comm.Recv([data, MPI.DOUBLE], source=source, tag=tag+2)
    
    # Receive indices size, then indices
    indices_size = np.empty(1, dtype=np.int64)
    comm.Recv([indices_size, MPI.INT64_T], source=source, tag=tag+3)
    indices = np.empty(indices_size[0], dtype=np.int32)
    comm.Recv([indices, MPI.INT32_T], source=source, tag=tag+4)
    
    # Receive indptr size, then indptr
    indptr_size = np.empty(1, dtype=np.int64)
    comm.Recv([indptr_size, MPI.INT64_T], source=source, tag=tag+5)
    indptr = np.empty(indptr_size[0], dtype=np.int32)
    comm.Recv([indptr, MPI.INT32_T], source=source, tag=tag+6)
    
    # Reconstruct the sparse matrix
    return sp.csr_matrix((data, indices, indptr), shape=tuple(shape))



def build_bootstrap_comm(comm, n_admm):
    
    size = comm.Get_size()  # should be n_boot*n_admm (or better n_boot*n_reg*n_admm)

    boot_rank = np.arange(size)[::n_admm]
        
    boot_group = comm.group.Incl(list(boot_rank))
    # every rank in boot_rank will now have a different comm called boot_comm, the other ranks will have COMM_NULL for boot_comm
    boot_comm = comm.Create(boot_group)
    
    if boot_comm == MPI.COMM_NULL:
        boot_comm = None
    return boot_comm
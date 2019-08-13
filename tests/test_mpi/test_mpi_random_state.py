import pytest
import numpy as np
import warnings

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from pyuoi import UoI_Lasso
from pyuoi.mpi_utils import Gatherv_rows


@pytest.mark.skipif(MPI is None, reason='MPI is not installed.')
def test_random_state():
    """Test that the default random state assignment in UoI
    yields different random streams across MPI ranks"""

    # Explicitly passing in random_state
    comm = MPI.COMM_WORLD
    uoi = UoI_Lasso(comm=comm, random_state=5)
    rand_state = uoi.random_state.get_state()[1].astype(np.int)
    rand_state = rand_state[np.newaxis, :]
    rand_state = Gatherv_rows(rand_state, comm)
    if comm.rank == 0:
        if comm.size >= 2:
            assert(not np.array_equal(rand_state[0, :], rand_state[1, :]))

    # Default behavior
    uoi = UoI_Lasso(comm=comm)
    rand_state = uoi.random_state.get_state()[1].astype(np.int)
    rand_state = rand_state[np.newaxis, :]
    rand_state = Gatherv_rows(rand_state, comm)
    if comm.rank == 0:
        if comm.size >= 2:
            assert(not np.array_equal(rand_state[0, :], rand_state[1, :]))
        else:
            # Will be unable to properly test mpi in this case
            warnings.warn(UserWarning("Unable to properly test mpi with only"
                                      "1 process. Try again with mpiexec/srun"
                                      " -n 2"))

.. PyUoI

===
MPI
===

MPI (Message Passing Interface) is a parallel computing interface that can be
used through the ``mpi4py`` library in Python. Currently, the models in the
``linear_model`` module can take advantage of MPI parallelism during model
fitting. We assume some familiarity with using ``mpi4py`` here.

During the UoI feature selection step, many models are fit across bootstraps and
regularization parameters. These can all potentially be done in parallel using
MPI. Similarly, during UoI estimation, many models are fit across bootstraps
and supports. These can also be done in parallel.

Using MPI parallelism requires ``mpi4py`` to be installed. In your code, the
two extra things you will need to do to use MPI parallelism is 1) to make sure
the dataset is on all ranks and 2) pass an MPI communicator into the model.

Broadcasting the dataset to all ranks
-------------------------------------

PyUoI provides helper functions to share data across MPI ranks. The two
strategies we support are 1) load the data from a HDF5 file and 2) the user
can load the data on a single rank by hand and broadcast the data.

Loading data from an HDF5 file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from pyuoi.mpi_utils import load_data_MPI

    # file with keys 'X' and 'y'
    h5_file = 'my_file.h5'

    X, y = load_data_MPI(h5_file)

Loading data by hand
^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from mpi4py import MPI
    import numpy as np
    from pyuoi.mpi_utils import Bcast_from_root


    comm = MPI.COMM_WORLD
    rank = comm.rank

    X = None
    y = None
    if rank == 0:
        # file with keys 'X' and 'y'
        data = np.load('my_file.npz')
        X = data['X']
        y = data['y']

    X = Bcast_from_root(X, comm)
    y = Bcast_from_root(y, comm)

Fitting with MPI parallelism
-------------------------------------

Fitting models with MPI parallelism is similar to fitting models with no
parallelism.

.. code:: python

    from mpi4py import MPI
    from pyuoi.mpi_utils import load_data_MPI
    from pyuoi.linear_model import UoI_Lasso

    comm = MPI.COMM_WORLD
    rank = comm.rank

    # file with keys 'X' and 'y'
    h5_file = 'my_file.h5'

    X, y = load_data_MPI(h5_file)

    model = UoI_Lasso(comm=comm)
    model.fit(X, y)

    # model will now have fit parameters across all ranks

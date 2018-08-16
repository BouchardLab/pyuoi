# PyUoI

This repo contains several implementations of the Union of Intersection
(UoI) algorithm.

## Installation

A conda environment is recommended.

In order to use MPI, you need to install mpi4py. This is best done with
anaconda:

`conda install mpi4py`

Then download the source:

```
git clone https://github.com/bendichter/PyUoI.git
cd PyUoI
```

Install without MPI support:

`pip install -e .`

Install with MPI support:

`pip install -e .['MPI']`

# PyUoI

This repo contains several implementation of the Union of Intersection
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



## Usage

To run serial algorithm on sample data:

run `Model_Data_Create.py`

rename generated data file to `data.h5`

run `UoI_Lasso.py`


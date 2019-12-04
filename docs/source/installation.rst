.. PyUoI

============
Installation
============

PyUoI is available for Python 3 on PyPI:

.. code-block:: bash

    $ pip install pyuoi

and through conda-forge:

.. code-block:: bash

    $ conda install pyuoi -c conda-forge

``pip`` and ``conda`` will install the required dependencies.

Requirements
------------

Runtime
^^^^^^^

PyUoI requires

  * numpy>=1.14
  * h5py>=2.8
  * scikit-learn>=0.20

and optionally

  * pycasso
  * mpi4py

to run.

Develop
^^^^^^^

To develop PyUoI you will additionally need

  * cython

to build from source and

  * pytest
  * flake8

to run the tests and check formatting.

Docs
^^^^

To build the docs you will additionally need

  * sphinx
  * sphinx_rtd_theme

Install from source
-------------------

The latest development version of the code can be installed from https://github.com/BouchardLab/PyUoI

.. code-block:: bash

    # use ssh
    $ git clone git@github.com:BouchardLab/pyuoi.git
    # or use https
    $ git clone https://github.com/BouchardLab/pyuoi.git
    $ cd pyuoi
    $ pip install -e .[dev]

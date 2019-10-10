.. PyUoI

============
Installation
============

PyUoI is available on PyPI:

``
pip install pyuoi
``

and will soon be through conda-forge:

``
conda install pyuoi -c conda-forge
``

``pip`` and ``conda`` will install the required dependencies.

Requirements
------------

Runtime
^^^^^^^

PyUoI requires

  * numpy
  * h5py
  * scikit-learn

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

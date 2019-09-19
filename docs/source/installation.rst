.. PyUoI

============
Installation
============

PyUoI will be available soon on PyPI (installable with ``pip``)

.. code-block:: bash

    $ pip install pyuoi

and through conda-forge (installable with ``conda``).

.. code-block:: bash

    $ conda install -c conda-forge pyuoi

``pip`` and ``conda`` will install the required dependencies.

Requirements
------------

Runtime
^^^^^^^

PyUoI requires

  * numpy
  * h5py
  * scikit-learn
  * pycasso

and optionally

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
    $ git clone git@github.com:BouchardLab/PyUoI.git
    # or use https
    $ git clone https://github.com/BouchardLab/PyUoI.git
    $ cd PyUoI
    $ pip install -e .[dev]

.. PyUoI

==========================
How to contribute to PyUoI
==========================

Code of Conduct
---------------

Contributing Patches and Changes
--------------------------------

First, check whether the feature or change has already been contributed. If not, from your local copy directory, use the following commands.

If you have not already, you will need to clone the repo:

.. code-block:: bash

    $ git clone https://github.com/BouchardLab/PyUoI.git

1) First create a new branch to work on

.. code-block:: bash

    $ git checkout -b <new_branch>

2) Make your changes.

3) We will automatically run tests to ensure that your contributions didn't break anything and that they follow our style guide. You can speed up the testing cycle by running these tests locally on your own computer running ``pytest -sv tests``, ``flake8 pyuoi``, and ``flake8 tests``.

4) Push your feature branch to origin

.. code-block:: bash

    $ git push origin <new_branch>

5) Once you have tested and finalized your changes, create a pull request (PR):

    * Ensure the PR description clearly describes the issue and changes.
    * Close the relevant issue number if applicable. Writing "Closes #29" in the PR description will automatically close issue #29 when the PR is merged.
    * If your changes fix a bug or add a feature, write a test so that it will not break in the future.
    * Before submitting, please ensure that the tests pass and that the code follows the standard coding style.

Styleguides
-----------

Documentation Styleguide
^^^^^^^^^^^^^^^^^^^^^^^^

All documentations is written in reStructuredText (RST) using Sphinx.

Format Specification Styleguide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Python Code Styleguide
^^^^^^^^^^^^^^^^^^^^^^

Python coding style is checked via ``flake8`` for automatic checking of PEP8 style during pull requests.

License and Copyright
---------------------

PyUol Copyright (c) 2019, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy).  All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

(1) Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

(2) Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

(3) Neither the name of the University of California, Lawrence Berkeley National Laboratory, U.S. Dept. of Energy nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches, or upgrades to the features, functionality or performance of the source code ("Enhancements") to anyone; however, if you choose to make your Enhancements available either publicly, or directly to Lawrence Berkeley National Laboratory, without imposing a separate written license agreement for such Enhancements, then you hereby grant the following license: a  non-exclusive, royalty-free perpetual license to install, use, modify, prepare derivative works, incorporate into other computer software, distribute, and sublicense such enhancements or derivative works thereof, in binary and source code form.

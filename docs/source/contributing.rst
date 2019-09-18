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

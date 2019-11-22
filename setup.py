from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from distutils.ccompiler import get_default_compiler
# To use a consistent encoding
from codecs import open
from os import path

import numpy as np


here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()
with open(path.join(here, 'requirements-dev.txt'), encoding='utf-8') as f:
    dev_requirements = f.read().splitlines()
    dev_requirements = dev_requirements[1:] # Throw away the first line which is not a package.

# Prepare lbfgs
from Cython.Build import cythonize

class custom_build_ext(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        if self.compiler is None:
            compiler = get_default_compiler()
        else:
            compiler = self.compiler

        if compiler == 'msvc':
            include_dirs.append('compat/win32')

include_dirs = ['liblbfgs', np.get_include()]

ext_modules = cythonize(
    [Extension('pyuoi.lbfgs._lowlevel',
               ['pyuoi/lbfgs/_lowlevel.pyx', 'liblbfgs/lbfgs.c'],
               include_dirs=include_dirs)])


setup(
    name='pyuoi',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.99.0',

    description='The Union of Intersections framework in Python.',
    long_description=long_description,
    long_description_content_type="text/markdown",


    # Author details
    author='',
    author_email='',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: BSD License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    # What does your project relate to?
    keywords='UoI',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),
    package_data={'pyuoi': ['data/*']},

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=requirements,

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'perf': ['mpi4py', 'pycasso'],
        'dev': dev_requirements
    },

    url='https://github.com/BouchardLab/pyuoi',
    ext_modules=ext_modules,
    cmdclass={'build_ext': custom_build_ext}


    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    # entry_points={
    #    'console_scripts': [
    #        'sample=sample:main',
    #    ],
    # },
)

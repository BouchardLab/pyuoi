# podman-hpc build  -f ubu24-causal-net.dockerfile -t balewski/causal-net:p2a .
# on PM use 'podman-hpc' instead of 'podman' and all should work
# additionaly do 1 time:  podman-hpc migrate balewski/casual-net:p1k

FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /opt
ENV DEBIAN_FRONTEND noninteractive
ENV TZ=America/Los_Angeles

RUN \
    apt-get update        &&   \   
    apt-get install --yes      \   
        build-essential autoconf cmake flex bison zlib1g-dev   \   
        fftw-dev fftw3 apbs libicu-dev libbz2-dev libgmp-dev   \   
        bc libblas-dev liblapack-dev git libtool swig uuid-dev \
        libfftw3-dev automake lsb-core libxc-dev libgsl-dev    \   
        unzip libhdf5-serial-dev ffmpeg libcurl4-openssl-dev   \   
        libedit-dev libyaml-cpp-dev make libquadmath0 gfortran \
        python3-yaml automake pkg-config libc6-dev libzmq3-dev \
        libjansson-dev liblz4-dev libarchive-dev python3-pip   \   
        libsqlite3-dev lua5.1 liblua5.1-dev lua-posix jq opam  \   
        python3-dev python3-cffi python3-ply python3-sphinx    \   
        aspell aspell-en valgrind libyaml-cpp-dev wget vim     \   
        make libzmq3-dev python3-yaml time valgrind  libeigen3-dev \
        ocaml ocamlbuild ocaml-findlib indent libnum-ocaml libnum-ocaml-dev \
        fig2dev texinfo ghostscript                            \   
        mlocate python3-jsonschema python-is-python3         &&\ 
    apt-get clean all 

 
WORKDIR /opt
ARG mpich=4.2.2
ARG mpich_prefix=mpich-$mpich
RUN \
    wget https://www.mpich.org/static/downloads/$mpich/$mpich_prefix.tar.gz && \
    tar xvzf $mpich_prefix.tar.gz                                           && \
    cd $mpich_prefix                                                        && \
    ./configure FFLAGS=-fallow-argument-mismatch FCFLAGS=-fallow-argument-mismatch \
    --prefix=/opt/mpich/install                                             && \
    make -j 16                                                              && \
    make install                                                            && \
    make clean                                                              && \
    cd ..                                                                   && \
    rm -rf $mpich_prefix.tar.gz
ENV PATH=$PATH:/opt/mpich/install/bin
ENV PATH=$PATH:/opt/mpich/install/include
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/mpich/install/lib
RUN /sbin/ldconfig

ENV PATH=$PATH:/usr/local/cuda/lib64/stubs
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs
ENV PATH=$PATH:/usr/local/cuda-11.8/targets/x86_64-linux/lib/stubs
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/targets/x86_64-linux/lib/stubs

RUN ln -s /usr/local/cuda-11.8/targets/x86_64-linux/lib/stubs/libnvidia-ml.so /usr/local/cuda-11.8/targets/x86_64-linux/lib/stubs/libnvidia-ml.so.1 

#Install HWLOC
WORKDIR /opt 
RUN git clone -b v2.11 https://github.com/open-mpi/hwloc.git hwloc          && \
    cd hwloc                                                                && \
    ./autogen.sh                                                            && \
    ./configure                                                             && \
    make -j 16                                                              && \
    make install 


RUN pip install setuptools numpy
RUN python -m pip install mpi4py -i https://pypi.anaconda.org/mpi4py/simple
RUN pip install matplotlib pytest flake8 cython sphinx-gallery sphinx-rtd-theme
RUN pip install h5py scikit-learn

# Final cleanup
RUN apt-get clean
 
# Set the default command to bash
CMD ["/bin/bash"]

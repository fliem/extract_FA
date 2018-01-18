FROM bids/base_fsl:latest


# https://github.com/BIDS-Apps/MRtrix3_connectome/blob/master/Dockerfile
RUN apt-get update && apt-get install -y curl git perl-modules python software-properties-common tar unzip wget
# Now that we have software-properties-common, can use add-apt-repository to get to g++ version 5, which is required by JSON for Modern C++
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y g++-5

# Additional dependencies for MRtrix3 compilation
RUN apt-get install -y libeigen3-dev zlib1g-dev

# MRtrix3 setup
ENV CXX=/usr/bin/g++-5
# Note: Current commit being checked out includes various fixes that have been necessary to get test data working; eventually it will instead point to a release tag that includes these updates
RUN git clone https://github.com/MRtrix3/mrtrix3.git mrtrix3 && cd mrtrix3 && git checkout 2742e2f && python configure -nogui && NUMBER_OF_PROCESSORS=1 python build && git describe --tags > /mrtrix3_version
ENV PATH=/mrtrix3/bin:$PATH

RUN apt-get update && \
    apt-get install -y ants && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
ENV ANTSPATH=/usr/lib/ants/
ENV PATH=$ANTSPATH:$PATH

RUN rm -f `which eddy`
RUN mkdir /opt/eddy/
RUN wget -qO- https://fsl.fmrib.ox.ac.uk/fsldownloads/patches/eddy-patch-fsl-5.0.11/centos6/eddy_openmp > /opt/eddy/eddy_openmp
RUN chmod 775 /opt/eddy/eddy_openmp
ENV PATH=/opt/eddy/:$PATH

#### PYTHON / NIPYPE
RUN apt-get update && \
    apt-get install -y graphviz pandoc

WORKDIR /root

# Install anaconda
RUN echo 'export PATH=/usr/local/anaconda:$PATH' > /etc/profile.d/conda.sh && \
#    wget --quiet https://repo.continuum.io/archive/Anaconda2-5.0.1-Linux-x86_64.sh -O anaconda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O anaconda.sh && \
    /bin/bash anaconda.sh -b -p /usr/local/anaconda && \
    rm anaconda.sh

ENV PATH=/usr/local/anaconda/bin:$PATH

RUN ln -s /opt/eddy/eddy_openmp /opt/eddy/eddy

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Create conda environment, use nipype's conda-forge channel
RUN conda config --add channels conda-forge && \
    conda install -y lockfile nipype joblib nilearn

RUN pip install pybids
RUN conda install -y pandas

RUN mkdir /scratch
RUN mkdir -p /code
COPY extract_FA.py /code/extract_FA.py
RUN chmod +x /code/extract_FA.py
ENTRYPOINT ["/code/extract_FA.py"]

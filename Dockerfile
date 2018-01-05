FROM bids/base_fsl:latest



#### PYTHON / NIPYPE
RUN apt-get update && \
    apt-get install -y graphviz pandoc

WORKDIR /root

# Install anaconda
RUN echo 'export PATH=/usr/local/anaconda:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda2-5.0.1-Linux-x86_64.sh -O anaconda.sh && \
    /bin/bash anaconda.sh -b -p /usr/local/anaconda && \
    rm anaconda.sh

ENV PATH=/usr/local/anaconda/bin:$PATH

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Create conda environment, use nipype's conda-forge channel
RUN conda config --add channels conda-forge && \
    conda install -y lockfile nipype joblib nilearn

RUN pip install pybids


RUN mkdir /scratch
RUN mkdir -p /code
COPY extract_FA.py /code/extract_FA.py
RUN chmod +x /code/extract_FA.py
ENTRYPOINT ["/code/extract_FA.py"]

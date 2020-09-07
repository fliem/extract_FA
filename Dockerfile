FROM bids/base_fsl:latest


RUN apt-get update && apt-get install -y unzip wget

#RUN apt-get update && \
#    apt-get install -y g++ libeigen3-dev zlib1g-dev libqt4-opengl-dev libgl1-mesa-dev libfftw3-dev libtiff5-dev

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
RUN ln -s /opt/eddy/eddy_openmp /opt/eddy/eddy


WORKDIR /root

# Install anaconda
RUN echo 'export PATH=/usr/local/anaconda:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O anaconda.sh && \
    /bin/bash anaconda.sh -b -p /usr/local/anaconda && \
    rm anaconda.sh
ENV PATH=/usr/local/anaconda/bin:$PATH

RUN conda install -c mrtrix3 mrtrix3

# Create conda environment, use nipype's conda-forge channel
RUN conda config --add channels conda-forge && \
    conda install -y lockfile nipype joblib nilearn

RUN pip install pybids
RUN conda install -y pandas

RUN apt-get update && \
    apt-get install -y graphviz pandoc
    
RUN mkdir /scratch
RUN mkdir -p /code
COPY extract_FA.py /code/extract_FA.py
RUN chmod +x /code/extract_FA.py
ENTRYPOINT ["/code/extract_FA.py"]

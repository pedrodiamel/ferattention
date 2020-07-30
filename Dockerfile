FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
ARG PYTHON_VERSION=3.7
ARG WITH_TORCHVISION=1

RUN apt-get update && apt-get -y upgrade && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         vim \
         git \
         curl \
         ca-certificates \
         libgtk2.0-dev \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -L -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda100 && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

RUN /opt/conda/bin/conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

WORKDIR /opt
RUN git clone https://github.com/pedrodiamel/pytorchvision.git && cd pytorchvision && python setup.py install

RUN pip install --upgrade pip

RUN apt-get update -y && apt-get upgrade -y && apt-get install -y vim emacs nano htop

# WORKDIR /.datasets
# RUN chmod -R a+w .

WORKDIR /workspace
ADD requirements.txt .
RUN pip install -r requirements.txt
